# %%
import numpy as np
import jax
import optax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import tensorflow_probability
import matplotlib.pyplot as plt
from jax.config import config
from IPython.display import set_matplotlib_formats
from tensorflow_probability.python.internal.backend import jax as tf2jax

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

set_matplotlib_formats("svg")

from riemannianvectorgp.utils import (
    GlobalRNG,
    mesh_to_polyscope,
    project_to_3d,
    cylinder_m_to_3d,
    cylinder_projection_matrix_to_3d,
    normalise_scaled_kernel,
    plot_scalar_field,
    plot_vector_field,
    plot_covariances,
    plot_mean_cov,
    plot_inference,
    plot_2d_sparse_gp,
    circle_distance,
)
from riemannianvectorgp.manifold import (
    S1,
    EmbeddedS1,
    EmbeddedR,
    ProductManifold,
    EmbeddedProductManifold,
)
from riemannianvectorgp.kernel import (
    FourierFeatures,
    ScaledKernel,
    TFPKernel,
    ManifoldProjectionVectorKernel,
    MaternCompactRiemannianManifoldKernel,
    SquaredExponentialCompactRiemannianManifoldKernel,
    ProductKernel,
)
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.sparse_gp import SparseGaussianProcess

from datasets.dynamical_systems import FrictionPendulumSystem, GPDynamicalSystem
from datasets.dynamical_systems_utils import (
    reverse_eular_integrate_rollouts,
    pendulum_statespace_kernel,
    pendulum_statespace_kernel_2,
    pendulum_statespace_euclidean_kernel,
    setup_sparse_gp_training,
    train_eular_integration_sparse_gp,
    train_trajectory_to_trajectory_sparse_gp,
    zero_grad_named_tuple,
)

rng = GlobalRNG()

# %% Set up the phase space grid
n_points = 15
positions = jnp.linspace(0, 2 * jnp.pi, n_points)
momentums = jnp.linspace(-3, 3, n_points)
positions, momentums = jnp.meshgrid(positions, momentums)
positions = positions.flatten()[:, np.newaxis]
momentums = momentums.flatten()[:, np.newaxis]
phase_space = m = jnp.concatenate([positions, momentums], axis=-1)
# %% Initialise a pendulum system with friction and get the dynamics field
system = FrictionPendulumSystem(mass=0.1, length=2.0, friction=0.03)
scale = 300
hgf = system.hamiltonian_gradient_field(positions, momentums)
ncf = system.non_conservative_field(positions, momentums)
dgf = system.dynamics_gradient_field(positions, momentums)
h = system.hamiltonian(positions, momentums)
# %% Plot the dynamice field
scale = 100
width = 0.004
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
plot_vector_field(phase_space, dgf, color="black", ax=ax, scale=scale, width=width)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")
plt.savefig("../figures/dynamics_statespace.pdf", bbox_inches="tight")

# %% rollout 2 trajectories to train on
initial_states = jnp.stack(
    [jnp.array([2.0, 3.0]), jnp.array([2.0, -3.0])],
    axis=0,
)
steps = 2000
rollouts = system.rollout(initial_states, steps)

# %% Plot the trainign data
scale = 100
width = 0.004
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
plot_vector_field(phase_space, dgf, ax=ax, color="black", scale=scale)
for i in range(rollouts.shape[0]):
    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")
plt.savefig("../figures/dynamics_training_data.pdf", bbox_inches="tight")

# %% compute reverse integrated trajectory data from the rollouts
scale = 400
m_cond, v_cond = reverse_eular_integrate_rollouts(
    rollouts, system, thinning=5, chuck_factor=4
)
m_cond_, v_cond_ = reverse_eular_integrate_rollouts(
    rollouts, system, thinning=5, chuck_factor=4, estimate_momentum=True
)

# %% Create a manifold kernel for the GP
kernel, kernel_params = pendulum_statespace_kernel_2(rng, 1.2, 0.3)
kernel_params = normalise_scaled_kernel(kernel, kernel_params, phase_space)

# %% Plot the kernel
k = kernel.matrix(kernel_params, phase_space, phase_space)

i = int(n_points ** 2 / 2)
plt.contourf(
    phase_space[:, 0].reshape(n_points, n_points),
    phase_space[:, 1].reshape(n_points, n_points),
    k[:, i, 0, 0].reshape(n_points, n_points),
    50,
)
plt.scatter(phase_space[i, 0], phase_space[i, 1])
plt.colorbar()
# %% Init a sparse GP with the kernel
n_ind = m_cond.shape[0]
n_samples = 10
sparse_gp = SparseGaussianProcess(kernel, n_ind, 67, n_samples)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))
sparse_gp_params = sparse_gp_params._replace(kernel_params=kernel_params)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))
# %% Set the inducing points of the spare GP to the data
ind_ind = jr.permutation(next(rng), jnp.arange(m_cond.shape[0]))[:n_ind]
m_ind = m_cond[ind_ind]
v_ind = jnp.zeros_like(m_ind)  # f(m_ind)
v_ind = v_cond[ind_ind]
noise_ind = jnp.ones_like(v_ind) * 0.5
sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    m_ind,
    v_ind,
    noise_ind,
)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))
sparse_gp_params = sparse_gp_params._replace(
    log_error_stddev=jnp.log(jnp.ones_like(sparse_gp_params.log_error_stddev) * 0.5)
)
# %% Create a GP dymanics system with the conditioned GP
gp_system = GPDynamicalSystem(sparse_gp, step_size=0.01)
# %% Rollout some test trajectories
initial_state = jnp.array([0 * jnp.pi, 0.01])[np.newaxis, :]
r = 2.5
n = 10
initial_states = jnp.stack(
    [jnp.array([(m + 1) * 2 * jnp.pi / (n + 1) - 0.01, 0.0]) for m in range(n)],
    axis=0,
)
steps = 400
gp_rollouts = gp_system.rollout(
    sparse_gp_params, sparse_gp_state, initial_states, steps
)
real_rollouts = system.rollout(initial_states, steps)
# %% Plot a sample of the rolledout trajectories
fig, ax = plt.subplots(1, 1, figsize=(2, 2))  # plot_scalar_field(phase_space, h, ax=ax)
plt.quiver(phase_space[:, 0], phase_space[:, 1], dgf[:, 0], dgf[:, 1])

for i in range(gp_rollouts.shape[0]):
    if i in [1, 3, 6, 8]:
        continue
    sc = plt.scatter(gp_rollouts[i, :, 0] % (2 * jnp.pi), gp_rollouts[i, :, 1], s=1)
    col = sc.get_facecolors()[0].tolist()
    col = [c * 0.5 for c in col[:-1]]
    sc = plt.scatter(
        initial_states[i, 0] % (2 * jnp.pi),
        initial_states[i, 1],
        s=30,
        marker="*",
        color=col,
    )
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")
plt.savefig("../figures/dynamics_manifold_statespace.pdf", bbox_inches="tight")
# %% Plot a sample of the rolledout trajectories
steps = 1000
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
for i in range(initial_states.shape[0]):
    if i in [1, 3, 6, 8]:
        continue
    print(i)
    init = jnp.repeat(initial_states[i][np.newaxis, :], n_samples, axis=0)
    ro = gp_system.rollout(sparse_gp_params, sparse_gp_state, init, steps)
    real = system.rollout(initial_states[i][np.newaxis, :], steps)

    ro = ((ro % (2 * jnp.pi)) + jnp.pi) % (2 * jnp.pi)
    real = ((real % (2 * jnp.pi)) + jnp.pi) % (2 * jnp.pi)

    ind = jnp.arange(steps)

    l = ax.plot(ind, real[0, :, 0])
    color = l[0].get_color()
    mean = ro[..., 0].mean(axis=0)
    std = ro[..., 0].std(axis=0)
    ax.plot(ind, mean, color=color, linestyle="--")
    ax.fill_between(ind, mean - std, mean + std, color=color, alpha=0.5, linewidth=0)

plt.xlabel("Step")
plt.ylabel("Position")
plt.ylim([0, 2 * np.pi])
plt.savefig("../figures/dynamics_manifold_rollouts.pdf", bbox_inches="tight")

# %% Create a Eucliden kernel for the state space
kernel, kernel_params = pendulum_statespace_euclidean_kernel(rng, 1.2, 1.2)
kernel_params = normalise_scaled_kernel(kernel, kernel_params, phase_space)
# %% Plot the kernel
k = kernel.matrix(kernel_params, phase_space, phase_space)

i = int(n_points ** 2 / 2)
plt.contourf(
    phase_space[:, 0].reshape(n_points, n_points),
    phase_space[:, 1].reshape(n_points, n_points),
    k[:, i, 0, 0].reshape(n_points, n_points),
    50,
)
plt.scatter(phase_space[i, 0], phase_space[i, 1])
plt.colorbar()
# %% set up a sparse GP
n_ind = m_cond.shape[0]
n_samples = 10
sparse_gp = SparseGaussianProcess(kernel, n_ind, 67, n_samples)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))
sparse_gp_params = sparse_gp_params._replace(kernel_params=kernel_params)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))
# %% set the inducing points to the data locations
ind_ind = jr.permutation(next(rng), jnp.arange(m_cond.shape[0]))[:n_ind]
m_ind = m_cond[ind_ind]
v_ind = jnp.zeros_like(m_ind)  # f(m_ind)
v_ind = v_cond[ind_ind]
noise_ind = jnp.ones_like(v_ind) * 0.5
sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    m_ind,
    v_ind,
    noise_ind,
)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))
sparse_gp_params = sparse_gp_params._replace(
    log_error_stddev=jnp.log(jnp.ones_like(sparse_gp_params.log_error_stddev) * 0.5)
)
# %% init a GP system
gp_system = GPDynamicalSystem(sparse_gp, step_size=0.01)
# %% Rollout test trajectories
n = 10
initial_states = jnp.stack(
    [jnp.array([(m + 1) * 2 * jnp.pi / (n + 1) - 0.01, 0.0]) for m in range(n)],
    axis=0,
)
steps = 400
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))
gp_rollouts = gp_system.rollout(
    sparse_gp_params, sparse_gp_state, initial_states, steps
)
real_rollouts = system.rollout(initial_states, steps)

# %% Plot trajectories
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
plt.quiver(phase_space[:, 0], phase_space[:, 1], dgf[:, 0], dgf[:, 1])

for i in range(gp_rollouts.shape[0]):
    if i in [1, 3, 6, 8]:
        continue
    sc = plt.scatter(gp_rollouts[i, :, 0] % (2 * jnp.pi), gp_rollouts[i, :, 1], s=1)
    col = sc.get_facecolors()[0].tolist()
    col = [c * 0.5 for c in col[:-1]]
    sc = plt.scatter(
        initial_states[i, 0] % (2 * jnp.pi),
        initial_states[i, 1],
        s=30,
        marker="*",
        color=col,
    )
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")
plt.savefig("../figures/dynamics_euclidean_statespace.pdf", bbox_inches="tight")
# %% Plot trajectories
steps = 1000
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
for i in range(initial_states.shape[0]):
    if i in [1, 3, 6, 8]:
        continue
    print(i)
    init = jnp.repeat(initial_states[i][np.newaxis, :], n_samples, axis=0)
    ro = gp_system.rollout(sparse_gp_params, sparse_gp_state, init, steps)
    real = system.rollout(initial_states[i][np.newaxis, :], steps)

    ro = ((ro % (2 * jnp.pi)) + jnp.pi) % (2 * jnp.pi)
    real = ((real % (2 * jnp.pi)) + jnp.pi) % (2 * jnp.pi)

    ind = jnp.arange(steps)

    l = ax.plot(ind, real[0, :, 0])
    color = l[0].get_color()
    mean = ro[..., 0].mean(axis=0)
    std = ro[..., 0].std(axis=0)
    ax.plot(ind, mean, color=color, linestyle="--")
    ax.fill_between(ind, mean - std, mean + std, color=color, alpha=0.5, linewidth=0)
plt.xlabel("Step")
plt.ylabel("Position")
plt.ylim([0, 2 * np.pi])
plt.savefig("../figures/dynamics_euclidean_rollouts.pdf", bbox_inches="tight")

# %%
