# %%
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from tensorflow_probability.python.internal.backend import jax as tf2jax
import optax
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

from riemannianvectorgp.utils import train_sparse_gp

set_matplotlib_formats("svg")
# import sys

from riemannianvectorgp.utils import (
    GlobalRNG,
    mesh_to_polyscope,
    project_to_3d,
    cylinder_m_to_3d,
    cylinder_projection_matrix_to_3d,
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

rng = GlobalRNG()

import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

from jax.config import config

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

from datasets.dynamical_systems import PendulumSystem, GPDynamicalSystem

import polyscope as ps

# %%

ps.init()

# %%
n_points = 25
positions = jnp.linspace(0, 2 * jnp.pi, n_points)
momentums = jnp.linspace(-3, 3, n_points)
positions, momentums = jnp.meshgrid(positions, momentums)
positions = positions.flatten()[:, np.newaxis]
momentums = momentums.flatten()[:, np.newaxis]
phase_space = m = jnp.concatenate([positions, momentums], axis=-1)

x_3d = cylinder_m_to_3d(phase_space)
cyl_mesh = ps.register_surface_mesh(
    "cylinder",
    *mesh_to_polyscope(x_3d.reshape((n_points, n_points, -1)), wrap_x=False),
    color=(0.9, 0.9, 0.9),
)
cyl_mesh.set_vertex_tangent_basisX(
    cylinder_projection_matrix_to_3d(phase_space)[..., 0, :].reshape((-1, 3))
)
# %%

system = PendulumSystem(mass=0.1, length=2.0)

hgf = system.hamiltonian_gradient_field(positions, momentums)
h = system.hamiltonian(positions, momentums)
cyl_mesh.add_intrinsic_vector_quantity("hgf", hgf, color=(1, 0, 0))
plt.quiver(phase_space[:, 0], phase_space[:, 1], hgf[:, 0], hgf[:, 1])
plt.gca().set_aspect("equal")
# %%

# initial_state = jnp.array([0 * jnp.pi, 0.01])[np.newaxis, :]
r = 2.5
n = 5
initial_states = jnp.stack(
    [jnp.array([jnp.pi + 0.01, -r + ((2 * r) / (n - 1)) * m]) for m in range(n)],
    axis=0,
)
initial_states_ = jnp.stack(
    [jnp.array([m * jnp.pi / n, 0.0]) for m in range(n)],
    axis=0,
)
initial_states = jnp.concatenate([initial_states, initial_states_], axis=0)
steps = 400
rollouts = system.rollout(initial_states, steps)

# %%
plt.contourf(
    positions[:, 0].reshape(n_points, n_points),
    momentums[:, 0].reshape(n_points, n_points),
    h.reshape(n_points, n_points),
    50,
)
plt.quiver(phase_space[:, 0], phase_space[:, 1], hgf[:, 0], hgf[:, 1])
plt.scatter(initial_states[:, 0], initial_states[:, 1])
for i in range(rollouts.shape[0]):
    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.xlabel("position")
plt.ylabel("momentum")
plt.gca().set_aspect("equal")
# %%


def rollouts_to_data(rollouts, thinning=1, estimate_momentum=False, chuck_factor=10):

    deltas = (rollouts[..., 1:, :] - rollouts[..., :-1, :]) / system.step_size
    rollouts = rollouts[..., :-1, :]

    deltas = deltas[..., ::thinning, :]
    rollouts = rollouts[..., ::thinning, :]

    rollouts = rollouts.reshape((-1, rollouts.shape[-1]))
    deltas = deltas.reshape((-1, rollouts.shape[-1]))
    # Sketchy, chuckout the big delats that are fake...
    delta_norm = jnp.linalg.norm(deltas, axis=-1)
    delta_mean = jnp.mean(jnp.linalg.norm(deltas, axis=-1))

    chuck_inds = delta_norm > delta_mean * chuck_factor

    return rollouts[~chuck_inds], deltas[~chuck_inds]


# %%
scale = 400
m_cond, v_cond = rollouts_to_data(rollouts, thinning=1, chuck_factor=4)
## HACK, not sure whats happening here to get field from trajectory
actuals = system.hamiltonian_gradient_field(
    m_cond[:, 0][:, np.newaxis], m_cond[:, 1][:, np.newaxis]
)
# v_cond = v_cond / (v_cond / actuals).mean()

plt.contourf(
    positions[:, 0].reshape(n_points, n_points),
    momentums[:, 0].reshape(n_points, n_points),
    h.reshape(n_points, n_points),
    50,
)
plt.quiver(
    phase_space[:, 0],
    phase_space[:, 1],
    hgf[:, 0],
    hgf[:, 1],
    color="black",
    scale=scale,
)
plt.quiver(
    m_cond[:, 0], m_cond[:, 1], v_cond[:, 0], v_cond[:, 1], color="blue", scale=scale
)
plt.xlabel("position")
plt.ylabel("momentum")
plt.gca().set_aspect("equal")
# %%
num_basis_functions = 100
num_samples = 20

s1 = EmbeddedS1(1.0)
r1 = EmbeddedR(1)
k_s1 = MaternCompactRiemannianManifoldKernel(1.5, s1, 100)
k_s1_params = k_s1.init_params(next(rng))
k_s1_params = k_s1_params._replace(log_length_scale=jnp.log(0.3))

k_r1 = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
k_r1_params = k_r1.init_params(next(rng))
k_r1_params = k_r1_params._replace(log_length_scales=jnp.log(2))

kernel = ProductKernel(k_s1, k_r1)
product_kernel_params = kernel.init_params(next(rng))
product_kernel_params = product_kernel_params._replace(
    sub_kernel_params=[k_s1_params, k_r1_params]
)

k_ = kernel.matrix(product_kernel_params, phase_space, phase_space)

kernel = ManifoldProjectionVectorKernel(kernel, s1 * r1)
manifold_kernel_params = kernel.init_params(next(rng))
manifold_kernel_params = product_kernel_params

kernel = ScaledKernel(kernel)
scaled_kernel_params = kernel.init_params(next(rng))
scaled_kernel_params = scaled_kernel_params._replace(
    sub_kernel_params=manifold_kernel_params
)
scaled_kernel_params = scaled_kernel_params._replace(
    log_amplitude=-jnp.log(
        kernel.matrix(scaled_kernel_params, phase_space, phase_space)[0, 0, 0, 0]
    )
)
kernel_params = scaled_kernel_params
k = kernel.matrix(kernel_params, phase_space, phase_space)

i = int(n_points ** 2 / 2)
plt.contourf(
    phase_space[:, 0].reshape(n_points, n_points),
    phase_space[:, 1].reshape(n_points, n_points),
    k_[:, i, 0, 0].reshape(n_points, n_points),
    50,
)
plt.scatter(phase_space[i, 0], phase_space[i, 1])
# %%
gp = GaussianProcess(kernel)
gp_params, gp_state = gp.init_params_with_state(next(rng))
gp_params = gp_params._replace(kernel_params=kernel_params)
gp_state = gp.condition(gp_params, m_cond, v_cond, jnp.ones_like(v_cond) * 0.01)
mean, K = gp(gp_params, gp_state, phase_space)
# %%
scale = 300
fig = plt.figure(figsize=(6, 6))
mean, K = gp(gp_params, gp_state, phase_space)
plt.quiver(
    phase_space[:, 0],
    phase_space[:, 1],
    hgf[:, 0],
    hgf[:, 1],
    color="black",
    scale=scale,
)

plt.quiver(
    m_cond[:, 0],
    m_cond[:, 1],
    v_cond[:, 0],
    v_cond[:, 1],
    color="red",
    scale=scale,
    zorder=5,
)
plt.quiver(
    phase_space[:, 0],
    phase_space[:, 1],
    mean[:, 0],
    mean[:, 1],
    color="blue",
    scale=scale,
    zorder=4,
)
# %%
samples = 10
obs_noise = 1e-1

posterior_samples = gp.sample(
    gp_params, gp_state, phase_space, samples, next(rng), obs_noise=obs_noise
)
# %%
scale = 300
fig = plt.figure(figsize=(6, 6))

plt.quiver(
    phase_space[:, 0],
    phase_space[:, 1],
    hgf[:, 0],
    hgf[:, 1],
    color="black",
    scale=scale,
)

plt.quiver(
    m_cond[:, 0],
    m_cond[:, 1],
    v_cond[:, 0],
    v_cond[:, 1],
    color="red",
    scale=scale,
    zorder=3,
)

plt.quiver(
    phase_space[:, 0],
    phase_space[:, 1],
    mean[:, 0],
    mean[:, 1],
    color="blue",
    scale=scale,
    zorder=2,
)

for i in range(samples):
    plt.quiver(
        phase_space[:, 0],
        phase_space[:, 1],
        posterior_samples[i, :, 0],
        posterior_samples[i, :, 1],
        color="grey",
        alpha=0.3,
        scale=scale,
        zorder=4,
    )
# %%
n_ind = 100
n_samples = 10
sparse_gp = SparseGaussianProcess(kernel, n_ind, 67, n_samples)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))
sparse_gp_params = sparse_gp_params._replace(kernel_params=kernel_params)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))
# %%
ind_ind = jr.permutation(next(rng), jnp.arange(m_cond.shape[0]))[:n_ind]
m_ind = m_cond[ind_ind]
v_ind = jnp.zeros_like(m_ind)  # f(m_ind)
v_ind = v_cond[ind_ind]
noise_ind = jnp.ones_like(v_ind) * 0.01
sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    m_ind,
    v_ind,
    noise_ind,
)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))

# %%
gp_system = GPDynamicalSystem(sparse_gp)
r = 20
initial_states = jnp.stack(
    [jnp.array([2.0, -r + ((2 * r) / (n_samples - 1)) * m]) for m in range(n_samples)],
    axis=0,
)
steps = 1000
real_rollouts = system.rollout(initial_states, steps)
# %%
opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.003))
opt_state = opt.init(sparse_gp_params)
debug_params = [sparse_gp_params]
debug_states = [sparse_gp_state]
debug_keys = [rng.key]
losses = []
for i in range(300):
    ((train_loss, sparse_gp_state), grads) = jax.value_and_grad(
        sparse_gp.loss, has_aux=True
    )(sparse_gp_params, sparse_gp_state, next(rng), m_cond, v_cond, m_cond.shape[0])
    (updates, opt_state) = opt.update(grads, opt_state)
    sparse_gp_params = optax.apply_updates(sparse_gp_params, updates)
    # if jnp.all(jnp.isnan(grads.kernel_params.sub_kernel_params.log_length_scale)):
    #     print("breaking for nan")
    #     break
    if i <= 10 or i % 20 == 0:
        print(i, "Loss:", train_loss)
    losses.append(train_loss)
    debug_params.append(sparse_gp_params)
    debug_states.append(sparse_gp_state)
    debug_keys.append(rng.key)


# %%

sparse_gp_params, sparse_gp_state, debug = train_sparse_gp(
    sparse_gp, sparse_gp_params, sparse_gp_state, m_cond, v_cond, rng
)

# %%
scale = 300
plt.quiver(
    m_cond[:, 0],
    m_cond[:, 1],
    v_cond[:, 0],
    v_cond[:, 1],
    color="red",
    scale=scale,
    zorder=3,
)


inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)
plt.quiver(
    sparse_gp_params.inducing_locations[:, 0],
    sparse_gp_params.inducing_locations[:, 1],
    inducing_means[:, 0],
    inducing_means[:, 1],
    color="green",
    scale=scale,
    zorder=4,
)

plt.quiver(
    phase_space[:, 0],
    phase_space[:, 1],
    hgf[:, 0],
    hgf[:, 1],
    color="black",
    scale=scale,
    zorder=2,
)


posterior_samples = sparse_gp(sparse_gp_params, sparse_gp_state, phase_space)

for i in range(posterior_samples.shape[0]):
    plt.quiver(
        phase_space[:, 0],
        phase_space[:, 1],
        posterior_samples[i, :, 0],
        posterior_samples[i, :, 1],
        color="grey",
        scale=scale,
        zorder=1,
    )
plt.xlabel("position")
plt.ylabel("momentum")
plt.gca().set_aspect("equal")
# %%
gp_system = GPDynamicalSystem(sparse_gp)
# %%
r = 10
# initial_states = jnp.stack(
#     [jnp.array([2.0, -r + ((2 * r) / (n_samples - 1)) * m]) for m in range(n_samples)],
#     axis=0,
# )
initial_states = jnp.stack(
    [
        jr.uniform(next(rng), (n_samples,), minval=0, maxval=2 * jnp.pi),
        jr.uniform(next(rng), (n_samples,), minval=-2.5, maxval=2.5),
    ],
    axis=1,
)
steps = 1000
gp_rollouts = gp_system.rollout(
    sparse_gp_params, sparse_gp_state, initial_states, steps
)
real_rollouts = system.rollout(initial_states, steps)

# %%
plt.contourf(
    positions[:, 0].reshape(n_points, n_points),
    momentums[:, 0].reshape(n_points, n_points),
    h.reshape(n_points, n_points),
    50,
)
plt.quiver(phase_space[:, 0], phase_space[:, 1], hgf[:, 0], hgf[:, 1])

for i in range(gp_rollouts.shape[0]):
    sc = plt.scatter(gp_rollouts[i, :, 0] % (2 * jnp.pi), gp_rollouts[i, :, 1], s=1)
    col = sc.get_facecolors()[0].tolist()
    # print(col)
    col = [c * 0.5 for c in col[:-1]]
    # print(col)
    sc = plt.scatter(
        initial_states[i, 0] % (2 * jnp.pi),
        initial_states[i, 1],
        s=30,
        marker="*",
        color=col,
    )
plt.xlabel("position")
plt.ylabel("momentum")
# %%
i = 4
plt.contourf(
    positions[:, 0].reshape(n_points, n_points),
    momentums[:, 0].reshape(n_points, n_points),
    h.reshape(n_points, n_points),
    50,
)
plt.quiver(phase_space[:, 0], phase_space[:, 1], hgf[:, 0], hgf[:, 1])

plt.scatter(gp_rollouts[i, :, 0] % (2 * jnp.pi), gp_rollouts[i, :, 1], s=1, c="orange")
plt.scatter(
    real_rollouts[i, :, 0] % (2 * jnp.pi), real_rollouts[i, :, 1], s=1, c="blue"
)
plt.xlabel("position")
plt.ylabel("momentum")
# %%
i = 5
l = plt.plot((gp_rollouts[i, ..., 0] - jnp.pi) % (2 * jnp.pi), linestyle="--")
plt.plot((real_rollouts[i, ..., 0] - jnp.pi) % (2 * jnp.pi), color=l[0].get_color())
# %%
