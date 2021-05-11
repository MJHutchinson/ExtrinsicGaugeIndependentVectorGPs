# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %load_ext autoreload
# %autoreload 2
# %%
# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from tensorflow_probability.python.internal.backend import jax as tf2jax

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
import sys

from riemannianvectorgp.manifold import (
    S1,
    EmbeddedS1,
    ProductManifold,
    EmbeddedProductManifold,
)
from riemannianvectorgp.kernel import (
    ScaledKernel,
    ManifoldProjectionVectorKernel,
    MaternCompactRiemannianManifoldKernel,
    SquaredExponentialCompactRiemannianManifoldKernel,
    FourierFeatures,
)
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from jax.config import config
from einops import rearrange

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
import polyscope as ps

# %%
ps.init()

# %%
class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key


rng = GlobalRNG()


def mesh_to_polyscope(mesh, wrap_x=True, wrap_y=True):
    n, m, _ = mesh.shape

    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    coords = jj + m * ii

    faces = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):
            faces[i, j, 0] = coords[i, j]
            faces[i, j, 1] = coords[(i + 1) % n, j]
            faces[i, j, 2] = coords[(i + 1) % n, (j + 1) % m]
            faces[i, j, 3] = coords[i, (j + 1) % m]
            # faces[i, j, 0] = j + (i * n)
            # faces[i, j, 1] = ((j + 1) % m) + (i * n)
            # faces[i, j, 2] = ((j + 1) % m) + ((i + 1) % n) * n
            # faces[i, j, 3] = j + ((i + 1) % n) * n

    mesh_ = mesh.reshape(-1, 3)
    faces_ = faces.reshape(-1, 4)

    return mesh_, faces_


def project_to_3d(M, V, m_to_3d, projection_matrix_to_3d):
    X = m_to_3d(M)
    Y = (
        jnp.swapaxes(projection_matrix_to_3d(M), -1, -2) @ V[..., np.newaxis]
    ).squeeze()
    return X, Y


def t2_m_to_3d(M, R=3, r=1):
    theta1, theta2 = jnp.take(M, 0, -1), jnp.take(M, 1, -1)
    s1 = jnp.sin(theta1)
    c1 = jnp.cos(theta1)
    s2 = jnp.sin(theta2)
    c2 = jnp.cos(theta2)
    return jnp.stack([(R + r * c1) * c2, (R + r * c1) * s2, r * s1], axis=-1)


def t2_projection_matrix_to_3d(M, R=3, r=1):
    theta1, theta2 = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s1 = jnp.sin(theta1)
    c1 = jnp.cos(theta1)
    s2 = jnp.sin(theta2)
    c2 = jnp.cos(theta2)
    z = jnp.zeros_like(theta1)
    e1 = jnp.stack([-r * s1 * c2, -r * s1 * s2, r * c1], axis=-1)
    e2 = jnp.stack([-s2, c2, z], axis=-1)
    return jnp.stack(
        [
            e1,
            e2,
        ],
        axis=-2,
    )


# %%
n_points = 25
m = np.linspace(0, jnp.pi * 2, n_points + 1)[:-1] % (2 * jnp.pi)
m = jnp.meshgrid(m, m)
m = jnp.stack([m_.flatten() for m_ in m], axis=-1)
s1 = EmbeddedS1(0.5)  # produce base manifold
man = EmbeddedProductManifold(s1, s1, num_eigenfunctions=1000)
x_3d = t2_m_to_3d(m)
torus_mesh = ps.register_surface_mesh(
    "torus",
    *mesh_to_polyscope(x_3d.reshape((n_points, n_points, -1))),
    color=(0.9, 0.9, 0.9),
)
torus_mesh.set_vertex_tangent_basisX(
    t2_projection_matrix_to_3d(x_3d)[..., 0, :].reshape((-1, 3))
)
# %%
for i in range(250, 350):
    torus_mesh.add_scalar_quantity(
        f"ef {i}, ev {man.laplacian_eigenvalue(jnp.array([i]))}",
        man.laplacian_eigenfunction(jnp.array([i]), m)[:, 0, 0],
        enabled=False,
    )
ps.show()
# %%
s1 = EmbeddedS1(0.5)  # produce base manifold
man = EmbeddedProductManifold(s1, s1, num_eigenfunctions=1000)
kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(0.5, man, 1000))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0])
)
k = kernel.matrix(kernel_params, m, m)
# %%
s1 = EmbeddedS1(0.5)  # produce base manifold
man = s1 * s1  # make product manifold
kernel = ScaledKernel(
    MaternCompactRiemannianManifoldKernel(0.5, man, 100)
)  # make matern kernel with 100 basis functions
kernel_params = kernel.init_params(next(rng))  # init kernel params
k = kernel.matrix(kernel_params, m, m)  # compute kernel
# %%
i = 0
torus_mesh.add_scalar_quantity(f"kernel", k[:, i, 0, 0], enabled=False)
ps.register_point_cloud("point", x_3d[i][np.newaxis, :])
ps.show()
# %%

i = int(n_points ** 2 * 0.5)  # + int(n_points *0.5)

# plt.imshow(eig_func[:, i, 0].reshape((n_points, n_points)))
plt.contourf(
    m[:, 0].reshape((n_points, n_points)),
    m[:, 1].reshape((n_points, n_points)),
    k[:, i, 0, 0].reshape((n_points, n_points)),
)
plt.scatter(m[i, 0], m[i, 1])
plt.gca().set_aspect("equal")
plt.colorbar()
# %%
samples = 10
ff = FourierFeatures(kernel, 100)
ff_state = ff.init_state(kernel_params, samples, next(rng))
f = ff(kernel_params, ff_state, m)
# %%
for i in range(samples):
    torus_mesh.add_scalar_quantity(f"sample {i}", f[i, :, 0], enabled=False)
# %%
i = 3  # + int(n_points *0.5)

# plt.imshow(eig_func[:, i, 0].reshape((n_points, n_points)))
plt.contourf(
    m[:, 0].reshape((n_points, n_points)),
    m[:, 1].reshape((n_points, n_points)),
    f[i, :, 0].reshape((n_points, n_points)),
    50,
)
plt.scatter(m[i, 0], m[i, 1])
plt.gca().set_aspect("equal")
plt.colorbar()
# %%

# %%
s1 = EmbeddedS1(0.5)  # produce base manifold
man = EmbeddedProductManifold(s1, s1, num_eigenfunctions=1000)
kernel = ScaledKernel(
    ManifoldProjectionVectorKernel(
        MaternCompactRiemannianManifoldKernel(0.5, man, 1000), man
    )
)
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0])
)
k = kernel.matrix(kernel_params, m, m)
# %%
samples = 10
ff = FourierFeatures(kernel, 100)
ff_state = ff.init_state(kernel_params, samples, next(rng))
f = ff(kernel_params, ff_state, m)
# %%
i = 9
plt.quiver(m[:, 0], m[:, 1], f[i, :, 0], f[i, :, 1], jnp.linalg.norm(f[i], axis=-1))
plt.gca().set_aspect("equal")
# %%
for i in range(samples):
    torus_mesh.add_intrinsic_vector_quantity(
        f"prior sample {i}", f[i], defined_on="vertices"
    )

# %%
s1 = EmbeddedS1(0.5)  # produce base manifold
man = EmbeddedProductManifold(s1, s1, num_eigenfunctions=1000)
kernel = ScaledKernel(
    ManifoldProjectionVectorKernel(
        MaternCompactRiemannianManifoldKernel(0.5, man, 100), man
    )
)
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0])
)
k = kernel.matrix(kernel_params, m, m)
# %%

scale = 20

gp = GaussianProcess(kernel)
gp_params, gp_state = gp.init_params_with_state(next(rng))

n_cond = 10

m_cond_ind = jr.permutation(next(rng), jnp.arange(m.shape[0]))[:n_cond]

m_cond = m[m_cond_ind]
v_cond = jr.normal(next(rng), (n_cond, 2))
noises_cond = jnp.ones_like(v_cond) * 0.01

plt.quiver(
    m_cond[:, 0], m_cond[:, 1], v_cond[:, 0], v_cond[:, 1], color="red", scale=scale
)
plt.gca().set_aspect("equal")
plt.xlim(0, 2 * jnp.pi)
plt.ylim(0, 2 * jnp.pi)
# %%
gp_state = gp.condition(gp_params, m_cond, v_cond, noises_cond)

# %%
scale = 50
fig = plt.figure(figsize=(6, 6))
mean, K = gp(gp_params, gp_state, m)
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
    m[:, 0], m[:, 1], mean[:, 0], mean[:, 1], color="blue", scale=scale, zorder=4
)
plt.gca().set_aspect("equal")
plt.xlim(0, 2 * jnp.pi)
plt.ylim(0, 2 * jnp.pi)
# %%
samples = 10
obs_noise = 1e-3
# mean, K = gp(gp_params, gp_state, m)
# M, OD = mean.shape
# mean = rearrange(m, "M OD -> (M OD)")
# K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")
# cholesky = jsp.linalg.cho_factor(
#     K + jnp.identity(M * OD) * obs_noise, lower=True
# )[0]

posterior_samples = gp.sample(
    gp_params, gp_state, m, samples, next(rng), obs_noise=obs_noise
)
# %%
scale = 50
fig = plt.figure(figsize=(6, 6))
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
    m[:, 0], m[:, 1], mean[:, 0], mean[:, 1], color="blue", scale=scale, zorder=2
)

for i in range(samples):
    plt.quiver(
        m[:, 0],
        m[:, 1],
        posterior_samples[i, :, 0],
        posterior_samples[i, :, 1],
        color="black",
        alpha=0.3,
        scale=scale,
        zorder=1,
    )

plt.gca().set_aspect("equal")
plt.xlim(0, 2 * jnp.pi)
plt.ylim(0, 2 * jnp.pi)

# %%

length = 0.05

v_cond_ = jnp.zeros_like(mean)
v_cond_ = jax.ops.index_update(v_cond_, m_cond_ind, v_cond)

torus_mesh.add_intrinsic_vector_quantity(
    "cond", v_cond_, enabled=True, color=(1, 0, 0), length=length, radius=0.006
)

torus_mesh.add_intrinsic_vector_quantity(
    "mean",
    mean,
    enabled=True,
    color=(0, 0, 1),
    length=length,
    radius=0.004,
)
for i in range(samples):
    torus_mesh.add_intrinsic_vector_quantity(
        f"posterior sample {i}",
        posterior_samples[i],
        enabled=True,
        color=(0.2, 0.2, 0.2),
        length=length,
        radius=0.002,
    )
ps.show()

# %%
s1 = EmbeddedS1(0.5)  # produce base manifold
man = EmbeddedProductManifold(s1, s1, num_eigenfunctions=1000)
kernel = ScaledKernel(
    ManifoldProjectionVectorKernel(
        MaternCompactRiemannianManifoldKernel(0.5, man, 100), man
    )
)
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0])
)
k = kernel.matrix(kernel_params, m, m)
# %%

scale = 20
n_cond = 10
n_ind = 10

sparse_gp = SparseGaussianProcess(kernel, n_ind, 99, 100)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))

sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    jr.uniform(next(rng), (n_ind, 2)) * jnp.pi * 2,
    jr.normal(next(rng), (n_ind, 2)),
    jnp.ones((n_ind, 2)) * 0.1,
)


# m_cond_ind = jr.permutation(next(rng), jnp.arange(m.shape[0]))[:n_cond]
# m_cond = m[m_cond_ind]
# v_cond = jr.normal(next(rng), (n_cond, 2))
# noises_cond = jnp.ones_like(v_cond) * 0.01

sample = sparse_gp.prior(
    sparse_gp_params.kernel_params, sparse_gp_state.prior_state, m
)[3]

plt.quiver(m[:, 0], m[:, 1], sample[:, 0], sample[:, 1], color="blue", scale=scale)

inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)

plt.quiver(
    sparse_gp_params.inducing_locations[:, 0],
    sparse_gp_params.inducing_locations[:, 1],
    inducing_means[:, 0],
    inducing_means[:, 1],
    color="green",
    scale=scale,
)

inducing_noise = jnp.exp(sparse_gp_params.inducing_pseudo_log_err_stddev)

for i in range(inducing_noise.shape[0]):
    e = Ellipse(
        sparse_gp_params.inducing_locations[i],
        width=0.1 * scale * inducing_noise[i, 0],
        height=0.1 * scale * inducing_noise[i, 1],
        color="green",
        alpha=0.3,
    )
    plt.gca().add_artist(e)

plt.gca().set_aspect("equal")
plt.xlim(0, 2 * jnp.pi)
plt.ylim(0, 2 * jnp.pi)
# %%
