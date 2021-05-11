# %%
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from tensorflow_probability.python.internal.backend import jax as tf2jax

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
import sys

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
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

from jax.config import config


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


def cylinder_m_to_3d(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    return jnp.stack([s, c, x], axis=-1)


def cylinder_projection_matrix_to_3d_1(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    z = jnp.zeros_like(c)
    e1 = jnp.stack([c, -s, z], axis=-1)
    return jnp.stack(
        [
            e1,
        ],
        axis=-2,
    )


def cylinder_projection_matrix_to_3d_2(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
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

n_points = 50
r = jnp.linspace(-5, 5, n_points)
m = jnp.linspace(0, 2 * jnp.pi, n_points + 1)[:-1]
r, m = jnp.meshgrid(r, m)
m = jnp.stack([m.flatten(), r.flatten()], axis=-1)

x_3d = cylinder_m_to_3d(m)
cyl_mesh = ps.register_surface_mesh(
    "cylinder",
    *mesh_to_polyscope(x_3d.reshape((n_points, n_points, -1)), wrap_y=False),
    color=(0.9, 0.9, 0.9),
)
cyl_mesh.set_vertex_tangent_basisX(
    cylinder_projection_matrix_to_3d_1(x_3d)[..., 0, :].reshape((-1, 3))
)
# %%

num_basis_functions = 1000
num_samples = 2000

s1 = S1(1.0)
k_s1 = MaternCompactRiemannianManifoldKernel(1.5, s1, num_basis_functions)
k_s1_params = k_s1.init_params(next(rng))
k_s1_params = k_s1_params._replace(log_length_scale=jnp.log(0.3))

k_r1 = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
k_r1_params = k_r1.init_params(next(rng))

kernel = ProductKernel(k_s1, k_r1)
product_kernel_params = kernel.init_params(next(rng))
product_kernel_params = product_kernel_params._replace(
    sub_kernel_params=[k_s1_params, k_r1_params]
)

kernel = ScaledKernel(kernel)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(sub_kernel_params=product_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0])
)
k = kernel.matrix(kernel_params, m, m)

# %%
num_basis_functions = 1000
num_samples = 2000
ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, next(rng))
f = ff(kernel_params, state, m)

m_ff = jnp.mean(f, axis=0)
k_ff = jnp.mean(
    f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0
)

# %%
k_err = k - k_ff
mean_err = jnp.mean(k_err)
max_err = jnp.max(jnp.abs(k_err))

print(mean_err, max_err)

# %%
n_basis = 99
basis_state = kernel.sample_fourier_features(kernel_params, next(rng), n_basis)
basis_funcs = kernel.basis_functions(kernel_params, basis_state, m)
for i in range(n_basis):
    cyl_mesh.add_scalar_quantity(
        f"ef {i}",
        basis_funcs[:, i, 0, 0],
        enabled=False,
    )
ps.show()
# %%
i = n_points * int(n_points * 0) + int(n_points / 2)
cyl_mesh.add_scalar_quantity(
    "kernel",
    k[i, :, 0, 0],
    enabled=True,
)
ps.register_point_cloud("point", x_3d[i][np.newaxis, :])
cyl_mesh.add_scalar_quantity(
    "kernel_fourier",
    k_ff[i, :, 0, 0],
    enabled=True,
)
# %%

num_basis_functions = 1000
num_samples = 2000

s1 = EmbeddedS1(1.0)
r1 = EmbeddedR(1)
k_s1 = MaternCompactRiemannianManifoldKernel(1.5, s1, 500)
k_s1_params = k_s1.init_params(next(rng))
k_s1_params = k_s1_params._replace(log_length_scale=jnp.log(0.3))

k_r1 = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
k_r1_params = k_r1.init_params(next(rng))

kernel = ProductKernel(k_s1, k_r1)
product_kernel_params = kernel.init_params(next(rng))
product_kernel_params = product_kernel_params._replace(
    sub_kernel_params=[k_s1_params, k_r1_params]
)

kernel = ManifoldProjectionVectorKernel(kernel, s1 * r1)
manifold_kernel_params = kernel.init_params(next(rng))
manifold_kernel_params = product_kernel_params

kernel = ScaledKernel(kernel)
scaled_kernel_params = kernel.init_params(next(rng))
scaled_kernel_params = scaled_kernel_params._replace(
    sub_kernel_params=manifold_kernel_params
)
scaled_kernel_params = scaled_kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(scaled_kernel_params, m, m)[0, 0, 0, 0])
)
kernel_params = scaled_kernel_params
k = kernel.matrix(kernel_params, m, m)

# %%
i = n_points * int(n_points * 0) + int(n_points / 2)
v = jnp.array([1, 0])
k_field = k[i, :] @ v
cyl_mesh.add_intrinsic_vector_quantity(
    "k_field_e1", k_field, enabled=True, color=(0, 1, 0)
)

v = jnp.array([0, 1])
k_field = k[i, :] @ v
cyl_mesh.add_intrinsic_vector_quantity(
    "k_field_e2", k_field, enabled=True, color=(0, 0, 1)
)
# %%

num_basis_functions = 1000
num_samples = 2000

s1 = EmbeddedS1(1.0)
r1 = EmbeddedR(1)
k_s1 = MaternCompactRiemannianManifoldKernel(1.5, s1, 1000)
k_s1_params = k_s1.init_params(next(rng))
k_s1_params = k_s1_params._replace(log_length_scale=jnp.log(0.3))

kernel = ManifoldProjectionVectorKernel(k_s1, s1)
manifold_kernel_params = kernel.init_params(next(rng))
manifold_kernel_params = k_s1_params

k_r1 = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
k_r1_params = k_r1.init_params(next(rng))

kernel = ProductKernel(kernel, k_r1)
product_kernel_params = kernel.init_params(next(rng))
product_kernel_params = product_kernel_params._replace(
    sub_kernel_params=[manifold_kernel_params, k_r1_params]
)

kernel = ScaledKernel(kernel)
scaled_kernel_params = kernel.init_params(next(rng))
scaled_kernel_params = scaled_kernel_params._replace(
    sub_kernel_params=product_kernel_params
)
scaled_kernel_params = scaled_kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(scaled_kernel_params, m, m)[0, 0, 0, 0])
)
kernel_params = scaled_kernel_params
k = kernel.matrix(kernel_params, m, m)

# %%
num_basis_functions = 100
num_samples = 2000
ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, next(rng))
f = ff(kernel_params, state, m)

m_ff = jnp.mean(f, axis=0)
k_ff = jnp.mean(
    f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0
)
# %%
