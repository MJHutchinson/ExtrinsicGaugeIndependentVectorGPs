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
    e2 = jnp.zeros_like(e1)
    return jnp.stack(
        [
            e1,
            # e2
        ],
        axis=-2,
    )


def cylinder_projection_matrix_to_3d_2(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    e1 = jnp.stack([c, -s, z], axis=-1)
    e2 = jnp.stack([z, z, o], axis=-1)
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
for i in range(4000, 4000 + n_basis):
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

num_basis_functions = 100
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
k_field = k[:, i] @ v
cyl_mesh.add_intrinsic_vector_quantity(
    "k_field_e1", k_field, enabled=True, color=(0, 1, 0)
)

v = jnp.array([0, 1])
k_field = k[:, i] @ v
cyl_mesh.add_intrinsic_vector_quantity(
    "k_field_e2", k_field, enabled=True, color=(0, 0, 1)
)

# %%
num_basis_functions = 100
num_samples = 2000
ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, next(rng))
f = ff(kernel_params, state, m)
cyl_mesh.add_intrinsic_vector_quantity(
    "prior sample", f[0], enabled=True, color=(1, 0, 0)
)
# %%
i = 1
v = f[i]
cyl_mesh.add_intrinsic_vector_quantity("function", v, enabled=True, color=(1, 0, 0))

from riemannianvectorgp.sparse_gp import SparseGaussianProcess

n_ind = 50

sparse_gp = SparseGaussianProcess(kernel, n_ind, 67, 20)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))
sparse_gp_params = sparse_gp_params._replace(kernel_params=kernel_params)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))
# %%
m_ind = jr.shuffle(next(rng), m)[:n_ind]
v_ind = jnp.zeros_like(m_ind)  # f(m_ind)
sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    m_ind,
    v_ind,
    jnp.ones_like(v_ind) * 0.01,
)
# %%
inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)
inducing_locs_, inducing_means_ = project_to_3d(
    sparse_gp_params.inducing_locations,
    inducing_means,
    cylinder_m_to_3d,
    cylinder_projection_matrix_to_3d_2,
)
inducing_cloud = ps.register_point_cloud(
    "inducing points", inducing_locs_, color=(0, 1, 0), enabled=True
)
inducing_cloud.add_vector_quantity(
    "inducing means", inducing_means_, color=(0, 1, 0), enabled=True, length=0.02
)

posterior_samples = sparse_gp(sparse_gp_params, sparse_gp_state, m)

mean = jnp.mean(posterior_samples, axis=0)
cyl_mesh.add_intrinsic_vector_quantity(
    "mean",
    mean,
    color=(0, 0, 1),
    length=0.02,
    enabled=True,
)
cyl_mesh.add_scalar_quantity(
    "mean_scalar",
    mean[:, 0],
)

for i in range(posterior_samples.shape[0]):
    cyl_mesh.add_intrinsic_vector_quantity(
        f"sample {i}",
        posterior_samples[i],
        color=(0.7, 0.7, 0.7),
        length=0.02,
        enabled=True,
        radius=0.001,
    )
# %%
opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.01))
opt_state = opt.init(sparse_gp_params)

# %%
debug_params = [sparse_gp_params]
debug_states = [sparse_gp_state]
debug_keys = [rng.key]

# %%
for i in range(300):
    ((train_loss, sparse_gp_state), grads) = jax.value_and_grad(
        sparse_gp.loss, has_aux=True
    )(sparse_gp_params, sparse_gp_state, next(rng), m, v, m.shape[0])
    (updates, opt_state) = opt.update(grads, opt_state)
    sparse_gp_params = optax.apply_updates(sparse_gp_params, updates)
    # if jnp.all(jnp.isnan(grads.kernel_params.sub_kernel_params.log_length_scale)):
    #     print("breaking for nan")
    #     break
    if i <= 10 or i % 20 == 0:
        print(i, "Loss:", train_loss)
    debug_params.append(sparse_gp_params)
    debug_states.append(sparse_gp_state)
    debug_keys.append(rng.key)

sparse_gp_params = debug_params[-1]
sparse_gp_state = debug_states[-1]
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
i = 1
x_, f_ = project_to_3d(m, f[i], cylinder_m_to_3d, cylinder_projection_matrix_to_3d_1)
cyl_mesh.add_vector_quantity("prior sample", f_, color=(1, 0, 0))
# %%

v = f[i]
x_, f_ = project_to_3d(m, f[i], cylinder_m_to_3d, cylinder_projection_matrix_to_3d_1)
cyl_mesh.add_vector_quantity("v", f_, color=(0, 0, 0))
cyl_mesh.add_scalar_quantity("v_scalar", f_[:, 0])

from riemannianvectorgp.sparse_gp import SparseGaussianProcess

n_ind = 50

sparse_gp = SparseGaussianProcess(kernel, n_ind, 67, 20)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))
sparse_gp_params = sparse_gp_params._replace(kernel_params=kernel_params)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))

# %%
m_ind = jr.shuffle(next(rng), m)[:n_ind]
v_ind = jnp.zeros_like(m_ind)[:, 0][:, np.newaxis]  # f(m_ind)
sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    m_ind,
    v_ind,
    jnp.ones_like(v_ind) * 0.01,
)
# %%
inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)
inducing_locs_, inducing_means_ = project_to_3d(
    sparse_gp_params.inducing_locations,
    inducing_means,
    cylinder_m_to_3d,
    cylinder_projection_matrix_to_3d_1,
)
inducing_cloud = ps.register_point_cloud(
    "inducing points", inducing_locs_, color=(0, 1, 0), enabled=True
)
inducing_cloud.add_vector_quantity(
    "inducing means", inducing_means_, color=(0, 1, 0), enabled=False, length=0.02
)

posterior_samples = sparse_gp(sparse_gp_params, sparse_gp_state, m)
posterior_samples = jnp.concatenate(
    [posterior_samples, jnp.zeros_like(posterior_samples)], axis=-1
)

mean = jnp.mean(posterior_samples, axis=0)
cyl_mesh.add_intrinsic_vector_quantity(
    "mean",
    mean,
    color=(0, 0, 1),
    length=0.02,
    enabled=False,
)
cyl_mesh.add_scalar_quantity(
    "mean_scalar",
    mean[:, 0],
)

for i in range(posterior_samples.shape[0]):
    cyl_mesh.add_intrinsic_vector_quantity(
        f"sample {i}",
        posterior_samples[i],
        color=(0.7, 0.7, 0.7),
        length=0.02,
        enabled=False,
    )

# %%
opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.01))
opt_state = opt.init(sparse_gp_params)

# %%
debug_params = [sparse_gp_params]
debug_states = [sparse_gp_state]
debug_keys = [rng.key]

# %%
for i in range(600):
    ((train_loss, sparse_gp_state), grads) = jax.value_and_grad(
        sparse_gp.loss, has_aux=True
    )(sparse_gp_params, sparse_gp_state, next(rng), m, v, m.shape[0])
    (updates, opt_state) = opt.update(grads, opt_state)
    sparse_gp_params = optax.apply_updates(sparse_gp_params, updates)
    # if jnp.all(jnp.isnan(grads.kernel_params.sub_kernel_params.log_length_scale)):
    #     print("breaking for nan")
    #     break
    if i <= 10 or i % 20 == 0:
        print(i, "Loss:", train_loss)
    debug_params.append(sparse_gp_params)
    debug_states.append(sparse_gp_state)
    debug_keys.append(rng.key)

sparse_gp_params = debug_params[-1]
sparse_gp_state = debug_states[-1]

# %%
sparse_gp_params = debug_params[-1]
sparse_gp_state = debug_states[-1]
# %%
sparse_gp.loss(sparse_gp_params, sparse_gp_state, next(rng), m, v, m.shape[0])[0]
# %%
