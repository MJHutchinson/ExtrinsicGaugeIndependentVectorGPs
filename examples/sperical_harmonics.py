# %%
import math

import torch
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import optax

from tensorflow_probability.python.internal.backend import jax as tf2jax
import jax
from jax import jit
from functools import partial

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


# %%
_Jd, _W3j_flat, _W3j_indices = torch.load("../constants.pt")
_Jd = [x.numpy() for x in _Jd]
_W3j_flat = _W3j_flat.numpy()

# %%
def wigner_3j(l1, l2, l3, flat_src=_W3j_flat, dtype=None, device=None):
    r"""Wigner 3j symbols :math:`C_{lmn}`.
    It satisfies the following two properties:
        .. math::
            C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)
        where :math:`D` are given by `wigner_D`.
        .. math::
            C_{ijk} C_{ijk} = 1
    Parameters
    ----------
    l1 : int
        :math:`l_1`
    l2 : int
        :math:`l_2`
    l3 : int
        :math:`l_3`
    dtype : torch.dtype or None
        ``dtype`` of the returned tensor. If ``None`` then set to ``torch.get_default_dtype()``.
    device : torch.device or None
        ``device`` of the returned tensor. If ``None`` then set to the default device of the current context.
    Returns
    -------
    `torch.Tensor`
        tensor :math:`C` of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`
    """
    assert abs(l2 - l3) <= l1 <= l2 + l3

    try:
        if l1 <= l2 <= l3:
            out = flat_src[_W3j_indices[(l1, l2, l3)]].reshape(
                2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
            )
        if l1 <= l3 <= l2:
            out = (
                jnp.swapaxes(
                    flat_src[_W3j_indices[(l1, l3, l2)]].reshape(
                        2 * l1 + 1, 2 * l3 + 1, 2 * l2 + 1
                    ),
                    1,
                    2,
                )
                * (-1) ** (l1 + l2 + l3)
            )
        if l2 <= l1 <= l3:
            out = (
                jnp.swapaxes(
                    flat_src[_W3j_indices[(l2, l1, l3)]].reshape(
                        2 * l2 + 1, 2 * l1 + 1, 2 * l3 + 1
                    ),
                    0,
                    1,
                )
                * (-1) ** (l1 + l2 + l3)
            )
        if l3 <= l2 <= l1:
            out = (
                jnp.swapaxes(
                    flat_src[_W3j_indices[(l3, l2, l1)]].reshape(
                        2 * l3 + 1, 2 * l2 + 1, 2 * l1 + 1
                    ),
                    0,
                    2,
                )
                * (-1) ** (l1 + l2 + l3)
            )
        if l2 <= l3 <= l1:
            out = jnp.swapaxes(
                jnp.swapaxes(
                    flat_src[_W3j_indices[(l2, l3, l1)]].reshape(
                        2 * l2 + 1, 2 * l3 + 1, 2 * l1 + 1
                    ),
                    0,
                    2,
                ),
                1,
                2,
            )
        if l3 <= l1 <= l2:
            out = jnp.swapaxes(
                jnp.swapaxes(
                    flat_src[_W3j_indices[(l3, l1, l2)]].reshape(
                        2 * l3 + 1, 2 * l1 + 1, 2 * l2 + 1
                    ),
                    0,
                    2,
                ),
                0,
                1,
            )
    except KeyError:
        raise NotImplementedError(
            f"Wigner 3j symbols maximum l implemented is {max(_W3j_indices.keys())[0]}, send us an email to ask for more"
        )

    return out


# %%
from riemannianvectorgp.utils import _spherical_harmonics

# %%
from riemannianvectorgp.manifold import EmbeddedS2

s2 = EmbeddedS2(1.0)

n = 30
phi = jnp.linspace(0, jnp.pi, n)
theta = jnp.linspace(0, 2 * jnp.pi, n + 1)[1:]
phi, theta = jnp.meshgrid(phi, theta)
phi = phi.flatten()
theta = theta.flatten()

x = jnp.cos(theta) * jnp.sin(phi)
y = jnp.sin(theta) * jnp.sin(phi)
z = jnp.cos(phi)

mesh = jnp.stack([x, y, z], axis=-1).reshape((n, n, 3))

sphere_mesh = ps.register_surface_mesh(
    "Sphere", *mesh_to_polyscope(mesh, wrap_y=False), color=(0.8, 0.8, 0.8)
)
sphere_mesh.set_vertex_tangent_basisX(s2.projection_matrix(m)[..., 0])

mesh_flat = jnp.stack([x, y, z], axis=-1)

m = jnp.stack([phi, theta], axis=-1)
# %%
sh = _spherical_harmonics(11, mesh_flat)
for i in range(sh.shape[-1]):
    sphere_mesh.add_scalar_quantity(f"Harmonic {i}", sh[:, i])

# %%


@partial(jit)
def spherical_laplacian_eigenfunctions(m, n):
    phi = m[..., 0]
    theta = m[..., 1]

    x = jnp.cos(theta) * jnp.sin(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(phi)

    e = jnp.stack([x, y, z], axis=-1)

    # l = int(math.ceil(math.sqrt(n))) + 1

    sh = _spherical_harmonics(11, e)

    return sh[..., n, np.newaxis]


# %%
selected_eigenfunctions = spherical_laplacian_eigenfunctions(m, jnp.arange(144))

for i in range(sh.shape[-1]):
    sphere_mesh.add_scalar_quantity(f"Harmonic {i}", selected_eigenfunctions[:, i, 0])

# %%
# phi = m[..., 0]
# theta = m[..., 1]

x = jnp.cos(theta) * jnp.sin(phi)
y = jnp.sin(theta) * jnp.sin(phi)
z = jnp.cos(phi)

e = jnp.stack([x, y, z], axis=-1)

# l = int(math.ceil(math.sqrt(n))) + 1

sh = _spherical_harmonics(11, e)

for i in range(sh.shape[-1]):
    sphere_mesh.add_scalar_quantity(f"Harmonic {i}", sh[:, i])

# %%

from riemannianvectorgp.kernel import (
    SquaredExponentialCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
)

# %%

scalar_kernel = SquaredExponentialCompactRiemannianManifoldKernel(s2, 144)
kernel = ManifoldProjectionVectorKernel(scalar_kernel, s2)
kernel_params = kernel.init_params(next(rng))
k_scalar = scalar_kernel.matrix(kernel_params, m, m)
k = kernel.matrix(kernel_params, m, m)

# %%

i = int(n ** 2 / 2)

ps.register_point_cloud("point", mesh_flat[i][np.newaxis, :])
# sphere_mesh.add_scalar_quantity(f"kernel", k[i, :, 0, 0])
v = jnp.array([1, 0])
_, v_proj = s2.project_to_e(m[i], v)
k_field = k[:, i] @ v
_, k_field_proj = s2.project_to_e(m, k_field)
sphere_mesh.add_intrinsic_vector_quantity(
    "k_field", k_field, enabled=True, color=(0, 1, 0)
)
v_field = jnp.zeros_like(k_field)
v_field = jax.ops.index_update(v_field, i, v)
sphere_mesh.add_intrinsic_vector_quantity("v", v_field, enabled=True, color=(1, 0, 0))
transported_v_field = jnp.zeros_like(k_field_proj)
transported_v_field = jax.ops.index_update(transported_v_field, (..., 0), v_proj[0])
transported_v_field = jax.ops.index_update(transported_v_field, (..., 1), v_proj[1])
transported_v_field = jax.ops.index_update(transported_v_field, (..., 2), v_proj[2])
transported_v_field = k_scalar[i, :, 0, :] * transported_v_field
sphere_mesh.add_vector_quantity(
    "transported v", transported_v_field, color=(1, 0, 0), enabled=True
)
_, k_proj = s2.project_to_m(s2.m_to_e(m), transported_v_field)
sphere_mesh.add_intrinsic_vector_quantity(
    "k_proj", k_proj, enabled=True, color=(1, 1, 0)
)


# v = jnp.array([0, 1])
# _, v_proj = s2.project_to_e(m[i], v)
# k_field = k[:, i] @ v
# _, k_field_proj = s2.project_to_e(m, k_field)
# sphere_mesh.add_intrinsic_vector_quantity(
#     "k_field", k_field, enabled=True, color=(0, 1, 0)
# )
# v_field = jnp.zeros_like(k_field)
# v_field = jax.ops.index_update(v_field, i, v)
# sphere_mesh.add_intrinsic_vector_quantity("v", v_field, enabled=True, color=(1, 0, 0))
# transported_v_field = jnp.zeros_like(k_field_proj)
# transported_v_field = jax.ops.index_update(transported_v_field, (..., 0), v_proj[0])
# transported_v_field = jax.ops.index_update(transported_v_field, (..., 1), v_proj[1])
# transported_v_field = jax.ops.index_update(transported_v_field, (..., 2), v_proj[2])
# transported_v_field = k_scalar[i, :, 0, :] * transported_v_field
# sphere_mesh.add_vector_quantity(
#     "transported v", transported_v_field, color=(1, 0, 0), enabled=True
# )
# _, k_proj = s2.project_to_m(s2.m_to_e(m), transported_v_field)
# sphere_mesh.add_intrinsic_vector_quantity(
#     "k_proj", k_proj, enabled=True, color=(1, 1, 0)
# )

# %%
scalar_kernel = SquaredExponentialCompactRiemannianManifoldKernel(s2, 144)
sub_kernel = ManifoldProjectionVectorKernel(scalar_kernel, s2)
sub_kernel_params = sub_kernel.init_params(next(rng))
kernel = ScaledKernel(sub_kernel)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(
    sub_kernel_params=sub_kernel_params,
    log_amplitude=-jnp.log(sub_kernel.matrix(sub_kernel_params, m, m)[0, 0, 0, 0]),
)

# %%
from riemannianvectorgp.sparse_gp import SparseGaussianProcess

n_ind = 5

sparse_gp = SparseGaussianProcess(kernel, n_ind ** 2, 67, 20)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))
sparse_gp_params = sparse_gp_params._replace(kernel_params=kernel_params)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))


# %%
def f(m):
    return 2 * jnp.sin(m) + jr.normal(next(rng), m.shape) / 10


v = f(m)

phi = jnp.linspace(0, jnp.pi, n_ind + 2)[1:-1]
theta = jnp.linspace(0, 2 * jnp.pi, n_ind + 1)[1:]
phi, theta = jnp.meshgrid(phi, theta)
phi = phi.flatten()
theta = theta.flatten()
m_ind = jnp.stack([phi, theta], axis=-1)
v_ind = f(m_ind)
sphere_mesh.add_intrinsic_vector_quantity("function", v, color=(0, 0, 1))

sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    m_ind,
    v_ind,
    jnp.ones_like(m_ind) * 0.01,
)
inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)
inducing_locs_, inducing_means_ = s2.project_to_e(m_ind, v_ind)
inducing_cloud = ps.register_point_cloud(
    "inducing points", inducing_locs_, color=(0, 1, 0)
)
inducing_cloud.add_vector_quantity("inducing means", inducing_means_, color=(0, 1, 0))

# %%

inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)
inducing_locs_, inducing_means_ = s2.project_to_e(m_ind, v_ind)
inducing_cloud = ps.register_point_cloud(
    "inducing points", inducing_locs_, color=(0, 1, 0), enabled=True
)
inducing_cloud.add_vector_quantity(
    "inducing means", inducing_means_, color=(0, 1, 0), enabled=True
)


posterior_samples = sparse_gp(sparse_gp_params, sparse_gp_state, m)
sphere_mesh.add_intrinsic_vector_quantity(
    "function", jnp.mean(posterior_samples, axis=0), color=(0, 0, 1), enabled=True
)

for i in range(posterior_samples.shape[0]):
    sphere_mesh.add_intrinsic_vector_quantity(
        f"sample {i}", posterior_samples[i], color=(0.7, 0.7, 0.7), enabled=True
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
    if jnp.all(jnp.isnan(grads.kernel_params.sub_kernel_params.log_length_scale)):
        print("breaking for nan")
        break
    if i <= 10 or i % 20 == 0:
        print(i, "Loss:", train_loss)
    debug_params.append(sparse_gp_params)
    debug_states.append(sparse_gp_state)
    debug_keys.append(rng.key)

# %%
