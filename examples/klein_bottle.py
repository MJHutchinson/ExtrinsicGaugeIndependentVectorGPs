# %%
%load_ext autoreload
%autoreload 2
import jax
import jax.numpy as jnp
from riemannianvectorgp.manifold import S1
import numpy as np

from riemannianvectorgp.utils import (
    klein_bottle_m_to_3d,
    klein_fig8_m_to_3d,
    klein_fig8_double_m_to_3d,
    GlobalRNG
)
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope

rng = GlobalRNG((0))

# %%
import polyscope as ps

ps.init()
ps.set_up_dir("z_up")

# %%
num_points = 30
u = np.linspace(0, 2*np.pi, 2*num_points - 1)
shift = 15
u = u[shift:shift+num_points]
v = np.linspace(0, 2 * np.pi, num_points + 1) [1:]
u, v = np.meshgrid(u, v, indexing="ij")
u = u.flatten()
v = v.flatten()
m = np.stack([u, v], axis=-1)

u = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
v = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
u, v = np.meshgrid(u, v, indexing="ij")
u = u.flatten()
v = v.flatten()
m2 = np.stack([u, v], axis=-1)


def _2d_to_3d(m):
    return jnp.stack([m[..., 0], m[..., 1], jnp.zeros_like(m[..., 0])], axis=-1)


klein_mesh = ps.register_surface_mesh(
    f"klein_8_double_surface",
    *mesh_to_polyscope(
        klein_fig8_double_m_to_3d(m).reshape((num_points, num_points, 3)),
        wrap_x=True,
        wrap_y=True,
        reverse_x=False,
    ),
    # color=(28/255,99/255,227/255),
    color=(1, 1, 1),
    # color=(39/255,119/255,177/255), # matplotlib blue
    # color=(252/255,128/255,43/255), # matplotlib orange
    # color=(51/255, 159/255, 54/255), # matplotlib green
    # color=(217/255, 95/255, 2/255), # colorbrewer orange
    # color=(231/255, 41/255, 139/255), # colorbrewer magenta
    smooth_shade=True,
    material="wax",
    enabled=False,
)
klein_mesh = ps.register_surface_mesh(
    f"klein_8_surface",
    *mesh_to_polyscope(
        klein_bottle_m_to_3d(m).reshape((num_points, num_points, 3)),
        wrap_x=False,
        wrap_y=True,
        reverse_x=False,
    ),
    # color=(28/255,99/255,227/255),
    color=(1, 1, 1),
    # color=(39/255,119/255,177/255), # matplotlib blue
    # color=(252/255,128/255,43/255), # matplotlib orange
    # color=(51/255, 159/255, 54/255), # matplotlib green
    # color=(217/255, 95/255, 2/255), # colorbrewer orange
    # color=(231/255, 41/255, 139/255), # colorbrewer magenta
    smooth_shade=True,
    material="wax",
)

tangent_vec = jnp.stack(
    [
        jax.vmap(jax.grad(lambda m: klein_bottle_m_to_3d(m)[..., i]))(m)
        for i in range(3)
    ],
    axis=-2,
)
tangent_vec = tangent_vec / jnp.linalg.norm(tangent_vec, axis=-2)[..., np.newaxis, :]
klein_mesh.set_vertex_tangent_basisX(tangent_vec[..., 0])
# %%

from riemannianvectorgp.manifold import EmbeddedKleinBottle

KB = EmbeddedKleinBottle()
kb_eigs = KB.laplacian_eigenfunction(jnp.arange(100), m)

for i in range(100):
    klein_mesh.add_scalar_quantity(f'eigfunc {i}', kb_eigs[:, i, 0])
# %%
from riemannianvectorgp.kernel import MaternCompactRiemannianManifoldKernel

kernel = MaternCompactRiemannianManifoldKernel(1.5, KB, 100)
kernel_parmas = kernel.init_params(next(rng))
kernel_parmas = kernel_parmas._replace(log_length_scale=jnp.array(-2))
k = kernel.matrix(kernel_parmas, m, m)
klein_mesh.add_scalar_quantity(f'kernel', k[450 + 10 - 5*30, :, 0, 0])

# %%
from riemannianvectorgp.kernel import ManifoldProjectionVectorKernel

kernel = ManifoldProjectionVectorKernel(MaternCompactRiemannianManifoldKernel(1.5, KB, 100), KB)
kernel_parmas = kernel.init_params(next(rng))
k = kernel.matrix(kernel_parmas, m, m)
klein_mesh.add_intrinsic_vector_quantity('vec_kernel', k[450 + 10 - 5*30, :, :, :] @ jnp.array([1, 0]))
# %%
from riemannianvectorgp.sparse_gp import SparseGaussianProcess

s = 5
kernel = ManifoldProjectionVectorKernel(MaternCompactRiemannianManifoldKernel(1.5, KB, 100), KB)
gp = SparseGaussianProcess(kernel, 1, 100, s)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))

samples = gp.prior(params.kernel_params, state.prior_state, m)

for i in range(s):
    klein_mesh.add_intrinsic_vector_quantity(f'sample {i}', samples[i])

# %%
