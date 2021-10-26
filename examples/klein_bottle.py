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
    GlobalRNG,
    save_obj,
    mesh_to_obj,
    flatten,
    project
)
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope

rng = GlobalRNG((0))
import os

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
m_3d = np.genfromtxt("/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/klein_bottle/poisson.csv", delimiter = ',')
point_cloud = ps.register_point_cloud('poisson', m_3d)
tangent_vec = tangent_vec / jnp.linalg.norm(tangent_vec, axis=-2)[..., np.newaxis, :]
klein_mesh.set_vertex_tangent_basisX(tangent_vec[..., 0])

# %%
n = 100
m = jnp.stack([jnp.array([i/n * jnp.pi, j/n *2*jnp.pi]) * jnp.ones((m_3d.shape[0], 2)) for i in range(n) for j in range(n)], axis=0)
inds = jnp.argmin(((m_3d - klein_bottle_m_to_3d(m)) ** 2).sum(axis=-1), axis=0)
m = m[inds, jnp.arange(inds.shape[0])]

def loss(m):
    return jnp.mean((m_3d - klein_bottle_m_to_3d(m)) ** 2)


import optax

optim = optax.adam(1)
params = {'m': m}
opt_state = optim.init(params)

for i in range(100):
    grads = jax.grad(lambda params: loss(params['m']))(params)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(f'{i}: {loss(params["m"])=}')


optim = optax.adam(1e-2)
params = {'m': m}
opt_state = optim.init(params)

for i in range(100):
    grads = jax.grad(lambda params: loss(params['m']))(params)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(f'{i}: {loss(params["m"])=}')


# %%
m = params['m']
np.savetxt("/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/klein_bottle/poisson_intrinsic.csv", m, delimiter = ',')
m = np.genfromtxt("/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/klein_bottle/poisson_intrinsic.csv", delimiter = ',')
m_cloud = ps.register_point_cloud('m', klein_bottle_m_to_3d(m))

# %%
from riemannianvectorgp.manifold import EmbeddedKleinBottle
from riemannianvectorgp.kernel import MaternCompactRiemannianManifoldKernel
from riemannianvectorgp.kernel import ManifoldProjectionVectorKernel
from riemannianvectorgp.sparse_gp import SparseGaussianProcess

s = 5
KB = EmbeddedKleinBottle()
kernel = ManifoldProjectionVectorKernel(MaternCompactRiemannianManifoldKernel(1.5, KB, 100), KB)
gp = SparseGaussianProcess(kernel, 1, 100, s)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))


# %%
samples = gp.prior(params.kernel_params, state.prior_state, m)[3]
m_cloud.add_vector_quantity('sample', project(m, samples, klein_bottle_m_to_3d)[1])
data_path = "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/klein_bottle"
np.savetxt(os.path.join(data_path, 'sample_vecs.csv'), jnp.concatenate([*project(m, samples, klein_bottle_m_to_3d)], axis=-1), delimiter=',')

# %%

save_obj(mesh_to_obj(*mesh_to_polyscope(
        klein_bottle_m_to_3d(m).reshape((num_points, num_points, 3)),
        wrap_x=False,
        wrap_y=True,
        reverse_x=False,
    ), uv_coords=(m / jnp.array([jnp.pi, 2 * jnp.pi])) - jnp.array([0.5172414, 0])), f"blender/klein_bottle/klein_bottle.obj")
# %%
