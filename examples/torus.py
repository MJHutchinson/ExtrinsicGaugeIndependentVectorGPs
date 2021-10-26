# %%
%load_ext autoreload
%autoreload 2
import jax
import jax.numpy as jnp
from riemannianvectorgp.manifold import EmbeddedS1
import numpy as np

from riemannianvectorgp.utils import (
    t2_m_to_3d,
    GlobalRNG,
    save_obj,
    mesh_to_obj,
    flatten,
    project
)
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope

rng = GlobalRNG((0))

# %%
import polyscope as ps

ps.init()
ps.set_up_dir("z_up")

# %%
num_points = 30
u = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
v = np.linspace(0, 2 * np.pi, num_points + 1) [:-1]
u, v = np.meshgrid(u, v, indexing="ij")
u = u.flatten()
v = v.flatten()
m_mesh = np.stack([u, v], axis=-1)

T2 = EmbeddedS1() * EmbeddedS1()

torus_mesh = ps.register_surface_mesh(
    f"torus",
    *mesh_to_polyscope(
        t2_m_to_3d(m_mesh).reshape((num_points, num_points, 3)),
        wrap_x=True,
        wrap_y=True,
        reverse_x=False,
    ),
    color=(1, 1, 1),
    smooth_shade=True,
    material="wax",
    enabled=False,
)

m_3d = np.genfromtxt("/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/torus/poisson.csv", delimiter = ',')
point_cloud = ps.register_point_cloud('poisson', m_3d)

# %%
# n = 10
# m = jnp.stack([jnp.array([i/n * 2*jnp.pi, j/n * 2*jnp.pi]) * jnp.ones((m_3d.shape[0], 2)) for i in range(n) for j in range(n)], axis=0)
# inds = jnp.argmin(((m_3d - t2_m_to_3d(m)) ** 2).sum(axis=-1), axis=0)
# m = m[inds, jnp.arange(inds.shape[0])]

# def loss(m):
#     return jnp.mean((m_3d - t2_m_to_3d(m)) ** 2)


# import optax

# optim = optax.adam(1)
# params = {'m': m}
# opt_state = optim.init(params)

# for i in range(100):
#     grads = jax.grad(lambda params: loss(params['m']))(params)
#     updates, opt_state = optim.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     print(f'{i}: {loss(params["m"])=}')


# optim = optax.adam(1e-2)
# params = {'m': m}
# opt_state = optim.init(params)

# for i in range(100):
#     grads = jax.grad(lambda params: loss(params['m']))(params)
#     updates, opt_state = optim.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     print(f'{i}: {loss(params["m"])=}')


# # %%
# m = params['m']
# np.savetxt("/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/torus/poisson_intrinsic.csv", m, delimiter = ',')
m = np.genfromtxt("/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/torus/poisson_intrinsic.csv", delimiter = ',')
m_cloud = ps.register_point_cloud('m', t2_m_to_3d(m))

# %%
from riemannianvectorgp.manifold import EmbeddedKleinBottle
from riemannianvectorgp.kernel import MaternCompactRiemannianManifoldKernel
from riemannianvectorgp.kernel import ManifoldProjectionVectorKernel
from riemannianvectorgp.sparse_gp import SparseGaussianProcess

s = 5
kernel = ManifoldProjectionVectorKernel(MaternCompactRiemannianManifoldKernel(1.5, T2, 100), T2)
gp = SparseGaussianProcess(kernel, 1, 100, s)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))

samples = gp.prior(params.kernel_params, state.prior_state, m)[0]

# %%
m_cloud.add_vector_quantity('sample', project(m, samples, t2_m_to_3d)[1])
data_path = "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/torus"
np.savetxt(os.path.join(data_path, 'sample_vecs.csv'), jnp.concatenate([*project(m, samples, t2_m_to_3d)], axis=-1), delimiter=',')

# %%

save_obj(mesh_to_obj(*mesh_to_polyscope(
        t2_m_to_3d(m_mesh).reshape((num_points, num_points, 3)),
        wrap_x=True,
        wrap_y=True,
        reverse_x=False,
    ), uv_coords=(m_mesh / jnp.array([2 * jnp.pi, 2 * jnp.pi]))), os.path.join(data_path, 'torus.obj'))
# %%
