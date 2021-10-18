# %%
%load_ext autoreload
%autoreload 2
import os

os.chdir("..")

# %%
import jax
import numpy as np
import jax.numpy as jnp

from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope
# from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    MaternThreeHalves
)
from riemannianvectorgp.utils import (
    mesh_to_obj,
    save_obj,
    make_faces_from_vectors,
    regular_square_mesh_to_obj,
    make_scalar_texture,
    square_mesh_to_obj,
    export_vec_field,
    GlobalRNG,
)
from riemannianvectorgp.utils import (
    sphere_flat_m_to_3d,
    sphere_m_to_3d,
    interp,
    projection_matrix,
    project,
)
import polyscope as ps

rng = GlobalRNG(0)
# %%
ps.init()
ps.set_up_dir("z_up")

color1 = (36 / 255, 132 / 255, 141 / 255)
color2 = (114 / 255, 0 / 255, 168 / 255)
color3 = (255 / 255, 112 / 255, 0 / 255)

colormap1 = "viridis"
colormap2 = "plasma"
colormap3 = "Oranges"

# %%
S2 = EmbeddedS2(1.0)
num_points = 30
e = 1e-3
phi = np.linspace(0 + e, np.pi - e, num_points)
theta = np.linspace(0, 2 * np.pi, num_points)
phi, theta = np.meshgrid(phi, theta, indexing="ij")
phi = phi.flatten()
theta = theta.flatten()
m = jnp.array(np.stack([phi, theta], axis=-1))

sphere_mesh = ps.register_surface_mesh(
    "Flat sphere",
    *mesh_to_polyscope(
        sphere_flat_m_to_3d(m).reshape((num_points, num_points, 3)),
        wrap_x=False,
        wrap_y=False,
    ),
    color=(1, 1, 1),
    smooth_shade=True,
    material="wax",
)
sphere_mesh.set_vertex_tangent_basisX(projection_matrix(m, sphere_flat_m_to_3d)[..., 0])
# %%
track_points = np.genfromtxt(
    f"blender/satellite_tracks/track_angles.csv", delimiter=","
)
track_vecs = np.genfromtxt(
    f"blender/satellite_tracks/track_intrinsic_vecs.csv", delimiter=","
)

track_points_3d, track_vecs_3d = project(track_points, track_vecs, sphere_flat_m_to_3d)

track = ps.register_point_cloud('track', track_points_3d)
track.add_vector_quantity('vecs', track_vecs_3d)

# %%
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    TFPKernel,
    ManifoldProjectionVectorKernel,
)
from riemannianvectorgp.gp import GaussianProcess

import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

kernel = ManifoldProjectionVectorKernel(
    MaternCompactRiemannianManifoldKernel(1.5, EmbeddedS2(), 100), EmbeddedS2()
)
# with jax.disable_jit():
kernel = TFPKernel(tfk.MaternThreeHalves, 2, 2)
gp = GaussianProcess(kernel)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(jnp.array([1.0,1.0])))
)

state = gp.condition(params, track_points, track_vecs, jnp.ones_like(track_vecs) * 1e-6)
mean_wrong, K = gp(params, state, m)
s_wrong = jnp.linalg.det(K[jnp.arange(K.shape[0]), jnp.arange(K.shape[0])])

sphere_mesh.add_intrinsic_vector_quantity('mean_wrong', mean_wrong)
sphere_mesh.add_scalar_quantity('variance_wrong', s_wrong)

# state = gp.randomize(params, state, next(rng))
# sample = gp.prior(params.kernel_params, state.prior_state, m)[0]

# %%
kernel = ManifoldProjectionVectorKernel(
    MaternCompactRiemannianManifoldKernel(1.5, EmbeddedS2(), 144), EmbeddedS2()
)
gp = GaussianProcess(kernel)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(jnp.array(1.0)))
)
# with jax.disable_jit():
state = gp.condition(params, track_points, track_vecs, jnp.ones_like(track_vecs) * 1e-6)
mean_right, K = gp(params, state, m)
s_right = jnp.linalg.det(K[jnp.arange(K.shape[0]), jnp.arange(K.shape[0])])

sphere_mesh.add_intrinsic_vector_quantity('mean_right', mean_right)
sphere_mesh.add_scalar_quantity('variance_right', s_right)

# %%
data_path = "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/kernels"
np.savetxt(os.path.join(data_path, 'mean_zero.csv'), jnp.concatenate([*project(m, jnp.zeros_like(mean_wrong), sphere_flat_m_to_3d)], axis=-1), delimiter=',')
np.savetxt(os.path.join(data_path, 'mean_wrong.csv'), jnp.concatenate([*project(m, mean_wrong, sphere_flat_m_to_3d)], axis=-1), delimiter=',')
np.savetxt(os.path.join(data_path, 'mean_right.csv'), jnp.concatenate([*project(m, mean_right, sphere_flat_m_to_3d)], axis=-1), delimiter=',')
np.savetxt(os.path.join(data_path, 's_wrong.csv'), s_wrong, delimiter=',')
np.savetxt(os.path.join(data_path, 's_right.csv'), s_right, delimiter=',')


# %%
