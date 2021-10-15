# %%
%load_ext autoreload
%autoreload 2
import os

os.chdir("..")

# %%
import numpy as np
import jax.numpy as jnp
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
)
from riemannianvectorgp.utils import (
    mesh_to_obj,
    save_obj,
    make_faces_from_vectors,
    regular_square_mesh_to_obj,
    make_scalar_texture,
    square_mesh_to_obj,
    import_obj
)
import potpourri3d as pp3d

import polyscope as ps

# %%

ps.init()
ps.set_up_dir("z_up")

color1 = (36 / 255, 132 / 255, 141 / 255)
color2 = (114 / 255, 0 / 255, 168 / 255)
color3 = (255 / 255, 112 / 255, 0 / 255)


# %%

S2 = EmbeddedS2(1.0)
num_points = 30
phi = np.linspace(0, np.pi, num_points)
theta = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
phi, theta = np.meshgrid(phi, theta, indexing="ij")
phi = phi.flatten()
theta = theta.flatten()
m = np.stack(
    [phi, theta], axis=-1
)

sphere_mesh = ps.register_surface_mesh(
    "Sphere",
    *mesh_to_polyscope(S2.m_to_e(m).reshape((num_points, num_points, 3)), wrap_x=False),
    color=(1, 1, 1),
    smooth_shade=True,
    material="wax",
)
sphere_mesh.set_vertex_tangent_basisX(S2.projection_matrix(m)[..., 0])


rng = GlobalRNG(0)
kernel = MaternCompactRiemannianManifoldKernel(1.5, S2, 144)
gp = SparseGaussianProcess(kernel, 1, 144, 3)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))

# %%

num_steps = 30
theta = np.linspace(0, 2 * np.pi, num_steps)
phi = np.zeros_like(theta) + np.pi/2
track = np.stack(
    [phi, theta], axis=-1
)

track_points = S2.m_to_e(track)
track_vecs = S2.project_to_e(*S2.project_to_m(track_points, gp.prior(params.kernel_params, state.prior_state, track)[..., 0].T))[1]

track_cloud = ps.register_point_cloud('track', track_points)
track_cloud.add_vector_quantity('samples', track_vecs)

# %%
sattellite_verts, sattellite_faces = import_obj('blender/satellite01.obj')
sattellite_verts = (sattellite_verts - jnp.mean(sattellite_verts, axis=0)) * 0.01 +jnp.array([0,-1.0,0])
sattellite_mesh = ps.register_surface_mesh(
    "sattellite",
    sattellite_verts,
    sattellite_faces,
    color=(1, 1, 1),
    smooth_shade=True,
    material="wax",
)
# %%
