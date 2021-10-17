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
    export_vec_field,
    GlobalRNG,
)
from riemannianvectorgp.utils import (
    sphere_flat_m_to_3d,
    sphere_m_to_3d,
    interp,
    projection_matrix,
)
import potpourri3d as pp3d

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
m = np.stack([phi, theta], axis=-1)
# %%
# sphere_mesh = ps.register_surface_mesh(
#     "Sphere",
#     *mesh_to_polyscope(sphere_m_to_3d(m).reshape((num_points, num_points, 3)), wrap_x=False),
#     color=(1, 1, 1),
#     smooth_shade=True,
#     material="wax",
# )
# sphere_mesh = ps.register_surface_mesh(
#     "Flat sphere",
#     *mesh_to_polyscope(sphere_flat_m_to_3d(m).reshape((num_points, num_points, 3)), wrap_x=False, wrap_y=False),
#     color=(1, 1, 1),
#     smooth_shade=True,
#     material="wax",
# )
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
)
from riemannianvectorgp.sparse_gp import SparseGaussianProcess

s = 1
kernel = ManifoldProjectionVectorKernel(
    MaternCompactRiemannianManifoldKernel(1.5, EmbeddedS2(), 100), EmbeddedS2()
)
gp = SparseGaussianProcess(kernel, 1, 100, s)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))
sample = gp.prior(params.kernel_params, state.prior_state, m)[0]

track_points = np.genfromtxt(f"blender/satellite_tracks/track_angles.csv", delimiter=',')
track_vecs = np.genfromtxt(f"blender/satellite_tracks/track_intrinsic_vecs.csv", delimiter=',')

# %%
frames = 60
frame_meshes = []
frame_vecs = []
for i in range(frames):
    embedding_func = lambda m: interp(
        m, sphere_m_to_3d, sphere_flat_m_to_3d, t=i / (frames - 1)
    )
    proj_mat_func = lambda m: projection_matrix(m, embedding_func)

    V, F = (
        *mesh_to_polyscope(
            embedding_func(m).reshape((num_points, num_points, 3)),
            wrap_x=False,
            wrap_y=False,
        ),
    )

    euclidean_vecs = (proj_mat_func(track_points) @ track_vecs[..., np.newaxis])[..., 0]

    frame_meshes.append((V, F))
    frame_vecs.append((embedding_func(track_points), euclidean_vecs))

# %%
for i, ((V, F), (TP, TV)) in enumerate(zip(frame_meshes, frame_vecs)):
    print(i)
    # save_obj(mesh_to_obj(V, F, uv_coords=m / jnp.array([jnp.pi, 2 * jnp.pi])), f"blender/unwrap_sphere/frame_{i}.obj")
    export_vec_field(TP, TV, f"blender/unwrap_sphere/frame_{i}.csv")

# %%
