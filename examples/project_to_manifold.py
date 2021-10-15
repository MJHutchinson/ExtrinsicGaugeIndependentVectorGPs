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
# ps.load_color_map("custom_oranges", "custom_oranges.png")
# ps.load_color_map("custom_greens", "custom_greens.png")
# ps.load_color_map("greens", "textures/greens.png")
# ps.load_color_map("Oranges", "textures/oranges.png")
# ps.load_color_map("plasma", "textures/plasma.png")


# color1 = (43 / 255, 77 / 255, 89 / 255)
# color2 = (218 / 255, 103 / 255, 74 / 255)
# color3 = (255 / 255, 112 / 255, 0 / 255)

# colormap1 = "blues"
# colormap2 = "reds"
# colormap3 = "blues"

color1 = (36 / 255, 132 / 255, 141 / 255)
color2 = (114 / 255, 0 / 255, 168 / 255)
color3 = (255 / 255, 112 / 255, 0 / 255)

# colormap1 = "viridis"
# colormap2 = "plasma"
# colormap3 = "Oranges"

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
)  ### NOTE this ordering, I can change it but it'll be a pain, its latitude, longitude

# density = 10
# m_dense = np.stack(
#     np.meshgrid(
#         np.linspace(0, np.pi, density * num_points),
#         np.linspace(0, 2 * np.pi, density * num_points + 1)[1:],
#         indexing="ij",
#     ),
#     axis=-1,
# )
# x = S2.m_to_e(m)
# verticies, faces = mesh_to_polyscope(
#     x.reshape((num_points, num_points, 3)), wrap_y=False
# )


sphere_mesh = ps.register_surface_mesh(
    "Sphere",
    *mesh_to_polyscope(S2.m_to_e(m).reshape((num_points, num_points, 3)), wrap_x=False),
    color=(1, 1, 1),
    smooth_shade=True,
    material="wax",
)
sphere_mesh.set_vertex_tangent_basisX(S2.projection_matrix(m)[..., 0])
# mesh_obj = regular_square_mesh_to_obj(
#     x.reshape((num_points, num_points, 3)), wrap_y=True
# )

# GP

rng = GlobalRNG(0)
kernel = MaternCompactRiemannianManifoldKernel(1.5, S2, 144)
gp = SparseGaussianProcess(kernel, 1, 144, 3)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))

# %%

ambient_vectorfield = gp.prior(params.kernel_params, state.prior_state, m)[..., 0].T
projected_vectorfield = S2.project_to_e(
    *S2.project_to_m(S2.m_to_e(m), ambient_vectorfield)
)[1]

frames = 2
for i in range(frames):
    t = i / (frames - 1)
    sphere_mesh.add_vector_quantity(
        f"frame {i}",
        (1 - t) * ambient_vectorfield + t * projected_vectorfield,
        enabled=True,
        length=0.15,
        radius=0.0075,
        color=color3,
    )  # , color=(188/255,188/255,188/255))

# %%
sattellite_verts, sattellite_faces = import_obj('blender/satellite01.obj')
sattellite_verts
sattellite_mesh = ps.register_surface_mesh(
    "sattellite",
    sattellite_verts,
    sattellite_faces,
    color=(1, 1, 1),
    smooth_shade=True,
    material="wax",
)
# %%
