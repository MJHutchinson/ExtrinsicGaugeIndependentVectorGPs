# %%
# %load_ext autoreload
# %autoreload 2
import os

os.chdir("..")
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
)
import potpourri3d as pp3d

import polyscope as ps

ps.init()
ps.set_up_dir("z_up")
# ps.load_color_map("custom_oranges", "custom_oranges.png")
# ps.load_color_map("custom_greens", "custom_greens.png")
ps.load_color_map("greens", "textures/greens.png")
ps.load_color_map("Oranges", "textures/oranges.png")
ps.load_color_map("plasma", "textures/plasma.png")


# color1 = (43 / 255, 77 / 255, 89 / 255)
# color2 = (218 / 255, 103 / 255, 74 / 255)
# color3 = (255 / 255, 112 / 255, 0 / 255)

# colormap1 = "blues"
# colormap2 = "reds"
# colormap3 = "blues"

color1 = (36 / 255, 132 / 255, 141 / 255)
color2 = (114 / 255, 0 / 255, 168 / 255)
color3 = (255 / 255, 112 / 255, 0 / 255)

colormap1 = "viridis"
colormap2 = "plasma"
colormap3 = "Oranges"

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
density = 10
m_dense = np.stack(
    np.meshgrid(
        np.linspace(0, np.pi, density * num_points),
        np.linspace(0, 2 * np.pi, density * num_points + 1)[1:],
        indexing="ij",
    ),
    axis=-1,
)
x = S2.m_to_e(m)
verticies, faces = mesh_to_polyscope(
    x.reshape((num_points, num_points, 3)), wrap_y=False
)


sphere_mesh = ps.register_surface_mesh(
    "Sphere",
    *mesh_to_polyscope(x.reshape((num_points, num_points, 3)), wrap_x=False),
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
uv = m / m.max(axis=0, keepdims=True)
sphere_mesh.add_parameterization_quantity("uv map", uv)
sphere_mesh.set_vertex_tangent_basisX(S2.projection_matrix(m)[..., 0])
mesh_obj = regular_square_mesh_to_obj(
    x.reshape((num_points, num_points, 3)), wrap_y=True
)

# GP

rng = GlobalRNG(0)
kernel = MaternCompactRiemannianManifoldKernel(1.5, S2, 144)
gp = SparseGaussianProcess(kernel, 1, 144, 5)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))

samples = gp.prior(params.kernel_params, state.prior_state, m)
samples_proj = S2.project_to_m(x, samples[2:5, :, 0].T)

# sphere_mesh.add_scalar_quantity("samples_x", -samples[0,:,0], enabled=False, cmap="custom_oranges")
# sphere_mesh.add_scalar_quantity("samples_y", -samples[1,:,0], enabled=False, cmap="custom_oranges")
sphere_mesh.add_scalar_quantity(
    "samples_x", -samples[0, :, 0], enabled=False, cmap=colormap1
)
sphere_mesh.add_scalar_quantity(
    "samples_y", -samples[1, :, 0], enabled=False, cmap=colormap1
)


samples_a = (samples[0, :, 0] + samples[1, :, 0]) / np.sqrt(2)
samples_b = (-samples[0, :, 0] + samples[1, :, 0]) / np.sqrt(2)

# sphere_mesh.add_scalar_quantity("samples_a", samples_a, enabled=False, cmap="custom_greens")
# sphere_mesh.add_scalar_quantity("samples_b", samples_b, enabled=False, cmap="custom_greens")
sphere_mesh.add_scalar_quantity("samples_a", samples_a, enabled=False, cmap=colormap2)
sphere_mesh.add_scalar_quantity("samples_b", samples_b, enabled=False, cmap=colormap2)

sphere_mesh.add_intrinsic_vector_quantity(
    "samples_xy",
    -samples[0:2, :, 0].T,
    enabled=False,
    length=0.1,
    radius=0.0075,
    color=color3,
)


# gauges

nonzero_idxs = np.array(
    (
        558,
        642,
        738,
    )
)  # 558, 588, 642, 708, 738

basis_vec_x = np.zeros((num_points ** 2, 2))
basis_vec_x[nonzero_idxs, 0] = 1

basis_vec_y = np.zeros((num_points ** 2, 2))
basis_vec_y[nonzero_idxs, 1] = 1

sphere_mesh.add_intrinsic_vector_quantity(
    "basis_vec_x", -basis_vec_x, enabled=False, length=0.25, radius=0.015, color=color2
)  # , color=(116/255,196/255,254/255)) (252/255,128/255,43/255)
sphere_mesh.add_intrinsic_vector_quantity(
    "basis_vec_y", -basis_vec_y, enabled=False, length=0.25, radius=0.015, color=color2
)  # , color=(0/255,43/255,101/255)) (252/255,128/255,43/255)


basis_vec_a = np.zeros((num_points ** 2, 2))
basis_vec_a[nonzero_idxs, 0] = -np.sqrt(2) / 2
basis_vec_a[nonzero_idxs, 1] = -np.sqrt(2) / 2

basis_vec_b = np.zeros((num_points ** 2, 2))
basis_vec_b[nonzero_idxs, 0] = -np.sqrt(2) / 2
basis_vec_b[nonzero_idxs, 1] = np.sqrt(2) / 2

sphere_mesh.add_intrinsic_vector_quantity(
    "basis_vec_a", basis_vec_a, enabled=False, length=0.25, radius=0.015, color=color1
)  # , color=(255/255,205/255,120/255)) (51/255, 159/255, 54/255)
sphere_mesh.add_intrinsic_vector_quantity(
    "basis_vec_b", basis_vec_b, enabled=False, length=0.25, radius=0.015, color=color1
)  # , color=(176/255,52/255,0/255)) (51/255, 159/255, 54/255)
# scalar GPs

sphere_mesh.add_scalar_quantity(
    "samples_rx", samples[2, :, 0], enabled=False, cmap=colormap3
)
sphere_mesh.add_scalar_quantity(
    "samples_ry", samples[3, :, 0], enabled=False, cmap=colormap3
)
sphere_mesh.add_scalar_quantity(
    "samples_rz", samples[4, :, 0], enabled=False, cmap=colormap3
)

sphere_mesh.add_vector_quantity(
    "samples_emb_vec",
    -samples[2:5, :, 0].T,
    enabled=False,
    length=0.15,
    radius=0.0075,
    color=color3,
)  # , color=(188/255,188/255,188/255))
sphere_mesh.add_intrinsic_vector_quantity(
    "samples_proj",
    -samples_proj[1],
    enabled=False,
    length=0.15,
    radius=0.0075,
    color=color3,
)  # , color=(188/255,188/255,188/255))


# %%
save_obj(mesh_obj, "blender/sphere.obj")

make_scalar_texture(
    lambda m: gp.prior(params.kernel_params, state.prior_state, m)[0, :, 0],
    m_dense,
    "blender/samples_x.png",
    cmap=colormap1,
)
make_scalar_texture(
    lambda m: gp.prior(params.kernel_params, state.prior_state, m)[1, :, 0],
    m_dense,
    "blender/samples_y.png",
    cmap=colormap1,
)

make_scalar_texture(
    lambda m: gp.prior(params.kernel_params, state.prior_state, m)[0, :, 0]
    + gp.prior(params.kernel_params, state.prior_state, m)[1, :, 0],
    m_dense,
    "blender/samples_a.png",
    cmap=colormap2,
)
make_scalar_texture(
    lambda m: -gp.prior(params.kernel_params, state.prior_state, m)[0, :, 0]
    + gp.prior(params.kernel_params, state.prior_state, m)[1, :, 0],
    m_dense,
    "blender/samples_b.png",
    cmap=colormap2,
)

save_obj(
    make_faces_from_vectors(verticies, S2.project_to_e(m, -samples[0:2, :, 0].T)[1]),
    "blender/samples_xy.obj",
)

make_scalar_texture(
    lambda m: gp.prior(params.kernel_params, state.prior_state, m)[2, :, 0],
    m_dense,
    "blender/samples_rx.png",
    cmap=colormap3,
)
make_scalar_texture(
    lambda m: gp.prior(params.kernel_params, state.prior_state, m)[3, :, 0],
    m_dense,
    "blender/samples_ry.png",
    cmap=colormap3,
)
make_scalar_texture(
    lambda m: gp.prior(params.kernel_params, state.prior_state, m)[4, :, 0],
    m_dense,
    "blender/samples_rz.png",
    cmap=colormap3,
)

save_obj(
    make_faces_from_vectors(verticies, -samples[2:5, :, 0].T),
    "blender/samples_emb_vec.obj",
)
save_obj(
    make_faces_from_vectors(verticies, S2.project_to_e(m, -samples_proj[1])[1]),
    "blender/samples_proj.obj",
)

# camera options: ground=shadow, 8x blur, 4x ssaa,
# {"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"viewMat":[1.0,-1.74845567357806e-07,0.0,0.0,0.0,-3.82137135384588e-15,1.0,-0.200000017881393,-1.74845567357806e-07,-1.0,-3.82137135384588e-15,-4.60390377044678,0.0,-0.0,0.0,1.0]}
# %%
ps.show()

# save at 960x480, 960x240
# convert samples_xy.png -crop 360x360+300+100 samples_xy.png
# convert basis_vec_ab.png -crop 180x180+390+60 basis_vec_ab.png

# %%

vertex_basis = S2.projection_matrix(m)
sample_proj_3d = jnp.einsum("nem,nm->ne", vertex_basis, samples_proj[1])
verticies, faces = mesh_to_polyscope(
    x.reshape((num_points, num_points, 3)), wrap_y=False
)

# %%
mesh_obj = mesh_to_obj(verticies, faces, uv_coords=m / (2 * jnp.pi))
save_obj(mesh_obj, "sphere.obj")

# %%
save_obj(make_faces_from_vectors(verticies, sample_proj_3d), "faces.obj")

# %%


# %%

make_scalar_texture(
    lambda m: gp.prior(params.kernel_params, state.prior_state, m)[0, :, 0],
    m_dense,
    "scalar.png",
)
# %%
rng = GlobalRNG(0)

kernel = MaternCompactRiemannianManifoldKernel(1.5, S2, 144)
kernel_params = kernel.init_params(next(rng))
k = kernel.matrix(kernel_params, m, m)
sphere_mesh.add_scalar_quantity("kernel", k[0, :, 0, 0])

# %%
