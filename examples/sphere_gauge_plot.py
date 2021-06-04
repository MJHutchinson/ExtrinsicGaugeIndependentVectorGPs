import numpy as np
import jax.numpy as jnp
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
)

import polyscope as ps

ps.init()
ps.set_up_dir("z_up")
ps.load_color_map("custom_oranges", "custom_oranges.png")
ps.load_color_map("custom_greens", "custom_greens.png")


S2 = EmbeddedS2(1.0)
num_points = 30
phi = np.linspace(0, np.pi, num_points)
theta = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
phi, theta = np.meshgrid(phi, theta)
phi = phi.flatten()
theta = theta.flatten()
m = np.stack(
    [phi, theta], axis=-1
)  ### NOTE this ordering, I can change it but it'll be a pain, its latitude, longitude
x = S2.m_to_e(m)

sphere_mesh = ps.register_surface_mesh(
    "Sphere",
    *mesh_to_polyscope(x.reshape((num_points, num_points, 3)), wrap_y=False),
    color=(28/255,99/255,227/255),
    # color=(39/255,119/255,177/255), # matplotlib blue
    # color=(252/255,128/255,43/255), # matplotlib orange
    # color=(51/255, 159/255, 54/255), # matplotlib green
    # color=(217/255, 95/255, 2/255), # colorbrewer orange
    # color=(231/255, 41/255, 139/255), # colorbrewer magenta
    smooth_shade=True,
    material="wax",
)
sphere_mesh.set_vertex_tangent_basisX(S2.projection_matrix(m)[...,0])

# GP 

rng = GlobalRNG(0)
kernel = MaternCompactRiemannianManifoldKernel(1.5, S2, 144)
gp = SparseGaussianProcess(kernel, 1, 144, 5)
(params,state) = gp.init_params_with_state(next(rng))
params = params._replace(kernel_params = params.kernel_params._replace(log_length_scale=jnp.log(0.1)))
state = gp.randomize(params, state, next(rng))

samples = gp.prior(params.kernel_params,state.prior_state, m)
samples_proj = S2.project_to_m(x, samples[2:5,:,0].T)

sphere_mesh.add_scalar_quantity("samples_x", -samples[0,:,0], enabled=False, cmap="custom_oranges")
sphere_mesh.add_scalar_quantity("samples_y", -samples[1,:,0], enabled=False, cmap="custom_oranges")

samples_a = (samples[0,:,0] + samples[1,:,0]) / np.sqrt(2)
samples_b = (-samples[0,:,0] + samples[1,:,0]) / np.sqrt(2)

sphere_mesh.add_scalar_quantity("samples_a", samples_a, enabled=False, cmap="custom_greens")
sphere_mesh.add_scalar_quantity("samples_b", samples_b, enabled=False, cmap="custom_greens")

sphere_mesh.add_intrinsic_vector_quantity("samples_xy", -samples[0:2,:,0].T, enabled=False, length=0.1, radius=0.0075, color=(188/255,188/255,188/255))



# gauges

nonzero_idxs = np.array((558,642,738,)) # 558, 588, 642, 708, 738

basis_vec_x = np.zeros((num_points**2, 2))
basis_vec_x[nonzero_idxs,0] = 1

basis_vec_y = np.zeros((num_points**2, 2))
basis_vec_y[nonzero_idxs,1] = 1

sphere_mesh.add_intrinsic_vector_quantity("basis_vec_x", -basis_vec_x, enabled=False, length=0.25, radius=0.015, color=(252/255,128/255,43/255)) #, color=(116/255,196/255,254/255))
sphere_mesh.add_intrinsic_vector_quantity("basis_vec_y", -basis_vec_y, enabled=False, length=0.25, radius=0.015, color=(252/255,128/255,43/255)) #, color=(0/255,43/255,101/255))


basis_vec_a = np.zeros((num_points**2, 2))
basis_vec_a[nonzero_idxs,0] = -np.sqrt(2)/2
basis_vec_a[nonzero_idxs,1] = -np.sqrt(2)/2

basis_vec_b = np.zeros((num_points**2, 2))
basis_vec_b[nonzero_idxs,0] = -np.sqrt(2)/2
basis_vec_b[nonzero_idxs,1] = np.sqrt(2)/2

sphere_mesh.add_intrinsic_vector_quantity("basis_vec_a", basis_vec_a, enabled=False, length=0.25, radius=0.015, color=(51/255, 159/255, 54/255)) #, color=(255/255,205/255,120/255))
sphere_mesh.add_intrinsic_vector_quantity("basis_vec_b", basis_vec_b, enabled=False, length=0.25, radius=0.015, color=(51/255, 159/255, 54/255)) #, color=(176/255,52/255,0/255))


# scalar GPs

sphere_mesh.add_scalar_quantity("samples_rx", samples[2,:,0], enabled=False, cmap="blues")
sphere_mesh.add_scalar_quantity("samples_ry", samples[3,:,0], enabled=False, cmap="blues")
sphere_mesh.add_scalar_quantity("samples_rz", samples[4,:,0], enabled=False, cmap="blues")
sphere_mesh.add_vector_quantity("samples_emb_vec", -samples[2:5,:,0].T, enabled=False, length=0.15, radius=0.0075, color=(188/255,188/255,188/255)) #, color=(188/255,188/255,188/255))
sphere_mesh.add_intrinsic_vector_quantity("samples_proj", -samples_proj[1], enabled=False, length=0.15, radius=0.0075, color=(188/255,188/255,188/255)) #, color=(188/255,188/255,188/255))


# camera options: ground=shadow, 8x blur, 4x ssaa, 
# {"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"viewMat":[1.0,-1.74845567357806e-07,0.0,0.0,0.0,-3.82137135384588e-15,1.0,-0.200000017881393,-1.74845567357806e-07,-1.0,-3.82137135384588e-15,-4.60390377044678,0.0,-0.0,0.0,1.0]}
ps.show()

# save at 960x480, 960x240
# convert samples_xy.png -crop 360x360+300+100 samples_xy.png 
# convert basis_vec_ab.png -crop 180x180+390+60 basis_vec_ab.png