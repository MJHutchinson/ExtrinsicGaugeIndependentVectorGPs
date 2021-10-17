# %%
# %load_ext autoreload
# %autoreload 2
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
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    TFPKernel,
    ManifoldProjectionVectorKernel,
)
from riemannianvectorgp.gp import GaussianProcess

s = 1
# kernel = ManifoldProjectionVectorKernel(
#     MaternCompactRiemannianManifoldKernel(1.5, EmbeddedS2(), 100), EmbeddedS2()
# )
kernel = TFPKernel(tfp.math.psd_kernels.MaternThreeHalves, 2, 2)
gp = GaussianProcess(kernel)(kernel, 1, 100, s)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)

track_points = np.genfromtxt(
    f"blender/satellite_tracks/track_angles.csv", delimiter=","
)
track_vecs = np.genfromtxt(
    f"blender/satellite_tracks/track_intrinsic_vecs.csv", delimiter=","
)

state = gp.condition(params, track_points, track_vecs, jnp.ones_like(track_vecs) * 1e-6)
m, K = gp(params, state, m)

state = gp.randomize(params, state, next(rng))
sample = gp.prior(params.kernel_params, state.prior_state, m)[0]
