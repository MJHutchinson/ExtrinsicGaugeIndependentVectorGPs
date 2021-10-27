# %%
import os

import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel,
)
from riemannianvectorgp.utils import GlobalRNG
import pickle

from riemannianvectorgp.utils import (
    sphere_m_to_3d,
    sphere_flat_m_to_3d,
    GlobalRNG,
    save_obj,
    mesh_to_polyscope,
    mesh_to_obj,
    square_mesh_to_obj,
    flatten,
    project,
)

data_path = "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/wind_plots"

# %%
rng1 = GlobalRNG()
rng2 = GlobalRNG()

lat_size = 32
lon_size = 64

# %%
# Load data
m_cond = np.load("log/m_cond.npy") # Conditioning locations
v_cond = np.load("log/v_cond.npy") # Conditioning values
m = np.load("log/m.npy") # Test locations

m_sphere = m.reshape((lon_size, lat_size, -1))
m_sphere = np.concatenate(
    (m_sphere, m_sphere[0, :, :][np.newaxis, :, :] + np.array([0, 2 * np.pi])), axis=0
)
top_line = m_sphere[:, 0:1, :].copy()
top_line[:, :, 0] = 0 + 1e-4
bottom_line = m_sphere[:, 0:1, :].copy()
bottom_line[:, :, 0] = np.pi - 1e-4
m_sphere = np.concatenate((top_line, m_sphere, bottom_line), axis=1)
m_sphere = m_sphere.reshape((-1, 2))

m_poisson = np.genfromtxt(
    "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/kernels/poisson_sample_2.csv",
    delimiter=",",
)
m_poisson = EmbeddedS2().e_to_m(m_poisson)
m_poisson = np.mod(m_poisson, np.array([np.pi, 2*np.pi]))

noises_cond = jnp.ones_like(v_cond) * 1.7 

# %%
# Setup Euclidean GP
kernel_r2 = ScaledKernel(TFPKernel(tfk.MaternThreeHalves, 2, 2))
gp_r2 = GaussianProcess(kernel_r2)

# Setup Spherical GP
S2 = EmbeddedS2(1.0)
kernel_s2 = ScaledKernel(
    ManifoldProjectionVectorKernel(
        MaternCompactRiemannianManifoldKernel(3 / 2, S2, 144), S2
    )  # 144 is the maximum number of basis functions we have implemented
)
gp_s2 = GaussianProcess(kernel_s2)

# %%
# Set length scale and amplitudes
log_length_scale_r2 = np.load("log/r2_log_length_scale.npy") # -1.4076507
log_length_scale_s2 = np.load("log/s2_log_length_scale.npy") #-1.522926
log_amplitude_r2 = 1.6 # np.load("log/r2_log_amplitude.npy")
log_amplitude_s2 = 9.7 # np.load("log/s2_log_amplitude.npy")

# Refresh r2 kernel
kernel_params_r2 = kernel_r2.init_params(rng1)
sub_kernel_params_r2 = kernel_params_r2.sub_kernel_params
sub_kernel_params_r2 = sub_kernel_params_r2._replace(
    log_length_scale=log_length_scale_r2
)
kernel_params_r2 = kernel_params_r2._replace(sub_kernel_params=sub_kernel_params_r2)
kernel_params_r2 = kernel_params_r2._replace(log_amplitude=log_amplitude_r2)

# Refresh s2 kernel
kernel_params_s2 = kernel_s2.init_params(rng2)
sub_kernel_params_s2 = kernel_params_s2.sub_kernel_params
sub_kernel_params_s2 = sub_kernel_params_s2._replace(
    log_length_scale=log_length_scale_s2
)
kernel_params_s2 = kernel_params_s2._replace(sub_kernel_params=sub_kernel_params_s2)
kernel_params_s2 = kernel_params_s2._replace(log_amplitude=log_amplitude_s2)

# Initialize GPs
gp_params_r2, gp_state_r2 = gp_r2.init_params_with_state(next(rng1))
gp_params_r2 = gp_params_r2._replace(kernel_params=kernel_params_r2)

gp_params_s2, gp_state_s2 = gp_s2.init_params_with_state(next(rng2))
gp_params_s2 = gp_params_s2._replace(kernel_params=kernel_params_s2)

# %%
# Fit and save Euclidean GP
gp_state_r2 = gp_r2.condition(gp_params_r2, m_cond, v_cond, noises_cond)

with open("log/model/gp_r2.pickle", "wb") as f:
    pickle.dump(gp_r2, f)

with open("log/model/gp_params_r2.pickle", "wb") as f:
    pickle.dump(gp_params_r2, f)

with open("log/model/gp_state_s2.pickle", "wb") as f:
    pickle.dump(gp_state_r2, f)

# %%
# Fit and save Spherical GP
gp_state_s2 = gp_s2.condition(gp_params_s2, m_cond, v_cond, noises_cond)

with open("log/model/gp_s2.pickle", "wb") as f:
    pickle.dump(gp_s2, f)

with open("log/model/gp_params_s2.pickle", "wb") as f:
    pickle.dump(gp_params_s2, f)

with open("log/model/gp_state_s2.pickle", "wb") as f:
    pickle.dump(gp_state_s2, f)

# %%
# Prediction on test locations (Euclidean)
posterior_mean_r2, var_r2 = gp_r2(gp_params_r2, gp_state_r2, m_poisson)
_, posterior_cov_r2 = gp_r2(gp_params_r2, gp_state_r2, m_sphere)

# %%
var_r2 = jnp.diagonal(var_r2).T
var_L_r2 = jnp.linalg.cholesky(var_r2)

var_L_x_sphere = project(m_poisson, var_L_r2[:, 0, :], sphere_m_to_3d)[1]
var_L_z_sphere = project(m_poisson, -var_L_r2[:, 1, :], sphere_m_to_3d)[1]
var_normal_sphere = np.cross(var_L_x_sphere, var_L_z_sphere)
var_unit_normal_extrinsic = var_normal_sphere / np.expand_dims(
    np.linalg.norm(var_normal_sphere, axis=-1), axis=-1
)
var_cc_sphere = np.concatenate(
    [
        sphere_m_to_3d(m_poisson),
        var_unit_normal_extrinsic,
        var_L_x_sphere,
        var_L_z_sphere,
    ],
    axis=-1,
)
np.savetxt(os.path.join(data_path, "covariance_r2.csv"), var_cc_sphere, delimiter=",")

var_L_x_flat = project(m_poisson, var_L_r2[:, 0, :], sphere_flat_m_to_3d)[1]
var_L_z_flat = project(m_poisson, -var_L_r2[:, 1, :], sphere_flat_m_to_3d)[1]
var_normal_flat = np.cross(var_L_x_flat, var_L_z_flat)
var_unit_normal_extrinsic = var_normal_flat / np.expand_dims(
    np.linalg.norm(var_normal_flat, axis=-1), axis=-1
)
var_cc_sphere = np.concatenate(
    [
        sphere_flat_m_to_3d(m_poisson),
        var_unit_normal_extrinsic,
        var_L_x_flat,
        var_L_z_flat,
    ],
    axis=-1,
)
np.savetxt(
    os.path.join(data_path, "covariance_r2_flat.csv"), var_cc_sphere, delimiter=","
)


# %%
# Prediction on test locations (Spherical)
posterior_mean_s2, var_s2 = gp_s2(gp_params_s2, gp_state_s2, m_poisson)
_, posterior_cov_s2 = gp_s2(gp_params_s2, gp_state_s2, m_sphere)
var_s2 = jnp.diagonal(var_s2).T
var_L_s2 = jnp.linalg.cholesky(var_s2)

# %%

var_L_x_sphere = project(m_poisson, var_L_s2[:, 0, :], sphere_m_to_3d)[1]
var_L_z_sphere = project(m_poisson, -var_L_s2[:, 1, :], sphere_m_to_3d)[1]
var_normal_sphere = np.cross(var_L_x_sphere, var_L_z_sphere)
var_unit_normal_extrinsic = var_normal_sphere / np.expand_dims(
    np.linalg.norm(var_normal_sphere, axis=-1), axis=-1
)
var_cc_sphere = np.concatenate(
    [
        sphere_m_to_3d(m_poisson),
        var_unit_normal_extrinsic,
        var_L_x_sphere,
        var_L_z_sphere,
    ],
    axis=-1,
)
np.savetxt(os.path.join(data_path, "covariance_s2.csv"), var_cc_sphere, delimiter=",")

var_L_x_flat = project(m_poisson, var_L_s2[:, 0, :], sphere_flat_m_to_3d)[1]
var_L_z_flat = project(m_poisson, -var_L_s2[:, 1, :], sphere_flat_m_to_3d)[1]
var_normal_flat = np.cross(var_L_x_flat, var_L_z_flat)
var_unit_normal_extrinsic = var_normal_flat / np.expand_dims(
    np.linalg.norm(var_normal_flat, axis=-1), axis=-1
)
var_cc_sphere = np.concatenate(
    [
        sphere_flat_m_to_3d(m_poisson),
        var_unit_normal_extrinsic,
        var_L_x_flat,
        var_L_z_flat,
    ],
    axis=-1,
)
np.savetxt(
    os.path.join(data_path, "covariance_s2_flat.csv"), var_cc_sphere, delimiter=","
)


# %%
np.savetxt(
    os.path.join(data_path, "mean_r2.csv"),
    jnp.concatenate([*project(m_poisson, posterior_mean_r2, sphere_m_to_3d)], axis=-1),
    delimiter=",",
)
np.savetxt(
    os.path.join(data_path, "mean_s2.csv"),
    jnp.concatenate([*project(m_poisson, posterior_mean_s2, sphere_m_to_3d)], axis=-1),
    delimiter=",",
)
np.savetxt(
    os.path.join(data_path, "tracks.csv"),
    jnp.concatenate([*project(m_cond, v_cond, sphere_m_to_3d)], axis=-1),
    delimiter=",",
)

np.savetxt(
    os.path.join(data_path, "mean_r2_flat.csv"),
    jnp.concatenate(
        [*project(m_poisson, posterior_mean_r2, sphere_flat_m_to_3d)], axis=-1
    ),
    delimiter=",",
)
np.savetxt(
    os.path.join(data_path, "mean_s2_flat.csv"),
    jnp.concatenate(
        [*project(m_poisson, posterior_mean_s2, sphere_flat_m_to_3d)], axis=-1
    ),
    delimiter=",",
)
np.savetxt(
    os.path.join(data_path, "tracks_flat.csv"),
    jnp.concatenate([*project(m_cond, v_cond, sphere_flat_m_to_3d)], axis=-1),
    delimiter=",",
)

np.savetxt(
    os.path.join(data_path, "s_r2.csv"),
    np.diag(np.trace(posterior_cov_r2, axis1=2, axis2=3)),
    delimiter=",",
)
np.savetxt(
    os.path.join(data_path, "s_s2.csv"),
    np.diag(np.trace(posterior_cov_s2, axis1=2, axis2=3)),
    delimiter=",",
)

# %%
save_obj(
    mesh_to_obj(
        *mesh_to_polyscope(
            sphere_m_to_3d(m_sphere).reshape((65, 34, -1)), wrap_x=False, wrap_y=False
        ),
        uv_coords=m_sphere / jnp.array([jnp.pi, 2 * jnp.pi]),
    ),
    os.path.join(data_path, "earth.obj"),
)

save_obj(
    mesh_to_obj(
        *mesh_to_polyscope(
            sphere_flat_m_to_3d(m_sphere).reshape((65, 34, -1)),
            wrap_x=False,
            wrap_y=False,
        ),
        uv_coords=m_sphere / jnp.array([jnp.pi, 2 * jnp.pi]),
    ),
    os.path.join(data_path, "earth_flat.obj"),
)
# %%
np.savetxt(
    os.path.join(data_path, "earth_line.csv"),
    sphere_m_to_3d(m_sphere).reshape((65, 34, -1))[0, :, :],
    delimiter=",",
)
np.savetxt(
    os.path.join(data_path, "earth_flat_line.csv"),
    sphere_flat_m_to_3d(m_sphere).reshape((65, 34, -1))[0, :, :],
    delimiter=",",
)
np.savetxt(
    os.path.join(data_path, "earth_flat_line2.csv"),
    sphere_flat_m_to_3d(m_sphere).reshape((65, 34, -1))[-1, :, :],
    delimiter=",",
)

# %%
