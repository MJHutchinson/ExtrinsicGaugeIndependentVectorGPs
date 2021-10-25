# %%
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
    TFPKernel
)
from riemannianvectorgp.utils import GlobalRNG
from examples.wind_interpolation.utils import deg2rad, rad2deg
import pickle

# %%
rng1 = GlobalRNG()
rng2 = GlobalRNG()

year = 2019
month = 1
day = 1
hour = 9

lat_size = 32
lon_size = 64

# %%
# Load data
prior_mean = np.load("log/data_for_plots/prior_mean.npy") # Climatological data
m_cond = np.load("log/data_for_plots/m_cond.npy") # Conditioning locations
v_cond_anomaly = np.load("log/data_for_plots/v_cond.npy") # Conditioning values 1 (offset from climatological mean)
v_cond_full = np.load("log/data_for_plots/v_cond_full.npy") # Conditioning values 2 (full velocity without subtracting mean)
m = np.load("log/data_for_plots/m.npy") # Test locations

noises_cond = jnp.ones_like(v_cond_anomaly) * 1.7 

# %%
# Setup Euclidean GP
kernel_r2 = ScaledKernel(TFPKernel(tfk.MaternThreeHalves, 2, 2))
gp_r2 = GaussianProcess(kernel_r2)

# Setup Spherical GP
S2 = EmbeddedS2(1.0)
kernel_s2 = ScaledKernel(
                ManifoldProjectionVectorKernel(
                    MaternCompactRiemannianManifoldKernel(3/2, S2, 144), S2
                ) # 144 is the maximum number of basis functions we have implemented
            )
gp_s2 = GaussianProcess(kernel_s2)

# %%
# Set length scale and amplitudes
log_length_scale = -1.63
log_amplitude_r2 = 2.2
log_amplitude_s2 = 11.5

# Refresh r2 kernel
kernel_params_r2 = kernel_r2.init_params(rng1)
sub_kernel_params_r2 = kernel_params_r2.sub_kernel_params
sub_kernel_params_r2 = sub_kernel_params_r2._replace(log_length_scales=log_length_scale)
kernel_params_r2 = kernel_params_r2._replace(sub_kernel_params=sub_kernel_params_r2)
kernel_params_r2 = kernel_params_r2._replace(log_amplitude=log_amplitude_r2)

# Refresh s2 kernel
kernel_params_s2 = kernel_s2.init_params(rng2)
sub_kernel_params_s2 = kernel_params_s2.sub_kernel_params
sub_kernel_params_s2 = sub_kernel_params_s2._replace(log_length_scale=log_length_scale)
kernel_params_s2 = kernel_params_s2._replace(sub_kernel_params=sub_kernel_params_s2)
kernel_params_s2 = kernel_params_s2._replace(log_amplitude=log_amplitude_s2)

# Initialize GPs
gp_params_r2, gp_state_r2 = gp_r2.init_params_with_state(next(rng1))
gp_params_r2 = gp_params_r2._replace(kernel_params=kernel_params_r2)

gp_params_s2, gp_state_s2 = gp_s2.init_params_with_state(next(rng2))
gp_params_s2 = gp_params_s2._replace(kernel_params=kernel_params_s2)

# %%
# Fit and save Euclidean GP
gp_state_r2 = gp_r2.condition(gp_params_r2, m_cond, v_cond_anomaly, noises_cond)

with open('log/model/gp_r2.pickle', "wb") as f:
        pickle.dump(gp_r2, f)

with open('log/model/gp_params_r2.pickle', "wb") as f:
    pickle.dump(gp_params_r2, f)

with open('log/model/gp_state_s2.pickle', "wb") as f:
    pickle.dump(gp_state_r2, f)

# %%
# Fit and save Spherical GP
gp_state_s2 = gp_s2.condition(gp_params_s2, m_cond, v_cond_anomaly, noises_cond)

with open('log/model/gp_s2.pickle', "wb") as f:
        pickle.dump(gp_s2, f)

with open('log/model/gp_params_s2.pickle', "wb") as f:
    pickle.dump(gp_params_s2, f)

with open('log/model/gp_state_s2.pickle', "wb") as f:
    pickle.dump(gp_state_s2, f)

# %%
# Prediction on test locations (Euclidean)
posterior_mean_r2, posterior_cov_r2 = gp_r2(gp_params_r2, gp_state_r2, m)
posterior_mean_r2 = posterior_mean_r2.reshape(lat_size*lon_size, 2)
posterior_mean_r2 += prior_mean

# %%
# Prediction on test locations (Spherical)
posterior_mean_s2, posterior_cov_s2 = gp_s2(gp_params_s2, gp_state_s2, m)
posterior_mean_s2 = posterior_mean_s2.reshape(lat_size*lon_size, 2)
posterior_mean_s2 += prior_mean

# %%
