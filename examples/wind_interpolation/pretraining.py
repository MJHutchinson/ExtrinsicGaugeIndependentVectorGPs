import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
import sys; sys.path.insert(0, '..')
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel
)
from riemannianvectorgp.utils import train_sparse_gp, GlobalRNG
import click
import pickle

def _get_v_cond(ds, date, climatology):
    u = ds.u10.sel(time=date)
    v = ds.v10.sel(time=date)
    week_number = u['time'].dt.isocalendar().week
    u_mean = climatology['u'][week_number]
    v_mean = climatology['v'][week_number]
    u_anomaly = u.values - u_mean
    v_anomaly = v.values - v_mean
    u_anomaly, v_anomaly = u_anomaly.transpose().flatten(), v_anomaly.transpose().flatten()
    v_cond = np.stack([v_anomaly, u_anomaly], axis=-1)
    return v_cond


def _refresh_kernel(key, kernel, init_log_length_scale, m, geometry):
    kernel_params = kernel.init_params(key)
    sub_kernel_params = kernel_params.sub_kernel_params
    if geometry == 'r2':
        sub_kernel_params = sub_kernel_params._replace(log_length_scales=init_log_length_scale)
    if geometry == 's2':
        sub_kernel_params = sub_kernel_params._replace(log_length_scale=init_log_length_scale)
    kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
    kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0]))
    return kernel_params


def _deg2rad(x: np.ndarray, offset: float=0.):
    return (np.pi/180)*x + offset


def _rad2deg(x: np.ndarray, offset: float=0.):
    return (180/np.pi)*(x - offset)
    

@click.command()
@click.option('--logdir', default='log', type=str)
@click.option('--samples', '-s', default=50, type=int)
@click.option('--epochs', '-e', default=500, type=int)
@click.option('--geometry', '-g', default='r2', type=click.Choice(['r2', 's2']))
def main(logdir, samples, epochs, geometry):
    rng = GlobalRNG()

    # Load past reanalysis data
    ds = xr.open_mfdataset('../../datasets/weatherbench_wind_data/*.nc')
    total_length = ds.dims['time']
    idxs = jnp.arange(total_length)
    idxs = jr.permutation(next(rng), idxs) # Shuffle indices

    # Load climatology
    climatology = np.load("../../datasets/climatology/weekly_climatology.npz")

    # Get input locations
    lon = ds.isel(time=0).lon
    lat = ds.isel(time=0).lat
    lat, lon = jnp.meshgrid(_deg2rad(lat, offset=jnp.pi/2), _deg2rad(lon)) # Reparametrise as lat=(0, pi) and lon=(0, 2pi) 
    lat = lat.flatten()
    lon = lon.flatten()
    m_cond = jnp.stack([lat, lon], axis=-1)

    # Set up kernel
    if geometry == 'r2':
        kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 2, 2))
    elif geometry == 's2':
        S2 = EmbeddedS2(1.0)
        kernel = ScaledKernel(
            ManifoldProjectionVectorKernel(
                MaternCompactRiemannianManifoldKernel(1.5, S2, 144), S2
            )
        ) # 144 is the maximum number of basis functions we have implemented

    # Set up sparse GP
    num_points = 15
    sparse_gp = SparseGaussianProcess(
                    kernel=kernel,
                    num_inducing=num_points**2,
                    num_basis=144,
                    num_samples=10)

    # Set initial inducing locations on a regular grid
    lat_init = jnp.linspace(0, jnp.pi, num_points)
    lon_init = jnp.linspace(0, 2*jnp.pi, num_points)
    phi_init, theta_init = jnp.meshgrid(lat_init, lon_init)
    phi_init, theta_init = phi_init.flatten(), theta_init.flatten()
    init_inducing_locations = jnp.stack([phi_init, theta_init], axis=-1)

    # Set initial length scale
    init_length_scale = 0.2
    init_log_length_scale = jnp.log(init_length_scale)

    log_length_scales, log_amplitudes = [], []
    for i, idx in enumerate(idxs[:samples]):
        date = ds.time[idx]
        print("Sample:", i, "Date:", date.values)

        # Initialise parameters and state
        params, state = sparse_gp.init_params_with_state(next(rng))
        kernel_params = _refresh_kernel(next(rng), kernel, init_log_length_scale, m_cond, geometry)

        params = params._replace(kernel_params=kernel_params)
        params = params._replace(inducing_locations=init_inducing_locations)
        
        state = sparse_gp.resample_prior_basis(params, state, next(rng))
        state =  sparse_gp.randomize(params, state, next(rng))

        # Get conditioning values
        v_cond = _get_v_cond(ds, date, climatology)

        # Train sparse GP
        params, state, _ = train_sparse_gp(sparse_gp, params, state, m_cond, v_cond, rng, epochs=epochs)

        if geometry == 'r2':
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scales
            log_amplitude = params.kernel_params.log_amplitude
        elif geometry == 's2':
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scale
            log_amplitude = params.kernel_params.log_amplitude

        print("Log length scale:", log_length_scale, "Log amplitude:", log_amplitude)

        log_length_scales.append(log_length_scale)
        log_amplitudes.append(log_amplitude)

    with open(logdir+'/'+geometry+'_params.pickle', "wb") as f:
        pickle.dump({'log_length_scale': np.stack(log_length_scales), 'log_amplitude': np.stack(log_amplitudes)}, f)


if __name__ == '__main__':
    main()