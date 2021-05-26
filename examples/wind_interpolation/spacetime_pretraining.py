import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel,
    ProductKernel
)
from riemannianvectorgp.utils import train_sparse_gp, GlobalRNG
from examples.wind_interpolation.utils import deg2rad, refresh_kernel
import matplotlib.pyplot as plt
import click
import pickle

def _get_v_cond(ds: xr.Dataset, climatology: np.ndarray, date: np.datetime64, num_hours: int):
    u = ds.u10.sel(time=slice(date, date+np.timedelta64(num_hours, 'h')))
    v = ds.v10.sel(time=slice(date, date+np.timedelta64(num_hours, 'h')))
    week_numbers = u['time'].dt.isocalendar().week
    u, v = u.values, v.values
    anomaly_list = []
    for t in range(num_hours):
        u_mean = climatology['u'][week_numbers[t]-1]
        v_mean = climatology['v'][week_numbers[t]-1]
        u_anomaly = u[t] - u_mean
        v_anomaly = v[t] - v_mean
        u_anomaly, v_anomaly = u_anomaly.transpose().flatten(), v_anomaly.transpose().flatten()
        anomaly_list.append(jnp.stack([v_anomaly, u_anomaly], axis=-1))
    v_cond = jnp.concatenate(anomaly_list)
    return v_cond
    

@click.command()
@click.option('--logdir', default='log', type=str)
@click.option('--samples', '-s', default=100, type=int)
@click.option('--epochs', '-e', default=500, type=int)
@click.option('--geometry', '-g', default='r2', type=click.Choice(['r2', 's2']))
@click.option('--num-hours',  '-h', default=6, type=int)
def main(logdir, samples, epochs, geometry, num_hours):
    rng = GlobalRNG()

    # Load past reanalysis data
    ds = xr.open_mfdataset('../../datasets/weatherbench_wind_data/*.nc')
    total_length = ds.dims['time']
    idxs = jnp.arange(total_length - num_hours)
    idxs = jr.permutation(next(rng), idxs) # Shuffle indices

    # Load climatology
    climatology = np.load("../../datasets/climatology/weekly_climatology.npz")

    # Get input locations
    lon = ds.isel(time=0).lon
    lat = ds.isel(time=0).lat
    time = np.arange(num_hours)
    lon_size = lon.shape[0]
    lat_size = lat.shape[0]
    # Reparametrise as lat=(0, pi) and lon=(0, 2pi)
    lat = deg2rad(lat, offset=jnp.pi/2)
    lon = deg2rad(lon)

    lat, lon, time = jnp.meshgrid(lat, lon, time)
    lat = lat.reshape(lat_size*lon_size, num_hours).transpose()
    lon = lon.reshape(lat_size*lon_size, num_hours).transpose()
    time = time.reshape(lat_size*lon_size, num_hours).transpose()
    lat = lat.flatten()
    lon = lon.flatten()
    time = time.flatten()
    m_cond = jnp.stack([lat, lon, time], axis=-1)

    # Set up kernel
    if geometry == 'r2':
        space_kernel = TFPKernel(tfk.ExponentiatedQuadratic, 2, 2)
    elif geometry == 's2':
        S2 = EmbeddedS2(1.0)
        space_kernel = ManifoldProjectionVectorKernel(
                MaternCompactRiemannianManifoldKernel(1.5, S2, 144), S2
            ) # 144 is the maximum number of basis functions we have implemented

    time_kernel = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)

    kernel = ScaledKernel(ProductKernel(space_kernel, time_kernel)) # Space-time kernel

    # Set up sparse GP
    num_points_lat = 10
    num_points_lon = 20
    num_points_time = 6
    sparse_gp = SparseGaussianProcess(
                    kernel=kernel,
                    num_inducing=num_points_lat * num_points_lon * num_points_time,
                    num_basis=67,
                    num_samples=10)

    # Set initial inducing locations on a regular grid
    lat_init = jnp.linspace(0, jnp.pi, num_points_lat)
    lon_init = jnp.linspace(0, 2*jnp.pi, num_points_lon)
    time_init = jnp.linspace(0, num_hours, num_points_time)
    lat_init, lon_init, time_init = jnp.meshgrid(lat_init, lon_init, time_init)
    lat_init = lat_init.reshape(num_points_lat * num_points_lon, num_points_time).transpose()
    lon_init = lon_init.reshape(num_points_lat * num_points_lon, num_points_time).transpose()
    time_init = time_init.reshape(num_points_lat * num_points_lon, num_points_time).transpose()
    lat_init, lon_init, time_init = lat_init.flatten(), lon_init.flatten(), time_init.flatten()
    init_inducing_locations = jnp.stack([lat_init, lon_init, time_init], axis=-1)

    # Set initial length scale
    init_spatial_length_scale = 0.3
    init_temporal_length_scale = 1.0
    init_log_length_scales = {'space': jnp.log(init_spatial_length_scale), 'time': jnp.log(init_temporal_length_scale)}

    log_length_scales, log_amplitudes = [], []
    for i, idx in enumerate(idxs[:samples]):
        date = ds.time[idx].values
        print("Sample:", i, "Date:", date)

        # Initialise parameters and state
        kernel_params = kernel.init_params(next(rng))
        product_params = kernel_params.sub_kernel_params
        space_params = product_params.sub_kernel_params[0]
        if geometry == "r2":
            space_params = space_params._replace(log_length_scales=init_log_length_scales['space'])
        elif geometry == "s2":
            space_params = space_params._replace(log_length_scale=init_log_length_scales['space'])
        time_params = product_params.sub_kernel_params[1]
        time_params = time_params._replace(log_length_scales=init_log_length_scales['time'])
        product_params = product_params._replace(sub_kernel_params=[space_params, time_params])
        kernel_params = kernel_params._replace(sub_kernel_params=product_params)
        kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m_cond, m_cond)[0, 0, 0, 0]))

        params, state = sparse_gp.init_params_with_state(next(rng))

        params = params._replace(kernel_params=kernel_params)
        params = params._replace(inducing_locations=init_inducing_locations)
        
        state = sparse_gp.resample_prior_basis(params, state, next(rng))
        state =  sparse_gp.randomize(params, state, next(rng))

        # Get conditioning values
        v_cond = _get_v_cond(ds, climatology, date, num_hours)

        # Train sparse GP
        params, state, _ = train_sparse_gp(sparse_gp, params, state, m_cond, v_cond, rng, epochs=epochs)
        print(params.kernel_params)
        import pdb; pdb.set_trace()

        if geometry == 'r2':
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scales
            log_amplitude = params.kernel_params.log_amplitude
        elif geometry == 's2':
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scale
            log_amplitude = params.kernel_params.log_amplitude

        print("Log length scale:", log_length_scale, "Log amplitude:", log_amplitude)

        log_length_scales.append(log_length_scale)
        log_amplitudes.append(log_amplitude)

    log_length_scales = np.stack(log_length_scales)
    log_amplitudes = np.stack(log_amplitudes)

    plt.figure()
    plt.hist(log_length_scales)
    plt.savefig("figs/"+geometry+"log_length_scale_distribution.png")

    plt.figure()
    plt.hist(log_amplitudes)
    plt.savefig("figs/"+geometry+"log_amplitude_distribution.png")

    with open(logdir+'/'+geometry+'_params_pretrained.pickle', "wb") as f:
        pickle.dump({'log_length_scale': log_length_scales, 'log_amplitude': log_amplitudes}, f)


if __name__ == '__main__':
    main()