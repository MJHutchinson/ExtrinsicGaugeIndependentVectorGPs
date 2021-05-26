import xarray as xr
import numpy as np
import jax
import optax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from functools import partial
from dataclasses import dataclass
from riemannianvectorgp.sparse_gp import SparseGaussianProcess, SparseGaussianProcessParameters, SparseGaussianProcessState
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel,
    ProductKernel
)
from riemannianvectorgp.utils import train_sparse_gp, GlobalRNG
from examples.wind_interpolation.utils import deg2rad, rad2deg, refresh_kernel, GetDataAlongSatelliteTrack
from examples.wind_interpolation.plot import space_time_plot as plot
from skyfield.api import load, EarthSatellite
from copy import deepcopy
import click
import pickle
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import xesmf as xe

def train_sparse_gp_with_fixed_inducing_points(
    gp, gp_params, gp_state, m_cond, v_cond, rng, epochs=300, b1=0.9, b2=0.999, eps=1e-8, lr=0.01
):
    opt = optax.chain(optax.scale_by_adam(b1=b1, b2=b2, eps=eps), optax.scale(-lr))
    opt_state = opt.init(gp_params)
    debug_params = [gp_params]
    debug_states = [gp_state]
    debug_keys = [rng.key]
    losses = []
    for i in range(epochs):
        ((train_loss, gp_state), grads) = jax.value_and_grad(gp.loss, has_aux=True)(
            gp_params, gp_state, next(rng), m_cond, v_cond, m_cond.shape[0]
        )

        grads_ = SparseGaussianProcessParameters(
            log_error_stddev=grads.log_error_stddev,
            inducing_locations=jnp.zeros_like(grads.inducing_locations),
            inducing_pseudo_mean=jnp.zeros_like(grads.inducing_pseudo_mean),
            inducing_pseudo_log_err_stddev=jnp.zeros_like(grads.inducing_pseudo_log_err_stddev),
            kernel_params=grads.kernel_params
        )

        (updates, opt_state) = opt.update(grads_, opt_state)
        gp_params = optax.apply_updates(gp_params, updates)

        if i <= 10 or i % 20 == 0:
            print(i, "Loss:", train_loss)

        losses.append(train_loss)
        debug_params.append(gp_params)
        debug_states.append(gp_state)
        debug_keys.append(rng.key)

    return gp_params, gp_state, (debug_params, debug_states, debug_keys, losses)


@click.command()
@click.option('--logdir', default='log', type=str)
@click.option('--epochs', '-e', default=500, type=int)
@click.option('--geometry', '-g', default='r2', type=click.Choice(['r2','s2']))
@click.option('--num-hours',  '-h', default=6, type=int)
@click.option('--train/--no-train', default=True, type=bool)
def main(logdir, epochs, geometry, num_hours, train):
    rng = GlobalRNG()

    year = 2019
    month = 1
    day = 1
    hour = 0

    # Get Aeolus trajectory data from TLE set
    ts = load.timescale()
    line1 = '1 43600U 18066A   21112.99668353  .00040037  00000-0  16023-3 0  9999'
    line2 = '2 43600  96.7174 120.6934 0007334 114.6816 245.5221 15.86410481154456'
    aeolus = EarthSatellite(line1, line2, 'AEOLUS', ts)

    # Load ERA5 data and regrid
    resolution = 5.625
    ds = xr.open_mfdataset('../../datasets/era5_dataset/*.nc')
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+resolution/2, 90, resolution)),
            'lon': (['lon'], np.arange(0, 360, resolution)),
        }
    )
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    regridder = xe.Regridder(ds, grid_out, 'bilinear', periodic=True)
    ds = regridder(ds)

    # Load weekly climatology
    climatology = np.load("../../datasets/climatology/weekly_climatology.npz")

    # Get input locations
    lon = ds.isel(time=0).lon
    lat = ds.isel(time=0).lat
    time = np.arange(num_hours)
    mesh = np.meshgrid(lon, lat) # create mesh for plotting later
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
    m = jnp.stack([lat, lon, time], axis=-1)

    # Get conditioning points and values
    m_cond, v_cond = GetDataAlongSatelliteTrack(ds,
                                                aeolus,
                                                year,
                                                month,
                                                day,
                                                hour,
                                                num_hours=num_hours,
                                                anomaly=True,
                                                climatology=climatology,
                                                space_time=True)
    noises_cond = jnp.ones_like(v_cond) * 1.7

    if train:
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
        sparse_gp = SparseGaussianProcess(
                        kernel=kernel,
                        num_inducing=m_cond.shape[0],
                        num_basis=67,
                        num_samples=20)

        # Initialize parameters and state
        kernel_params = kernel.init_params(next(rng))
        product_params = kernel_params.sub_kernel_params
        space_params = product_params.sub_kernel_params[0]
        if geometry == "r2":
            space_params = space_params._replace(log_length_scales=jnp.log(0.3))
        elif geometry == "s2":
            space_params = space_params._replace(log_length_scale=jnp.log(0.3))
        time_params = product_params.sub_kernel_params[1]
        time_params = time_params._replace(log_length_scales=jnp.log(1.0))
        product_params = product_params._replace(sub_kernel_params=[space_params, time_params])
        kernel_params = kernel_params._replace(sub_kernel_params=product_params)
        kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0]))

        params, state = sparse_gp.init_params_with_state(next(rng))

        params = params._replace(kernel_params=kernel_params)
        params = sparse_gp.set_inducing_points(params, m_cond, v_cond, noises_cond)

        state = sparse_gp.resample_prior_basis(params, state, next(rng))
        state = sparse_gp.randomize(params, state, next(rng))
        

        # Train sparse GP
        params, state, _ = train_sparse_gp_with_fixed_inducing_points(
                            sparse_gp,
                            params,
                            state,
                            m_cond,
                            v_cond,
                            rng,
                            epochs=epochs)

        # Save params and state
        with open(logdir+"/"+geometry+"_params_and_state_for_space_time_interpolation.pickle", "wb") as f:
            pickle.dump({"params": params, "state": state}, f)

        # Save sparse gp
        with open(logdir+"/"+geometry+"_sparse_gp_for_space_time_interpolation.pickle", "wb") as f:
            pickle.dump(sparse_gp, f)

        # Print results
        kernel_params = params.kernel_params
        product_params = kernel_params.sub_kernel_params
        space_params = product_params.sub_kernel_params[0]
        time_params = product_params.sub_kernel_params[1]
        if geometry == "r2":
            print("spatial length scale:", space_params.log_length_scales)
        elif geometry == "s2":
            print("spatial length scale:", space_params.log_length_scale)
        print("temporal length scale:", time_params.log_length_scales)
        print("amplitude:", kernel_params.log_amplitude)

    else:
        # Load params and state
        with open(logdir+"/"+geometry+"_params_and_state_for_space_time_interpolation.pickle", "rb") as f:
            params_and_state = pickle.load(f)

        params = params_and_state['params']
        state = params_and_state['state']

        # Load sparse gp
        with open(logdir+"/"+geometry+"_sparse_gp_for_space_time_interpolation.pickle", "rb") as f:
            sparse_gp = pickle.load(f)

    final_loss, _ = sparse_gp.loss(params, state, next(rng), m_cond, v_cond, m_cond.shape[0])
    print("-"*10)
    print(f"Final loss: {final_loss}")
    print("-"*10)

    # Plot
    fname1 = "figs/"+geometry+"_sparse_gp_mean_spacetime.png"
    fname2 = "figs/"+geometry+"_sparse_gp_std_spacetime.png"
    prediction = sparse_gp(params, state, m)

    plot(fname1, fname2, m, prediction, m_cond, v_cond, num_hours)


if __name__ == "__main__":
    main()