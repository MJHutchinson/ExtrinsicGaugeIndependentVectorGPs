import xarray as xr
import numpy as np
import jax
import optax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from riemannianvectorgp.sparse_gp import SparseGaussianProcess, SparseGaussianProcessParameters, SparseGaussianProcessState
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel
)
from riemannianvectorgp.utils import GlobalRNG
from examples.wind_interpolation.utils import (
    deg2rad,
    rad2deg,
    refresh_kernel,
    GetDataAlongSatelliteTrack,
    Hyperprior,
    SparseGaussianProcessWithHyperprior
    )
from examples.wind_interpolation.plot import spatial_plot as plot
from skyfield.api import load, EarthSatellite
from copy import deepcopy
import click
import pickle
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import xesmf as xe


def train_sparse_gp_with_fixed_inducing_points(
    gp, gp_params, gp_state, m_cond, v_cond, rng, geometry, epochs=300, b1=0.9, b2=0.999, eps=1e-8, lr=0.01
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

        if geometry == 'r2':
            log_length_scale = gp_params.kernel_params.sub_kernel_params.log_length_scales
        elif geometry == 's2':
            log_length_scale = gp_params.kernel_params.sub_kernel_params.log_length_scale

        if i <= 10 or i % 20 == 0:
            print(i, "Loss:", train_loss,
                     "Length scale:", log_length_scale,
                     "Amplitude:", gp_params.kernel_params.log_amplitude)

        losses.append(train_loss)
        debug_params.append(gp_params)
        debug_states.append(gp_state)
        debug_keys.append(rng.key)

    return gp_params, gp_state, (debug_params, debug_states, debug_keys, losses)


@click.command()
@click.option('--logdir', default='log', type=str)
@click.option('--geometry', '-g', default='r2', type=click.Choice(['r2','s2']))
@click.option('--epochs', '-e', default=500, type=int)
@click.option('--train/--no-train', default=True, type=bool)
def main(logdir, epochs, geometry, train):
    rng = GlobalRNG()

    year = 2019
    month = 1
    day = 1
    hour = 0
    min = 0
    date = f"{year}-{month}-{day} {hour}:{min}"

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
    mesh = np.meshgrid(lon, lat) # create mesh for plotting later
    lon_size = lon.shape[0]
    lat_size = lat.shape[0]
    lat, lon = jnp.meshgrid(deg2rad(lat, offset=jnp.pi/2), deg2rad(lon)) # Reparametrise as lat=(0, pi) and lon=(0, 2pi) 
    lat = lat.flatten()
    lon = lon.flatten()
    m = jnp.stack([lat, lon], axis=-1)

    # Get conditioning points and values
    m_cond, v_cond = GetDataAlongSatelliteTrack(ds, aeolus, climatology, year, month, day, hour)
    noises_cond = jnp.ones_like(v_cond) * 1.7

    if train:
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

        # Set up hyperprior regularizer
        with open(logdir+'/'+geometry+'_params_pretrained.pickle', "rb") as f:
            params_dict = pickle.load(f)

        hyperprior_amplitudes = params_dict['log_amplitude']
        hyperprior_length_scales = params_dict['log_length_scale']

        hyperprior = Hyperprior()
        hyperprior.set_amplitude_prior(hyperprior_amplitudes.mean(), hyperprior_amplitudes.std())
        hyperprior.set_length_scale_prior(hyperprior_length_scales.mean(), hyperprior_length_scales.std())

        # Set up sparse GP
        sparse_gp = SparseGaussianProcessWithHyperprior(
                        kernel=kernel,
                        num_inducing=m_cond.shape[0],
                        num_basis=144,
                        num_samples=100,
                        hyperprior=hyperprior, 
                        geometry=geometry)

        # Set initial length scale
        init_log_length_scale = hyperprior.length_scale.mean
        init_log_amplitude = hyperprior.amplitude.mean

        # Initialize parameters and state
        params, state = sparse_gp.init_params_with_state(next(rng))
        kernel_params = refresh_kernel(next(rng), kernel, m_cond, geometry, init_log_length_scale, init_log_amplitude)

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
                            geometry,
                            epochs=epochs)

        # Save params and state
        with open(logdir+"/"+geometry+"_params_and_state_for_spatial_interpolation.pickle", "wb") as f:
            pickle.dump({"params": params, "state": state}, f)

        # Save sparse gp
        with open(logdir+"/"+geometry+"_sparse_gp_for_spatial_interpolation.pickle", "wb") as f:
            pickle.dump(sparse_gp, f)
    else:
        # Load params and state
        with open(logdir+"/"+geometry+"_params_and_state_for_spatial_interpolation.pickle", "rb") as f:
            params_and_state = pickle.load(f)

        params = params_and_state['params']
        state = params_and_state['state']

        # Load sparse gp
        with open(logdir+"/"+geometry+"_sparse_gp_for_spatial_interpolation.pickle", "rb") as f:
            sparse_gp = pickle.load(f)

    final_loss, _ = sparse_gp.loss(params, state, next(rng), m_cond, v_cond, m_cond.shape[0])
    print("-"*10)
    print(f"Final loss: {final_loss}")
    print("-"*10)

    # Plot
    fname1 = "figs/"+geometry+"_sparse_gp_mean.png"
    fname2 = "figs/"+geometry+"_sparse_gp_std.png"
    plot(fname1, fname2, sparse_gp, params, state, m, m_cond, v_cond, mesh, lon_size, lat_size)


if __name__ == '__main__':
    main()


