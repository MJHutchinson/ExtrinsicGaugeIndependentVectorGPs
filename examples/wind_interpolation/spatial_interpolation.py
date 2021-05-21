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
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel,
    AbstractKernel
)
from riemannianvectorgp.utils import train_sparse_gp, GlobalRNG
from examples.wind_interpolation.utils import deg2rad, rad2deg, refresh_kernel, GetDataAlongSatelliteTrack
from skyfield.api import load, EarthSatellite
from copy import deepcopy
import click
import pickle
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import xesmf as xe


class Hyperprior():
    def __init__(self):
        self.amplitude = None
        self.length_scale = None

    @dataclass
    class mean_and_std:
        mean: float
        std: float

    def set_amplitude_prior(self, mean, std):
        self.amplitude = self.mean_and_std(mean, std)

    def set_length_scale_prior(self, mean, std):
        self.length_scale = self.mean_and_std(mean, std)


class SparseGaussianProcessWithHyperprior(SparseGaussianProcess):
    def __init__(
        self,
        kernel: AbstractKernel,
        num_inducing: int,
        num_basis: int,
        num_samples: int,
        hyperprior: Hyperprior,
        geometry: str,
    ):
        super().__init__(kernel, num_inducing, num_basis, num_samples)
        if geometry != 'r2' and geometry != 's2':
            raise ValueError("geometry must be either 'r2' or 's2'.")
        self._hyperprior = hyperprior
        self._geometry = geometry

    @partial(jax.jit, static_argnums=(0,))
    def hyperprior(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
    ) -> jnp.ndarray:
        """Returns the log hyperprior regularization term of the GP."""
        # Amplitude regularizer
        amp_mean = self._hyperprior.amplitude.mean
        amp_std = self._hyperprior.amplitude.std
        amp_reg = (params.kernel_params.log_amplitude - amp_mean)**2 / (2*amp_std**2)
        # Length scale regularizer
        scale_mean = self._hyperprior.length_scale.mean
        scale_std = self._hyperprior.length_scale.std
        if self._geometry == 'r2':
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scales
        elif self._geometry == 's2':
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scale
        scale_reg = (log_length_scale - scale_mean)**2 / (2*scale_std**2)
        return amp_reg + scale_reg


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


def plot(fname1, fname2, gp, params, state, m, m_cond, v_cond, mesh, lon_size, lat_size):
    prediction = gp(params, state, m)

    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    scale = 250
    
    mean = prediction.mean(axis=0)
    std = jnp.sqrt(((prediction - mean)**2).mean(axis=0).sum(axis=-1))
    plt.quiver(rad2deg(m[:,1]),
               rad2deg(m[:,0], offset=jnp.pi/2),
               mean[:,1],
               mean[:,0],
               alpha=0.5,
               color='blue',
               scale=scale,
               width=0.003,
               headwidth=3,
               zorder=2)

    plt.quiver(rad2deg(m_cond[:,1]),
               rad2deg(m_cond[:,0], offset=jnp.pi/2),
               v_cond[:,1],
               v_cond[:,0],
               color='red',
               scale=scale,
               width=0.003,
               headwidth=3,
               zorder=3)

    # Plot satellite trajectories (we split it in two parts to respect periodicity)
    def _where_is_jump(x):
        for i in range(1,len(x)):
            if np.abs(x[i-1] - x[i]) > 180:
                return i

    idx = _where_is_jump(rad2deg(m_cond[:, 1]))

    x1 = deepcopy(rad2deg(m_cond[:idx+1, 1]))
    y1 = rad2deg(m_cond[:idx+1, 0], offset=jnp.pi/2)
    x1[idx] = x1[idx] - 360

    x2 = deepcopy(rad2deg(m_cond[idx-1:, 1]))
    y2 = rad2deg(m_cond[idx-1:, 0], offset=jnp.pi/2)
    x2[0] = x2[0] + 360

    plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2)
    plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2)

    plt.savefig(fname1)

    std = std.reshape(lon_size, lat_size).transpose()
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.contourf(*mesh, std, levels=30, zorder=1)

    plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2)
    plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2)

    plt.title("posterior std")
    plt.savefig(fname2)


@click.command()
@click.option('--logdir', default='log', type=str)
@click.option('--geometry', '-g', default='r2', type=click.Choice(['r2','s2']))
@click.option('--epochs', '-e', default=500, type=int)
def main(logdir, epochs, geometry):
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
    with open(logdir+'/'+geometry+'_params.pickle', "rb") as f:
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

    # Plot posterior mean
    fname1 = "figs/"+geometry+"_sparse_gp_mean.png"
    fname2 = "figs/"+geometry+"_sparse_gp_std.png"
    plot(fname1, fname2, sparse_gp, params, state, m, m_cond, v_cond, mesh, lon_size, lat_size)

    # Save params and state
    with open(logdir+"/"+geometry+"_params_and_state_for_spatial_interpolation.pickle", "wb") as f:
        pickle.dump({"params": params, "state": state}, f)


if __name__ == '__main__':
    main()


