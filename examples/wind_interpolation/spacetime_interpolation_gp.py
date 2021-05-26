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
from skyfield.api import load, EarthSatellite
from copy import deepcopy
import click
import pickle
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import xesmf as xe

def plot(
    fname1,
    fname2,
    m,
    mean,
    m_cond,
    v_cond,
    num_hours,
    ):
    m = np.asarray(m)
    m_cond = np.asarray(m_cond)
    v_cond = np.asarray(v_cond)
    mean = np.asarray(mean)

    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0-(5.625/2)))
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    scale = 250

    num_points = m[:,1].shape[0] // num_hours
    ax.quiver(rad2deg(m[:num_points,1]),
              rad2deg(m[:num_points,0], offset=jnp.pi/2), #export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/opt/cuda/cuda-11.0
              mean[:num_points,1],
              mean[:num_points,0],
              alpha=0.5,
              color='blue',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=2,
              transform=ccrs.PlateCarree())

    num_points = m_cond[:,1].shape[0] // num_hours
    ax.quiver(rad2deg(m_cond[:num_points,1]),
              rad2deg(m_cond[:num_points,0], offset=jnp.pi/2),
              v_cond[:num_points,1],
              v_cond[:num_points,0],
              color='red',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=3,
              transform=ccrs.PlateCarree())

    plt.savefig(fname1)


@click.command()
@click.option('--logdir', default='log', type=str)
@click.option('--geometry', '-g', default='r2', type=click.Choice(['r2','s2']))
@click.option('--num-hours',  '-h', default=6, type=int)
def main(logdir, geometry, num_hours):
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

    # Set up kernel
    if geometry == 'r2':
        space_kernel = TFPKernel(tfk.ExponentiatedQuadratic, 2, 2)
    elif geometry == 's2':
        S2 = EmbeddedS2(1.0)
        space_kernel = ManifoldProjectionVectorKernel(
                MaternCompactRiemannianManifoldKernel(1.5, S2, 120), S2
            ) # 144 is the maximum number of basis functions we have implemented

    time_kernel = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)

    kernel = ScaledKernel(ProductKernel(space_kernel, time_kernel)) # Space-time kernel

    gp = GaussianProcess(kernel)

    # Initialize parameters and state
    kernel_params = kernel.init_params(next(rng))
    product_params = kernel_params.sub_kernel_params
    space_params = product_params.sub_kernel_params[0]
    time_params = product_params.sub_kernel_params[1]
    if geometry == "r2":
        space_params = space_params._replace(log_length_scales=-2.7)
        time_params = time_params._replace(log_length_scales=1.8)
        kernel_params = kernel_params._replace(log_amplitude=2.3)
    elif geometry == "s2":
        space_params = space_params._replace(log_length_scale=-1.68)
        time_params = time_params._replace(log_length_scales=1.8)
        kernel_params = kernel_params._replace(log_amplitude=10.5)
    product_params = product_params._replace(sub_kernel_params=[space_params, time_params])
    kernel_params = kernel_params._replace(sub_kernel_params=product_params)

    # Initialize GP
    gp_params, gp_state = gp.init_params_with_state(next(rng))
    gp_params = gp_params._replace(kernel_params=kernel_params)
    gp_state = gp.condition(gp_params, m_cond, v_cond, noises_cond)

    mean, K = gp(gp_params, gp_state, m)

    # Plot
    fname1 = "figs/"+geometry+"_gp_mean_spacetime.png"
    fname2 = "figs/"+geometry+"_gp_std_spacetime.png"

    plot(fname1, fname2, m, mean, m_cond, v_cond, num_hours)


if __name__ == "__main__":
    main()