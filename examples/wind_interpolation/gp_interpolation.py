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
    ProductKernel
)
from riemannianvectorgp.utils import GlobalRNG
from examples.wind_interpolation.plot import *
from examples.wind_interpolation.utils import deg2rad, rad2deg, GetDataAlongSatelliteTrack
from skyfield.api import load, EarthSatellite
import click
import pickle
import xesmf as xe

def refresh_kernel(key, kernel, m, geometry, init_log_length_scale, init_log_amplitude=None, spacetime=False):
    if spacetime:
        kernel_params = kernel.init_params(key)
        product_params = kernel_params.sub_kernel_params
        space_params = product_params.sub_kernel_params[0]
        time_params = product_params.sub_kernel_params[1]
        if geometry == "r2":
            space_params = space_params._replace(log_length_scales=init_log_length_scale['space'])
            time_params = time_params._replace(log_length_scales=init_log_length_scale['time'])
        elif geometry == "s2":
            space_params = space_params._replace(log_length_scale=init_log_length_scale['space'])
            time_params = time_params._replace(log_length_scales=init_log_length_scale['time'])
        log_amplitude = -jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0]) if init_log_amplitude is None else init_log_amplitude
        kernel_params = kernel_params._replace(log_amplitude=log_amplitude)
        product_params = product_params._replace(sub_kernel_params=[space_params, time_params])
        kernel_params = kernel_params._replace(sub_kernel_params=product_params)
    else:
        kernel_params = kernel.init_params(key)
        sub_kernel_params = kernel_params.sub_kernel_params
        if geometry == 'r2':
            sub_kernel_params = sub_kernel_params._replace(log_length_scales=init_log_length_scale)
        if geometry == 's2':
            sub_kernel_params = sub_kernel_params._replace(log_length_scale=init_log_length_scale)
        kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
        log_amplitude = -jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0]) if init_log_amplitude is None else init_log_amplitude
        kernel_params = kernel_params._replace(log_amplitude=log_amplitude)
    return kernel_params


@click.command()
@click.option('--logdir', default='log', type=str)
@click.option('--geometry', '-g', default='s2', type=click.Choice(['r2','s2']))
@click.option('--num-hours',  '-h', default=6, type=int)
@click.option('--spacetime/--no-spacetime', default=False, type=bool)
@click.option('--plot-sphere/--no-plot-sphere', default=False, type=bool)
def main(logdir, geometry, num_hours, spacetime, plot_sphere):
    rng = GlobalRNG()

    year = 2019
    month = 1
    day = 1
    hour = 9

    num_hours = num_hours if spacetime else 1

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
    lon_size = lon.shape[0]
    lat_size = lat.shape[0]

    # Extend to include poles
    lon_ext = np.append(ds.isel(time=0).lon.values, np.array([360]))
    lat_ext = np.append(np.append(np.array([-90]), ds.isel(time=0).lat.values), np.array([90]))
    lon_size_ext = lon_ext.shape[0]
    lat_size_ext= lat_ext.shape[0]

    time = np.arange(num_hours)
    time_ext = time
    mesh = np.meshgrid(lon_ext, lat_ext) # create mesh for plotting later

    # Reparametrise as lat=(0, pi) and lon=(0, 2pi)
    lat = deg2rad(lat, offset=jnp.pi/2)
    lon = deg2rad(lon)

    lat_ext = deg2rad(lat_ext, offset=jnp.pi/2)
    lon_ext = deg2rad(lon_ext)

    if spacetime:
        lat, lon, time = jnp.meshgrid(lat, lon, time)
        lat = lat.reshape(lat_size*lon_size, num_hours).transpose()
        lon = lon.reshape(lat_size*lon_size, num_hours).transpose()
        time = time.reshape(lat_size*lon_size, num_hours).transpose()
        lat = lat.flatten()
        lon = lon.flatten()
        time = time.flatten()
        m = jnp.stack([lat, lon, time], axis=-1)

        lat_ext, lon_ext, time = jnp.meshgrid(lat_ext, lon_ext, time_ext)
        lat_ext = lat_ext.reshape(lat_size_ext*lon_size_ext, num_hours).transpose()
        lon_ext = lon_ext.reshape(lat_size_ext*lon_size_ext, num_hours).transpose()
        time_ext = time.reshape(lat_size_ext*lon_size_ext, num_hours).transpose()
        lat_ext = lat_ext.flatten()
        lon_ext = lon_ext.flatten()
        time_ext = time_ext.flatten()
        m_ext = jnp.stack([lat_ext, lon_ext, time_ext], axis=-1)

    else:
        lat, lon = jnp.meshgrid(lat, lon)
        lat = lat.flatten()
        lon = lon.flatten()
        m = jnp.stack([lat, lon], axis=-1)

        lat_ext, lon_ext = jnp.meshgrid(lat_ext, lon_ext)
        lat_ext = lat_ext.flatten()
        lon_ext = lon_ext.flatten()
        m_ext = jnp.stack([lat_ext, lon_ext], axis=-1)

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
                                                space_time=spacetime)
    noises_cond = jnp.ones_like(v_cond) * 1.7

    # Set up kernel
    if geometry == 'r2':
        space_kernel = TFPKernel(tfk.MaternThreeHalves, 2, 2)
    elif geometry == 's2':
        S2 = EmbeddedS2(1.0)
        space_kernel = ManifoldProjectionVectorKernel(
                MaternCompactRiemannianManifoldKernel(3/2, S2, 120), S2
            ) # 144 is the maximum number of basis functions we have implemented

    if spacetime:
        time_kernel = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
        kernel = ScaledKernel(ProductKernel(space_kernel, time_kernel)) # Space-time kernel
    else:
        kernel = ScaledKernel(space_kernel)

    gp = GaussianProcess(kernel)

    # Initialize parameters
    if spacetime:
        init_log_length_scale = {'space': -1.70, 'time': 1.3}
        init_log_amplitude = 2.2 if geometry=="r2" else 12.0
    else:
        with open(logdir+'/'+geometry+'_params_final.pickle', "rb") as f:
            params_dict = pickle.load(f)

        init_log_length_scale = params_dict['log_length_scale']
        init_log_amplitude = 2.2 if geometry=="r2" else 11.5

    kernel_params = refresh_kernel(next(rng), kernel, m_cond, geometry, init_log_length_scale, init_log_amplitude, spacetime=spacetime)

    # Initialize and condition GP
    gp_params, gp_state = gp.init_params_with_state(next(rng))
    gp_params = gp_params._replace(kernel_params=kernel_params)
    gp_state = gp.condition(gp_params, m_cond, v_cond, noises_cond)

    # Plot
    spacetime_ = "spacetime" if spacetime else "space"
    sphere_or_plane = "sphere" if plot_sphere else "plane"
    fname = "figs/"+geometry+"_gp_"+spacetime_+"_"+sphere_or_plane+".png"
    fname_vid = "figs/"+geometry+"_gp_"+spacetime_+"_"+sphere_or_plane+"_video.mp4"

    m_cond, v_cond, prior_mean = GetDataAlongSatelliteTrack(
                                    ds,
                                    aeolus,
                                    year,
                                    month,
                                    day,
                                    hour,
                                    num_hours=num_hours,
                                    return_mean=True,
                                    climatology=climatology,
                                    space_time=spacetime)

    posterior_mean = gp(gp_params, gp_state, m)[0].reshape(num_hours, lat_size*lon_size, 2)
    posterior_mean = posterior_mean + prior_mean
    m = m.reshape(num_hours, lat_size*lon_size, -1)
    m_cond = m_cond.reshape(num_hours, 60, -1)
    v_cond = v_cond.reshape(num_hours, 60, 2) 
    K = gp(gp_params, gp_state, m_ext)[1]

    # plot(fname, posterior_mean, K, m, m_cond, v_cond, num_hours, mesh, lon_size_ext, lat_size_ext, plot_sphere)
    animate(fname_vid, posterior_mean, K, m, m_cond, v_cond, num_hours, mesh, lon_size_ext, lat_size_ext)

    # Plot ground truth


if __name__ == "__main__":
    main()

