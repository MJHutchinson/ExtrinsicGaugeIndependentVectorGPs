import xarray as xr
import numpy as np
from typing import List
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
import optax
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.kernel.scaled import ScaledKernel
from riemannianvectorgp.kernel.TFP import TFPKernel
from skyfield.api import wgs84, load, EarthSatellite
from scipy.interpolate import interp2d
import cartopy
import cartopy.crs as ccrs
import xesmf as xe
import copy

class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self
    
    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key


def _deg2rad(x: np.ndarray):
    return (np.pi/180)*x


def _rad2deg(x: np.ndarray):
    return (180/np.pi)*x


def _jump_where(x):
        for i in range(1,len(x)):
            if np.abs(x[i-1] - x[i]) > 180:
                return i


def GetDataAlongSatelliteTrack(
    ds: xr.Dataset,
    satellite: EarthSatellite,
    year: int,
    month: int,
    day: int,
    hour: int) -> List[jnp.ndarray]:
    """
        Generate wind data along the trajectories of Aeolus (satellite)
        More information about the Aeolus satellite: https://www.n2yo.com/satellite/?s=43600 
    """
    date = f"{year}-{month}-{day}"
    lon = _deg2rad(ds.isel(time=0).lon.values)
    lat = _deg2rad(ds.isel(time=0).lat.values)
    u = ds.u100.sel(time=date).values
    v = ds.v100.sel(time=date).values

    ts = load.timescale()
    time_span = ts.utc(year, month, day, hour, range(0, 60))
    geocentric = satellite.at(time_span)
    subpoint = wgs84.subpoint(geocentric)
    lon_location = subpoint.longitude.radians
    lat_location = subpoint.latitude.radians

    u_interp = interp2d(lon, lat, u[hour], kind='linear')
    v_interp = interp2d(lon, lat, v[hour], kind='linear')
    u_along_sat_track, v_along_sat_track = [], []
    for x, y in zip(lon_location, lat_location):
        u_along_sat_track.append(u_interp(x, y).item())
        v_along_sat_track.append(v_interp(x, y).item())

    location = jnp.stack([lat_location, lon_location])
    wind = jnp.stack([jnp.stack(u_along_sat_track), jnp.stack(v_along_sat_track)])

    return location.transpose(), wind.transpose()


if __name__ == '__main__':
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
    ds = xr.open_mfdataset('../datasets/era5_dataset/*.nc')
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+resolution/2, 90, resolution)),
            'lon': (['lon'], np.arange(-180+resolution/2, 180, resolution)),
        }
    )
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    regridder = xe.Regridder(ds, grid_out, 'bilinear', periodic=True)
    ds = regridder(ds)
    lon = ds.isel(time=0).lon
    lat = ds.isel(time=0).lat
    lon_size = lon.shape[0]
    lat_size = lat.shape[0]
    mesh = np.meshgrid(lon, lat)

    phi, theta = jnp.meshgrid(_deg2rad(lat), _deg2rad(lon))
    phi = phi.flatten()
    theta = theta.flatten()
    m = jnp.stack([phi, theta], axis=-1)
    
    # Get inputs and outputs
    m_cond, v_cond = GetDataAlongSatelliteTrack(ds, aeolus, year, month, day, hour)
    noises_cond = jnp.ones_like(v_cond) * 1.7

    # Set up kernel
    ev_kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 2, 2))
    ev_kernel_params = ev_kernel.init_params(next(rng))
    sub_kernel_params = ev_kernel_params.sub_kernel_params
    sub_kernel_params = sub_kernel_params._replace(log_length_scales=jnp.log(0.32))
    ev_kernel_params = ev_kernel_params._replace(sub_kernel_params=sub_kernel_params)
    # ev_kernel_params = ev_kernel_params._replace(
    #     log_amplitude=-jnp.log(ev_kernel.matrix(ev_kernel_params, m, m)[0, 0, 0, 0])
    # )
    ev_kernel_params = ev_kernel_params._replace(log_amplitude=-0.31)

    # Set up Euclidean Vector GP
    ev_gp = GaussianProcess(ev_kernel)
    ev_gp_params, ev_gp_state = ev_gp.init_params_with_state(next(rng))
    ev_gp_params = ev_gp_params._replace(kernel_params=ev_kernel_params)
    ev_gp_state = ev_gp.condition(ev_gp_params, m_cond, v_cond, noises_cond)

    # Plot predictions
    scale = 300
    mean, K = ev_gp(ev_gp_params, ev_gp_state, m)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()

    plt.quiver(
        _rad2deg(theta),
        _rad2deg(phi),
        mean[:, 0],
        mean[:, 1],
        color="blue",
        alpha=0.5,
        scale=scale,
        width=0.003,
        headwidth=3
    )

    plt.quiver(
        _rad2deg(m_cond[:, 1]),
        _rad2deg(m_cond[:, 0]),
        v_cond[:, 0],
        v_cond[:, 1],
        color="red",
        scale=scale,
        width=0.003,
        headwidth=3
    )

    # Plot satellite trajectories
    idx = _jump_where(_rad2deg(m_cond[:, 1]))
    x1 = copy.deepcopy(_rad2deg(m_cond[:idx+1, 1]))
    y1 = _rad2deg(m_cond[:idx+1, 0])
    x1[idx] = x1[idx]-360

    x2 = copy.deepcopy(_rad2deg(m_cond[idx-1:, 1]))
    y2 = _rad2deg(m_cond[idx-1:, 0])
    x2[0] = x2[0] + 360

    plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2)
    plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2)

    plt.title("posterior mean")
    plt.savefig("figs/wind_interpolation_gp_euclidean_mean.png")

    # Plot standard deviations
    var_norm = jnp.diag(jnp.trace(K, axis1=2, axis2=3)).reshape(lon_size, lat_size)
    std_norm = jnp.sqrt(var_norm).transpose()

    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.contourf(*mesh, std_norm, levels=30, zorder=1)

    plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2)
    plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2)

    plt.title("posterior std")
    plt.savefig("figs/wind_interpolation_gp_euclidean_std.png")

    # Plot ground truth
    u_gt = ds.u100.sel(time=date).values
    v_gt = ds.v100.sel(time=date).values
    
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)

    plt.quiver(*mesh, u_gt, v_gt,
               alpha=0.3,
               zorder=1,
               color='black',
               scale=350,
               width=0.003,
               headwidth=3)

    plt.quiver(_rad2deg(m_cond[:, 1]),
               _rad2deg(m_cond[:, 0]),
               v_cond[:, 0],
               v_cond[:, 1],
               zorder=2,
               color='red',
               scale=350,
               width=0.003,
               headwidth=3)

    plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2)
    plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2)

    plt.savefig("figs/ground_truth.png")


    
