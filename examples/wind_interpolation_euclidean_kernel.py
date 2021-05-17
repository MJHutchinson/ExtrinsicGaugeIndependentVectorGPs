import xarray as xr
import numpy as np
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


def GetDataAlongSatelliteTrack(
    ds: xr.Dataset,
    satellite: EarthSatellite,
    year: int = 2021,
    month: int = 1,
    day: int = 1,
    hour: int = 0):
    """
        Generate wind data along the trajectories of Aeolus (satellite)
        More information about the Aeolus satellite: https://www.n2yo.com/satellite/?s=43600 
    """
    date = f'{year}-{month}-{day}'
    lon = _deg2rad(ds.isel(time=0).longitude.values)
    lat = _deg2rad(ds.isel(time=0).latitude.values)
    u = ds.u100.sel(time=date).values
    v = ds.v100.sel(time=date).values

    time_span = ts.utc(year, month, day, hour, range(0, 60))
    geocentric = satellite.at(time_span)
    subpoint = wgs84.subpoint(geocentric)
    lon_location = subpoint.longitude.radians + np.pi
    lat_location = subpoint.latitude.radians
    u_interp = interp2d(lon, lat, u[hour], kind='linear')
    v_interp = interp2d(lon, lat, v[hour], kind='linear')
    u_along_sat_track, v_along_sat_track = [], []
    for x, y in zip(lon_location, lat_location):
        u_along_sat_track.append(u_interp(x, y).item())
        v_along_sat_track.append(v_interp(x, y).item())
    location = [lat_location, lon_location]
    wind = [u_along_sat_track, v_along_sat_track]

    return np.stack(location).transpose(), np.stack(wind).transpose()


if __name__ == '__main__':
    # Get Aeolus trajectory data from TLE set
    ts = load.timescale()
    line1 = '1 43600U 18066A   21112.99668353  .00040037  00000-0  16023-3 0  9999'
    line2 = '2 43600  96.7174 120.6934 0007334 114.6816 245.5221 15.86410481154456'
    aeolus = EarthSatellite(line1, line2, 'AEOLUS', ts)

    rng = GlobalRNG()
    ds = xr.open_mfdataset('../datasets/era5_dataset/*.nc')
    lon = _deg2rad(ds.isel(time=0).longitude.values)
    lat = _deg2rad(ds.isel(time=0).latitude.values)
    mesh = np.meshgrid(lon, lat)

    phi, theta = jnp.meshgrid(lat, lon)
    phi = phi.flatten()[::20]
    theta = theta.flatten()[::20]
    lat_size = phi.shape[0]
    lon_size = theta.shape[0]
    m = jnp.stack([phi, theta], axis=-1)
    
    # Get inputs and outputs
    m_cond, v_cond = GetDataAlongSatelliteTrack(ds, aeolus)
    m_cond, v_cond = jnp.asarray(m_cond), jnp.asarray(v_cond)
    noises_cond = jnp.ones_like(v_cond) * 0.01

    # Set up kernel
    ev_kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 2, 2))
    ev_kernel_params = ev_kernel.init_params(next(rng))
    sub_kernel_params = ev_kernel_params.sub_kernel_params
    sub_kernel_params = sub_kernel_params._replace(log_length_scales=jnp.log(0.5))
    ev_kernel_params = ev_kernel_params._replace(sub_kernel_params=sub_kernel_params)
    ev_kernel_params = ev_kernel_params._replace(
        log_amplitude=-jnp.log(ev_kernel.matrix(ev_kernel_params, m, m)[0, 0, 0, 0])
    )

    # Set up Euclidean Vector GP
    ev_gp = GaussianProcess(ev_kernel)
    ev_gp_params, ev_gp_state = ev_gp.init_params_with_state(next(rng))
    ev_gp_params = ev_gp_params._replace(kernel_params=ev_kernel_params)
    ev_gp_state = ev_gp.condition(ev_gp_params, m_cond, v_cond, noises_cond)

    # Plot predictions
    scale = 300
    mean, K = ev_gp(ev_gp_params, ev_gp_state, m)

    fig = plt.figure(figsize=(10, 5))

    plt.quiver(
        theta,
        phi,
        mean[:, 0],
        mean[:, 1],
        color="blue",
        alpha=0.5,
        scale=scale,
        width=0.003,
        headwidth=3,
        zorder=4
    )

    plt.quiver(
        m_cond[:, 1],
        m_cond[:, 0],
        v_cond[:, 0],
        v_cond[:, 1],
        color="red",
        scale=scale,
        width=0.003,
        headwidth=3,
        zorder=5,
    )

    constants = xr.open_dataset("../datasets/constants/constants.nc")
    lsm = constants.lsm.values

    plt.contour(*mesh, lsm, zorder=1)

    plt.title("posterior mean")
    plt.savefig("figs/wind_interpolation_gp_euclidean_mean.png")

    # Plot standard deviations
    var_norm = jnp.diag(jnp.trace(K, axis1=2, axis2=3)).reshape(lon_size, lat_size)
    std_norm = jnp.sqrt(var_norm).transpose()

    fig = plt.figure(figsize=(10,5))
    plt.contourf(*mesh, std_norm, levels=30, zorder=1)
    plt.contour(*mesh, lsm, zorder=2)

    plt.title("posterior std")
    plt.savefig("figs/wind_interpolation_gp_euclidean_std.png")

    # Plot ground truth
    u_gt = ds.u100.sel(time='2021-01-01').values[0]
    v_gt = ds.v100.sel(time='2021-01-01').values[0]
    plt.figure(figsize=(10, 5))
    plt.contour(*mesh, lsm, zorder=1)

    plt.quiver(*mesh, u_gt, v_gt,
               alpha=0.5,
               zorder=2,
               color='black',
               scale=350,
               width=0.003,
               headwidth=3)

    plt.quiver(m_cond[:, 1],
               m_cond[:, 0],
               v_cond[:, 0],
               v_cond[:, 1],
               color='red',
               scale=350,
               width=0.003,
               headwidth=3)

    plt.savefig("figs/ground_truth.png")


    
