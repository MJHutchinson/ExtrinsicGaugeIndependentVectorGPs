import math
from functools import partial
from typing import List
import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
from jax import jit
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
import matplotlib.pyplot as plt
#from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
)
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


def _deg2rad(x: np.ndarray, offset: float=0.):
    return (np.pi/180)*x + offset


def GetDataAlongSatelliteTrack(
    ds: xr.Dataset,
    satellite: EarthSatellite,
    year: int,
    month: int,
    day: int,
    hour: int) -> List[np.ndarray]:
    """
        Generate wind data along the trajectories of Aeolus (satellite)
        More information about the Aeolus satellite: https://www.n2yo.com/satellite/?s=43600 
    """
    date = f"{year}-{month}-{day}"
    lon = _deg2rad(ds.isel(time=0).lon.values) + jnp.pi
    lat = _deg2rad(ds.isel(time=0).lat.values) + jnp.pi/2
    u = ds.u10.sel(time=date).values
    v = ds.v10.sel(time=date).values

    time_span = ts.utc(year, month, day, hour, range(0, 60))
    geocentric = satellite.at(time_span)
    subpoint = wgs84.subpoint(geocentric)
    lon_location = subpoint.longitude.radians + jnp.pi
    lat_location = subpoint.latitude.radians + jnp.pi/2

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

    year = 2018
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

    ds = xr.open_mfdataset('../datasets/global_wind_dataset/*.nc')
    lon = _deg2rad(ds.isel(time=0).lon.values)
    lat = _deg2rad(ds.isel(time=0).lat.values) + jnp.pi/2
    lon_size = lon.shape[0]
    lat_size = lat.shape[0]
    mesh = np.meshgrid(lon, lat)

    phi, theta = jnp.meshgrid(lat, lon)
    phi = phi.flatten()
    theta = theta.flatten()
    m = jnp.stack([phi, theta], axis=-1)
    
    # Get inputs and outputs
    m_cond, v_cond = GetDataAlongSatelliteTrack(ds, aeolus, year, month, day, hour)
    noises_cond = jnp.ones_like(v_cond) * 1.7

    # Set up manifold vector kernel
    S2 = EmbeddedS2(1.0)
    mv_kernel = ScaledKernel(
        ManifoldProjectionVectorKernel(
            MaternCompactRiemannianManifoldKernel(1.5, S2, 144), S2
        )
    )  # 144 is the maximum number of basis functions we have implemented
    mv_kernel_params = mv_kernel.init_params(next(rng))
    sub_kernel_params = mv_kernel_params.sub_kernel_params
    sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.2))
    mv_kernel_params = mv_kernel_params._replace(sub_kernel_params=sub_kernel_params)
    mv_kernel_params = mv_kernel_params._replace(
        log_amplitude=-jnp.log(mv_kernel.matrix(mv_kernel_params, m, m)[0, 0, 0, 0])
    )

    # Set up manifold vector GP
    mv_gp = GaussianProcess(mv_kernel)
    mv_gp_params, mv_gp_state = mv_gp.init_params_with_state(next(rng))
    mv_gp_params = mv_gp_params._replace(kernel_params=mv_kernel_params)
    mv_gp_state = mv_gp.condition(mv_gp_params, m_cond, v_cond, noises_cond)

    # Plot predictions
    scale = 300
    fig = plt.figure(figsize=(10, 5))
    mean, K = mv_gp(mv_gp_params, mv_gp_state, m)

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
    lsm = constants.lsm.values # land-sea mask

    plt.contour(*mesh, lsm, zorder=1)

    plt.title("posterior mean")
    plt.savefig("figs/wind_interpolation_gp_manifold_mean.png")

    # Plot standard deviations
    var_norm = jnp.diag(jnp.trace(K, axis1=2, axis2=3)).reshape(lon_size, lat_size)
    std_norm = jnp.sqrt(var_norm).transpose()

    fig = plt.figure(figsize=(10,5))
    plt.contourf(*mesh, std_norm, levels=30, zorder=1)
    plt.contour(*mesh, lsm, zorder=2)

    plt.title("posterior std")
    plt.savefig("figs/wind_interpolation_gp_manifold_std.png")