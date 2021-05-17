import math
from functools import partial
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
    TFPKernel,
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
)
from skyfield.api import wgs84, load, EarthSatellite
from scipy.interpolate import interp2d
import sys
import polyscope as ps

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


def GetDataAlongSatelliteTrack(ds: xr.Dataset, satellite: EarthSatellite):
    """
        Generate wind data along the trajectories of Aeolus (satellite)
        More information about the Aeolus satellite: https://www.n2yo.com/satellite/?s=43600 
    """
    lon = _deg2rad(ds.isel(time=0).lon.values)
    lat = _deg2rad(ds.isel(time=0).lat.values)
    u_2018_01_01 = ds.u10.sel(time='2018-01-01').values
    v_2018_01_01 = ds.v10.sel(time='2018-01-01').values

    time_span = ts.utc(2018, 1, 1, 0, range(0, 60))
    geocentric = satellite.at(time_span)
    subpoint = wgs84.subpoint(geocentric)
    lon_location = subpoint.longitude.radians + np.pi
    lat_location = subpoint.latitude.radians
    u_interp = interp2d(lon, lat, u_2018_01_01[0], kind='linear')
    v_interp = interp2d(lon, lat, v_2018_01_01[0], kind='linear')
    u, v = [], []
    for x, y in zip(lon_location, lat_location):
        u.append(u_interp(x, y).item())
        v.append(v_interp(x, y).item())
    location = [lat_location, lon_location]
    wind = [u, v]

    return np.stack(location).transpose(), np.stack(wind).transpose()


if __name__ == '__main__':
    # Get Aeolus trajectory data from TLE set
    ts = load.timescale()
    line1 = '1 43600U 18066A   21112.99668353  .00040037  00000-0  16023-3 0  9999'
    line2 = '2 43600  96.7174 120.6934 0007334 114.6816 245.5221 15.86410481154456'
    aeolus = EarthSatellite(line1, line2, 'AEOLUS', ts)

    rng = GlobalRNG()
    ds = xr.open_mfdataset('../datasets/global_wind_dataset/*.nc')
    lon = _deg2rad(ds.isel(time=0).lon.values)
    lat = _deg2rad(ds.isel(time=0).lat.values)
    lon_size = lon.shape[0]
    lat_size = lat.shape[0]
    mesh = np.meshgrid(lon, lat)

    phi, theta = jnp.meshgrid(lat, lon)
    phi = phi.flatten()
    theta = theta.flatten()
    m = jnp.stack([phi, theta], axis=-1)
    
    # Get inputs and outputs
    m_cond, v_cond = GetDataAlongSatelliteTrack(ds, aeolus)
    m_cond, v_cond = jnp.asarray(m_cond), jnp.asarray(v_cond)
    noises_cond = jnp.ones_like(v_cond) * 0.01

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
    mv_k = mv_kernel.matrix(mv_kernel_params, m, m)

    i = 1000
    vec = jnp.array([0, 1])
    operator = mv_k[:, i] @ vec
    plt.quiver(m[:, 1], m[:, 0], operator[:, 0], operator[:, 1], color="blue")
    plt.quiver(m[i, 1], m[i, 0], vec[0], vec[1], color="red")
    plt.gca().set_aspect("equal")
    plt.title("Vector S2 Matern 3/2 kernel")

    plt.savefig("figs/vector_kernel_spherical.png")

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
