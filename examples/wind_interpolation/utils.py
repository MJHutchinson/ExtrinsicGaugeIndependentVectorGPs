import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from typing import List, Union
from skyfield.api import wgs84, load, EarthSatellite
from riemannianvectorgp.sparse_gp import SparseGaussianProcess, SparseGaussianProcessParameters, SparseGaussianProcessState
from riemannianvectorgp.kernel import AbstractKernel
from examples.wind_interpolation import climatology
from scipy.interpolate import interp2d
from dataclasses import dataclass
from functools import partial

def deg2rad(x: np.ndarray, offset: float=0.):
    return (np.pi/180)*x + offset


def rad2deg(x: np.ndarray, offset: float=0.):
    return (180/np.pi)*(x - offset)


def refresh_kernel(key, kernel, m, geometry, init_log_length_scale, init_log_amplitude=None):
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


def GetDataAlongSatelliteTrack(
    ds: xr.Dataset,
    satellite: EarthSatellite,
    year: int,
    month: int,
    day: int,
    hour: int,
    num_hours: int=1,
    anomaly: bool=False,
    return_mean: bool=False,
    climatology: Union[np.ndarray, None]=None,
    space_time: bool=False) -> List[jnp.ndarray]:
    """Generate wind data along the trajectories of Aeolus (satellite)
        More information about the Aeolus satellite: https://www.n2yo.com/satellite/?s=43600
        https://www.ecmwf.int/sites/default/files/elibrary/2016/16851-esa-adm-aeolus-doppler-wind-lidar-mission-status-and-validation-strategy.pdf
    """
    date = f"{year}-{month}-{day}"
    lon = deg2rad(ds.isel(time=0).lon.values) # Range: [0, 2*pi]
    lat = deg2rad(ds.isel(time=0).lat.values, offset=jnp.pi/2) # Range: [0, pi]
    u = ds.u10.sel(time=date)
    v = ds.v10.sel(time=date)
    week_numbers = u['time'].dt.isocalendar().week
    u, v = u.values, v.values # Convert to numpy array

    def compute_mean(climatology, t):
        if climatology is None:
            climatology.weekly_climatology()
            climatology = np.load("../../datasets/climatology/weekly_climatology.npz")
        # Compute difference from weekly climatology
        u_mean = climatology['u'][week_numbers[t]-1]
        v_mean = climatology['v'][week_numbers[t]-1]
        return u_mean, v_mean

    ts = load.timescale()
    location, wind = [], []
    for t in range(num_hours):
        if anomaly:
            u_mean, v_mean = compute_mean(climatology, t)
            u_ = u[hour+t] - u_mean
            v_ = v[hour+t] - v_mean
        else:
            u_, v_ = u[hour+t], v[hour+t]

        time_span = ts.utc(year, month, day, hour+t, range(0, 60))
        geocentric = satellite.at(time_span)
        subpoint = wgs84.subpoint(geocentric)

        lon_location = subpoint.longitude.radians # Range: [-pi, pi]
        lon_location = np.where(lon_location > 0, lon_location, 2*np.pi + lon_location) # Range: [0, 2pi]
        lat_location = subpoint.latitude.radians + jnp.pi/2 # Range: [0, pi]

        u_interp = interp2d(lon, lat, u_, kind='linear')
        v_interp = interp2d(lon, lat, v_, kind='linear')
        u_along_sat_track, v_along_sat_track = [], []
        for x, y in zip(lon_location, lat_location):
            u_along_sat_track.append(u_interp(x, y).item())
            v_along_sat_track.append(v_interp(x, y).item())

        if t == 0:
            location = jnp.stack([lat_location, lon_location], axis=1)
            wind = jnp.stack([jnp.stack(v_along_sat_track), jnp.stack(u_along_sat_track)], axis=1)
            time = jnp.zeros((60,1))
        else:
            location = jnp.concatenate([location, jnp.stack([lat_location, lon_location], axis=1)])
            wind = jnp.concatenate([wind, jnp.stack([jnp.stack(v_along_sat_track), jnp.stack(u_along_sat_track)], axis=1)])
            time = jnp.concatenate([time, t*jnp.ones((60,1))])
    
    if space_time:
        location = jnp.concatenate([location, time], axis=1)

    if return_mean:
        mean_list = []
        for t in range(num_hours):
            u_mean, v_mean = compute_mean(climatology, t)
            u_mean = u_mean.transpose().flatten()
            v_mean = v_mean.transpose().flatten()
            mean_list.append(jnp.stack([v_mean, u_mean], axis=-1))
        mean = jnp.concatenate(mean_list)
        return location, wind, mean
    
    else:
        return location, wind

