import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from typing import List, Union
from skyfield.api import wgs84, load, EarthSatellite
from riemannianvectorgp.sparse_gp import (
    SparseGaussianProcess,
    SparseGaussianProcessParameters,
    SparseGaussianProcessState,
)
from riemannianvectorgp.kernel import AbstractKernel
from examples.wind_interpolation import climatology
from scipy.interpolate import interp2d
from dataclasses import dataclass
from functools import partial


def deg2rad(x: np.ndarray, offset: float = 0.0):
    return (np.pi / 180) * x + offset


def rad2deg(x: np.ndarray, offset: float = 0.0):
    return (180 / np.pi) * (x - offset)


def refresh_kernel(
    key, kernel, m, init_log_length_scale, init_log_amplitude=None
):
    kernel_params = kernel.init_params(key)
    sub_kernel_params = kernel_params.sub_kernel_params
    sub_kernel_params = sub_kernel_params._replace(
        log_length_scale=init_log_length_scale
    )
    kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
    log_amplitude = (
        -jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0])
        if init_log_amplitude is None
        else init_log_amplitude
    )
    kernel_params = kernel_params._replace(log_amplitude=log_amplitude)
    return kernel_params


def GetDataAlongSatelliteTrack(
    ds: xr.Dataset,
    satellite: EarthSatellite,
    year: int,
    month: int,
    day: int,
    hour: int,
    num_hours: int = 1,
    anomaly: bool = False,
    return_mean: bool = False,
    climatology: Union[np.ndarray, None] = None,
    space_time: bool = False,
) -> List[jnp.ndarray]:
    """Generate wind data along the trajectories of a given satellite
    
        Args:
            ds: an xarray dataset containing the global wind velocity data
            satellite: the satellite track used for observation
            year: year of observation
            month: month of observation
            day: day of observation
            hour: initial hour of observation (e.g. 8 if observation starts at 8am)
            num_hours: observation duration (in hours)
            anomaly: option to output the wind speed anomaly around the climatology
            return_mean: option to output the climatology together with the observation data
            climatology: an optional numpy array containing the climatology data. If None, it generates the climatology automatically.
            space_time: option for 3D spacetime output (lat, lon, time)
            
        Output:
            location: an array of size (num_obs, 2) or (num_obs, 3) (the latter if the space_time flag is set to True)
                      containing the (lat, lon) or (lat, lon, time) information of the satellite track.
            wind: an array of size (num_obs, 2) containing the (vertical, horizontal) wind speed data 
                  along the satellite track.
            mean: the climatology data. Only available when the return_mean flag is set to True.
    """
    date = f"{year}-{month}-{day}"
    lon = deg2rad(ds.isel(time=0).lon.values)  # Range: [0, 2*pi]
    lat = deg2rad(ds.isel(time=0).lat.values, offset=jnp.pi / 2)  # Range: [0, pi]
    u = ds.u10.sel(time=date)
    v = ds.v10.sel(time=date)
    week_numbers = u["time"].dt.isocalendar().week
    u, v = u.values, v.values  # Convert to numpy array

    def compute_mean(climatology, t):
        if climatology is None:
            climatology.weekly_climatology()
            climatology = np.load("../../datasets/climatology/weekly_climatology.npz")
        # Compute difference from weekly climatology
        u_mean = climatology["u"][week_numbers[t] - 1]
        v_mean = climatology["v"][week_numbers[t] - 1]
        return u_mean, v_mean

    ts = load.timescale()
    location, wind = [], []
    for t in range(num_hours):
        if anomaly:
            u_mean, v_mean = compute_mean(climatology, t)
            u_ = u[hour + t] - u_mean
            v_ = v[hour + t] - v_mean
        else:
            u_, v_ = u[hour + t], v[hour + t]

        time_span = ts.utc(year, month, day, hour + t, range(0, 60))
        geocentric = satellite.at(time_span)
        subpoint = wgs84.subpoint(geocentric)

        lon_location = subpoint.longitude.radians  # Range: [-pi, pi]
        lon_location = np.where(
            lon_location > 0, lon_location, 2 * np.pi + lon_location
        )  # Range: [0, 2pi]
        lat_location = subpoint.latitude.radians + jnp.pi / 2  # Range: [0, pi]

        u_interp = interp2d(lon, lat, u_, kind="linear")
        v_interp = interp2d(lon, lat, v_, kind="linear")
        u_along_sat_track, v_along_sat_track = [], []
        for x, y in zip(lon_location, lat_location):
            u_along_sat_track.append(u_interp(x, y).item())
            v_along_sat_track.append(v_interp(x, y).item())

        if t == 0:
            location = jnp.stack([lat_location, lon_location], axis=1)
            wind = jnp.stack(
                [jnp.stack(v_along_sat_track), jnp.stack(u_along_sat_track)], axis=1
            )
            time = jnp.zeros((60, 1))
        else:
            location = jnp.concatenate(
                [location, jnp.stack([lat_location, lon_location], axis=1)]
            )
            wind = jnp.concatenate(
                [
                    wind,
                    jnp.stack(
                        [jnp.stack(v_along_sat_track), jnp.stack(u_along_sat_track)],
                        axis=1,
                    ),
                ]
            )
            time = jnp.concatenate([time, t * jnp.ones((60, 1))])

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
