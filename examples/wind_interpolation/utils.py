import numpy as np
import xarray as xr
import jax.numpy as jnp
from typing import List
from skyfield.api import wgs84, load, EarthSatellite
from scipy.interpolate import interp2d

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


def GetDataAlongSatelliteTrack(
    ds: xr.Dataset,
    satellite: EarthSatellite,
    climatology: np.ndarray,
    year: int,
    month: int,
    day: int,
    hour: int) -> List[jnp.ndarray]:
    """
        Generate wind anomaly data along the trajectories of Aeolus (satellite)
        More information about the Aeolus satellite: https://www.n2yo.com/satellite/?s=43600 
    """
    date = f"{year}-{month}-{day}"
    lon = deg2rad(ds.isel(time=0).lon.values) # Range: [0, 2*pi]
    lat = deg2rad(ds.isel(time=0).lat.values, offset=jnp.pi/2) # Range: [0, pi]
    u = ds.u10.sel(time=date)
    v = ds.v10.sel(time=date)

    # Compute difference from weekly climatology
    week_number = u['time'].dt.isocalendar().week[0]
    u_mean = climatology['u'][week_number-1]
    v_mean = climatology['v'][week_number-1]
    u_anomaly = u[hour].values - u_mean
    v_anomaly = v[hour].values - v_mean

    ts = load.timescale()
    time_span = ts.utc(year, month, day, hour, range(0, 60))
    geocentric = satellite.at(time_span)
    subpoint = wgs84.subpoint(geocentric)

    lon_location = subpoint.longitude.radians # Range: [-pi, pi]
    lon_location = np.where(lon_location > 0, lon_location, 2*np.pi + lon_location) # Range: [0, 2pi]
    lat_location = subpoint.latitude.radians + jnp.pi/2 # Range: [0, pi]

    u_interp = interp2d(lon, lat, u_anomaly, kind='linear')
    v_interp = interp2d(lon, lat, v_anomaly, kind='linear')
    u_along_sat_track, v_along_sat_track = [], []
    for x, y in zip(lon_location, lat_location):
        u_along_sat_track.append(u_interp(x, y).item())
        v_along_sat_track.append(v_interp(x, y).item())

    location = jnp.stack([lat_location, lon_location])
    wind = jnp.stack([jnp.stack(v_along_sat_track), jnp.stack(u_along_sat_track)])

    return location.transpose(), wind.transpose()

