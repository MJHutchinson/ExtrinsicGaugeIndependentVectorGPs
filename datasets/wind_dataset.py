import xarray as xr
import numpy as np
import tensorflow.keras as keras
import requests
from zipfile import ZipFile
from typing import Union
from skyfield.api import wgs84, load, EarthSatellite
from scipy.interpolate import interp2d
import os
import click

@click.command()
@click.option("--target-dir", default = "global_wind_dataset", type=str, help="directory to save data")
@click.option("--resolution", default = 5, type=click.Choice([1, 2, 5]), help="resolution of wind data (choose from 1, 2 or 5)")
def get_winddata(target_dir, resolution):
    """
    This function downloads the regridded ERA5 global wind data
    Args:
        target_dir: target directory to save wind data
        resolution: resolution of wind data. To choose between 1, 2 or 5.
    """
    if resolution == 1:
        res = 1.40625
    elif resolution == 2:
        res = 2.8125
    elif resolution == 5:
        res = 5.625
    else:
        raise ValueError("Resolution must be either 1, 2 or 5")

    url_u = f"https://dataserv.ub.tum.de/s/m1524895/download?path=%2F{res}deg%2F10m_u_component_of_wind&files=10m_u_component_of_wind_{res}deg.zip"
    url_v = f"https://dataserv.ub.tum.de/s/m1524895/download?path=%2F{res}deg%2F10m_v_component_of_wind&files=10m_v_component_of_wind_{res}deg.zip"

    # Download u and v components of global wind
    print("fetching url for u component...")
    r_u = requests.get(url_u, verify=False)
    print("fetching url for v component...")
    r_v = requests.get(url_v, verify=False)

    zipfile_u = f"10m_u_component_of_wind_{res}deg.zip"
    zipfile_v = f"10m_v_component_of_wind_{res}deg.zip"

    print("saving contents of u component...")
    open(zipfile_u, 'wb').write(r_u.content)
    print("saving contents of v component...")
    open(zipfile_v, 'wb').write(r_v.content)
    
    os.mkdir(target_dir) if not os.path.exists(target_dir) else None

    # Unzip files to target directory
    print("unzipping file for u component...")
    with ZipFile(zipfile_u, 'r') as zipObj:
        zipObj.extractall(target_dir)
    print("unzipping file for v component...")
    with ZipFile(zipfile_v, 'r') as zipObj:
        zipObj.extractall(target_dir)
    
    # Delete zip files
    print("delete zip files...")
    os.remove(zipfile_u)
    os.remove(zipfile_v)

    print("complete! data saved on: " + target_dir)


def GetDataAlongSatelliteTrack(
    ds: xr.Dataset,
    satellite: EarthSatellite,
    year: int = 2018,
    month: int = 1,
    day: int = 1,
    hour: int = 0):
    """
        Generate wind data along the trajectories of Aeolus (satellite)
        More information about the Aeolus satellite: https://www.n2yo.com/satellite/?s=43600 
    """
    date = f'{year}-{month}-{day}'
    lon = _deg2rad(ds.isel(time=0).lon.values)
    lat = _deg2rad(ds.isel(time=0).lat.values)
    u = ds.u10.sel(time=date).values
    v = ds.v10.sel(time=date).values

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
    get_winddata()
