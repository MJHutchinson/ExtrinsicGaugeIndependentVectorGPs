import xarray as xr
import numpy as np
import tensorflow.keras as keras
import requests
from zipfile import ZipFile
from typing import Union
from skyfield.api import wgs84, load, EarthSatellite
from scipy.interpolate import interp2d
import os


def get_winddata(target_dir: str = "global_wind_dataset", resolution: int = 5):
    """
    This function downloads the global wind data from weatherbench
    https://github.com/pangeo-data/WeatherBench
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
    r_u = requests.get(url_u)
    print("fetching url for v component...")
    r_v = requests.get(url_v)

    zipfile_u = "10m_u_component_of_wind_{res}deg.zip"
    zipfile_v = "10m_v_component_of_wind_{res}deg.zip"

    print("saving contents of u component...")
    open(zipfile_u, 'wb').write(r_u.content)
    print("saving contents of v component...")
    open(zipfile_v, 'wb').write(r_v.content)
    
    os.mkdir(target_dir)

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


class WindDataGeneratorRandom(keras.utils.Sequence):
    def __init__(
        self,
        ds: xr.Dataset,
        num_obs: int,
        window_size: int,
        gap_size: int = 6,
        shuffle: bool = True,
        load: bool = True):
        """
        Data generator for global wind dataset
        Adapted from https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py
        Args:
            ds: Dataset containing all variables
            num_obs: Number of random locations where wind velocity is measured
            window_size: Number of consecutive observations in assimilation window
            gap_size: Gap between two observations (in hours)
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
        """
        self.ds = ds
        self.lon = ds.isel(time=0).lon.values
        self.lat = ds.isel(time=0).lat.values
        self.num_obs = num_obs
        self.window_size = window_size
        self.gap_size = gap_size
        self.shuffle = shuffle
        self.n_samples = ds.isel(time=slice(0, -window_size)).dims["time"]
        self.n_windows = int(self.n_samples - self.gap_size * self.window_size)

        self.on_epoch_end()

        # Load data into RAM (this speeds up the data loading process)
        if load: self.ds.load()

    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, i):
        start_idx = self.idxs[i]
        X, y = [], []
        for t in range(self.window_size):
            u, v = [], []
            self.lon_idxs = np.random.randint(0, self.ds.dims['lon'], size=(self.num_obs,))
            self.lat_idxs = np.random.randint(0, self.ds.dims['lat'], size=(self.num_obs,))
            for n in range(self.num_obs):
                u.append(self.ds.isel(
                    time=start_idx + self.gap_size * t,
                    lon=self.lon_idxs[n],
                    lat=self.lat_idxs[n]).u10.values)
                v.append(self.ds.isel(
                    time=start_idx + self.gap_size * t,
                    lon=self.lon_idxs[n],
                    lat=self.lat_idxs[n]).v10.values)
            X.append([self.lon[self.lon_idxs], self.lat[self.lat_idxs]])
            y.append([u, v])
        return start_idx, np.stack(X), np.stack(y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_windows)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


def GetDataAlongSatellite(ds, hours):
    """
        Generate wind data along the trajectories of Aeolus (satellite)
        More information about Aeolus: https://www.n2yo.com/satellite/?s=43600 
    """
    # Get Aeolus data from its TLE set
    ts = load.timescale()
    line1 = '1 43600U 18066A   21112.99668353  .00040037  00000-0  16023-3 0  9999'
    line2 = '2 43600  96.7174 120.6934 0007334 114.6816 245.5221 15.86410481154456'
    satellite = EarthSatellite(line1, line2, 'AEOLUS', ts)
    lon = ds.isel(time=0).lon.values
    lat = ds.isel(time=0).lat.values
    u_2018_01_01 = ds.u10.sel(time='2018-01-01').values
    v_2018_01_01 = ds.v10.sel(time='2018-01-01').values
    location, wind = [], []
    for hour in range(hours):
        time_span = ts.utc(2018, 1, 1, hour, range(0, 60))
        geocentric = satellite.at(time_span)
        subpoint = wgs84.subpoint(geocentric)
        lon_location = subpoint.longitude.degrees + 180
        lat_location = subpoint.latitude.degrees
        u_interp = interp2d(lon, lat, u_2018_01_01[hour], kind='linear')
        v_interp = interp2d(lon, lat, v_2018_01_01[hour], kind='linear')
        u, v = [], []
        for x, y in zip(lon_location, lat_location):
            u.append(u_interp(x, y).item())
            v.append(v_interp(x, y).item())
        location.append([lon_location, lat_location])
        wind.append([u, v])
    return np.stack(location), np.stack(wind)


class WindDataGeneratorSatellite(keras.utils.Sequence):
    def __init__(
        self,
        ds: xr.Dataset,
        num_obs: int,
        window_size: int,
        gap_size: int = 6,
        shuffle: bool = True,
        load: bool = True):
        """
        Generate wind data along the trajectories of Aeolus (satellite)
        More information about Aeolus: https://www.n2yo.com/satellite/?s=43600 
        Args:
            ds: Dataset containing all variables
            num_obs: Number of random locations where wind velocity is measured
            window_size: Number of consecutive observations in assimilation window
            gap_size: Gap between two observations (in hours)
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
        """
        self.ds = ds
        self.lon = ds.isel(time=0).lon.values
        self.lat = ds.isel(time=0).lat.values
        self.num_obs = num_obs
        self.window_size = window_size
        self.gap_size = gap_size
        self.shuffle = shuffle
        self.n_samples = ds.isel(time=slice(0, -window_size)).dims["time"]
        self.n_windows = int(self.n_samples - self.gap_size * self.window_size)

        self.on_epoch_end()

        # Load data into RAM (this speeds up the data loading process)
        if load: self.ds.load()

    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, i):
        start_idx = self.idxs[i]
        X, y = [], []
        for t in range(self.window_size):
            u, v = [], []
            self.lon_idxs = np.random.randint(0, self.ds.dims['lon'], size=(self.num_obs,))
            self.lat_idxs = np.random.randint(0, self.ds.dims['lat'], size=(self.num_obs,))
            for n in range(self.num_obs):
                u.append(self.ds.isel(
                    time=start_idx + self.gap_size * t,
                    lon=self.lon_idxs[n],
                    lat=self.lat_idxs[n]).u10.values)
                v.append(self.ds.isel(
                    time=start_idx + self.gap_size * t,
                    lon=self.lon_idxs[n],
                    lat=self.lat_idxs[n]).v10.values)
            X.append([self.lon[self.lon_idxs], self.lat[self.lat_idxs]])
            y.append([u, v])
        return start_idx, np.stack(X), np.stack(y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_windows)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)