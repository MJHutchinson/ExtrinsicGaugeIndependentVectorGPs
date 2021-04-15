import xarray as xr
import numpy as np
import tensorflow.keras as keras
import requests
from zipfile import ZipFile
import os


def get_winddata(target_dir: str = "global_wind_dataset", resolution: int = 5):
    """
    This function downloads the global wind data from weatherbench
    https://github.com/pangeo-data/WeatherBench
    Args:
        target_dir: target directory to save wind data
        resolution: resolution of wind data. To choose from 1, 2 or 5.
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


class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """

        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.lead_time).values
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

if __name__ == "__main__":
    get_winddata()