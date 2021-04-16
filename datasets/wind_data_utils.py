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


class WindDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        ds: xr.Dataset,
        step_size: int,
        seq_len: int,
        lead_time: int,
        batch_size: int,
        shuffle = True,
        load = True,
        mean = None,
        std = None):
        """
        Adapted from https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py
        Args:
            ds: Dataset containing all variables
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.step_size = step_size
        self.seq_len = seq_len
        
        # Normalize
        data_array = ds.to_array('components')
        self.mean = data_array.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = data_array.std('time').mean(('lat', 'lon')).compute() if std is None else std
        self.data = (data_array - self.mean) / self.std

        if lead_time == 0:
            self.n_samples = self.data.shape[0]
        elif lead_time > 0 and isinstance(lead_time, int):
            self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        else:
            raise ValueError('Lead time must be a non-negative integer')

        self.on_epoch_end()

        # Load data into RAM (this speeds up the data loading process)
        if load: self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        start_idxs = idxs
        end_idxs = idxs + self.seq_len * self.step_size
        step = self.step_size
        X = []
        for start, end in zip(start_idxs, end_idxs):
            X.append(self.data.isel(time=slice(start, end, step)).values)
        X = np.stack(X) # Size (batch, seq_len, 2, lat, lon)
        y = self.data.isel(time=end_idxs + self.lead_time).values[:, None, :, :, :]
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
