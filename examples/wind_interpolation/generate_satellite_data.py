# %%
import xarray as xr
import numpy as np
import jax.numpy as jnp
from examples.wind_interpolation.utils import (
    deg2rad,
    rad2deg,
    GetDataAlongSatelliteTrack,
)
from skyfield.api import load, EarthSatellite
import xesmf as xe
import os

# %%
# Set the date of the ERA5 wind analysis data
year = 2019
month = 1
day = 1
hour = 9
num_hours = 1

# %%
# Get Aeolus trajectory data from TLE set
ts = load.timescale()
line1 = "1 43600U 18066A   21112.99668353  .00040037  00000-0  16023-3 0  9999"
line2 = "2 43600  96.7174 120.6934 0007334 114.6816 245.5221 15.86410481154456"
aeolus = EarthSatellite(line1, line2, "AEOLUS", ts)

# %%
# Load ERA5 data and regrid to desired spatial resolution
resolution = 5.625
ds = xr.open_mfdataset("../../datasets/era5_dataset/*.nc")
grid_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-90 + resolution / 2, 90, resolution)),
        "lon": (["lon"], np.arange(0, 360, resolution)),
    }
)
ds = ds.rename({"latitude": "lat", "longitude": "lon"})
regridder = xe.Regridder(ds, grid_out, "bilinear", periodic=True)
ds = regridder(ds)

# Load weekly climatology if it is already computed and saved
if os.path.exists("../../datasets/climatology/weekly_climatology.npz"):
    climatology = np.load("../../datasets/climatology/weekly_climatology.npz")
else:
    climatology = None

# %%
# Get input locations
lon = ds.isel(time=0).lon
lat = ds.isel(time=0).lat
lon_size = lon.shape[0]
lat_size = lat.shape[0]

# Reparametrise as lat=(0, pi) and lon=(0, 2pi)
lat = deg2rad(lat, offset=jnp.pi / 2)
lon = deg2rad(lon)

# Reshape to get an array of size (lat*lon, 2)
lat, lon = jnp.meshgrid(lat, lon)
lat = lat.flatten()
lon = lon.flatten()
m = jnp.stack([lat, lon], axis=-1)

# %%
# Get conditioning points and values
m_cond, v_cond = GetDataAlongSatelliteTrack(
    ds,
    aeolus,
    year,
    month,
    day,
    hour,
    num_hours=num_hours,
    anomaly=True,
    return_mean=False,
    climatology=climatology,
    space_time=False,
)

# %%
# Save data
np.save("log/m_cond.npy", np.asarray(m_cond))
np.save("log/v_cond.npy", np.asarray(v_cond))
np.save("log/m.npy", np.asarray(m))
