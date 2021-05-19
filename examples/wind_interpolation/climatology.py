"""
Script to compute the total/monthly/weekly mean from past reanalysis data (1979-2018)
"""
import numpy as np
import xarray as xr

# Load reanalysis data
ds = xr.open_mfdataset('../../datasets/weatherbench_wind_data/*.nc')

# Compute climatology
u_mean = ds.u10.mean('time')
v_mean = ds.v10.mean('time')
np.savez("../../datasets/climatology/climatology.npz", u=u_mean.values, v=v_mean.values)

# Monthly climatology
ds['month'] = ds['time.month']
monthly_averages = ds.groupby('month').mean('time')
u_monthly_mean = monthly_averages.u10
v_monthly_mean = monthly_averages.v10
np.savez("../../datasets/climatology/monthly_climatology.npz", u=u_monthly_mean.values, v=v_monthly_mean.values)

# Weekly climatology
ds['week'] = ds['time'].dt.isocalendar().week
weekly_averages = ds.groupby('week').mean('time')
u_weekly_mean = weekly_averages.u10
v_weekly_mean = weekly_averages.v10
np.savez("../../datasets/climatology/weekly_climatology.npz", u=u_weekly_mean.values, v=v_weekly_mean.values)
