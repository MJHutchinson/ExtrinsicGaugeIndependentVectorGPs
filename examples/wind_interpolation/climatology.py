"""
Script to compute the total/monthly/weekly mean from past reanalysis data (1979-2018)
"""
import numpy as np
import xarray as xr

def climatology(ds):
    u_mean = ds.u10.mean('time')
    v_mean = ds.v10.mean('time')
    print("computing climatology...")
    np.savez("../../datasets/climatology/climatology.npz", u=u_mean.values, v=v_mean.values)


def monthly_climatology(ds):
    ds['month'] = ds['time.month']
    monthly_averages = ds.groupby('month').mean('time')
    u_monthly_mean = monthly_averages.u10
    v_monthly_mean = monthly_averages.v10
    print("computing monthly climatology...")
    np.savez("../../datasets/climatology/monthly_climatology.npz", u=u_monthly_mean.values, v=v_monthly_mean.values)


def weekly_climatology(ds):
    ds['week'] = ds['time'].dt.isocalendar().week
    weekly_averages = ds.groupby('week').mean('time')
    u_weekly_mean = weekly_averages.u10
    v_weekly_mean = weekly_averages.v10
    print("computing weekly climatology...")
    np.savez("../../datasets/climatology/weekly_climatology.npz", u=u_weekly_mean.values, v=v_weekly_mean.values)


if __name__ == "__main__":
    # Load reanalysis data
    ds = xr.open_mfdataset('../../datasets/weatherbench_wind_data/*.nc')

    climatology(ds)
    monthly_climatology(ds)
    weekly_climatology(ds)
