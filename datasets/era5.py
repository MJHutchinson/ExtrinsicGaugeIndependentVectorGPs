import cdsapi
import xarray as xr
import xesmf as xe
import pandas as pd
from urllib.request import urlopen
import os

def get_era5(dataset_name='reanalysis-era5-single-levels', 
             var=None, 
             dates=None,
             times=None,
             pressure_level=None,
             grid=[1.0, 1,0],
             area=[90, 0, -90, 360],
             download_flag = False,
             download_file='./output.nc'
            ):
    ''' 
    Code adapted from https://towardsdatascience.com/read-era5-directly-into-memory-with-python-511a2740bba0

    Get ERA5 reanalysis output from the web
    this script grabs ERA5 variables from the web and stores them 
    in an xarray dataset. 
    
    The ERA5 CDS API must be installed on the local machine.
    See section 4 here: https://cds.climate.copernicus.eu/api-how-to
    
    Parameters
    ----------                  
    dataset_name: str, default 'reanalysis-era5-single-levels'
        name of dataset to use. Options include:
        * 'reanalysis-era5-single-levels'
        * 'reanalysis-era5-single-levels-monthly-means'
        * 'reanalysis-era5-pressure-levels'
        * 'reanalysis-era5-pressure-levels-monthly-means'
        * 'reanalysis-era5-land'
        * 'reanalysis-era5-land-monthly-means'
        
    dates: list of strings or datetime64, default None
        example ['1980-01-01', '2020-12-31']
    times: list of strings of form 'HH:MM', default None
        example ['00:00', '01:00']
    var: str, default None
        name of variable to download
        example '2m_temperature'
    pressure_level: str, default None
        pressure level to grab data on
    grid: list, deafult [1.0, 1.0]
        spatial lat, lon grid resolution in deg
    area: list, default [90,-180,-90, 180]
        area extent download [N, W, S, E]
    download_flag = True or False, default False
        flag to download data or not
    download_file= str, default './output.nc'
        path to where data should be downloaed to.
        data only downloaded if download_flag is True
    Returns
    -------
    ds: xarrayDataSet
        all the data will be in an xarray dataset
        
    Example
    -------
    ds = get_era5(dataset_name='reanalysis-era5-single-levels-monthly-means', 
                 var='2m_temperature', 
                 dates=['2021-02-01'],
                 grid=[0.25, 0.25])
        
    Notes
    -------    
    # cdsapi code is here
    https://github.com/ecmwf/cdsapi/tree/master/cdsapi
    # information on api is here
    https://confluence.ecmwf.int/display/CKB/Climate+Data+Store+%28CDS%29+API+Keywords
    # era5 dataset information is here
    https://confluence.ecmwf.int/display/CKB/The+family+of+ERA5+datasets
    '''
    
    # test if acceptable pressure level
    acceptable_pressures = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, range(100, 1000, 25)]
    if pressure_level not in [str(lev) for lev in acceptable_pressures]:
        print(f"!! Pressure level must be in this list: {acceptable_pressures}")
    
    # start the cdsapi client
    c = cdsapi.Client()

    # parameters
    params = dict(
        format = "netcdf",
        product_type = "reanalysis",
        variable = var,
        grid = grid,
        area = area,
        date = dates,
        time = times
        )   
    
    # what to do if asking for monthly means
    if dataset_name in ["reanalysis-era5-single-levels-monthly-means", 
                        "reanalysis-era5-pressure-levels-monthly-means",
                        "reanalysis-era5-land-monthly-means"]:
        params["product_type"] = "monthly_averaged_reanalysis"
        _ = params.pop("date")
        params["time"] = "00:00"
        
        # if time is in list of pandas format
        if isinstance(dates, list):
            dates_pd = pd.to_datetime(dates)
            params["year"] = sorted(list(set(dates_pd.strftime("%Y"))))
            params["month"] = sorted(list(set(dates_pd.strftime("%m"))))
        else:
            params["year"] = sorted(list(set(dates.strftime("%Y"))))
            params["month"] = sorted(list(set(dates.strftime("%m"))))

    # if pressure surface
    if dataset_name in ["reanalysis-era5-pressure-levels-monthly-means",
                        "reanalysis-era5-pressure-levels"]:
        params["pressure_level"] = pressure_level
    
    # product_type not needed for era5_land
    if dataset_name in ["reanalysis-era5-land"]:
        _ = params.pop("product_type")
        
    # file object
    fl = c.retrieve(dataset_name, params) 
    
    # download the file 
    if download_flag:
        fl.download(f"{download_file}")
    
    # load into memory and return xarray dataset
    with urlopen(fl.location) as f:
        return xr.open_dataset(f.read())


if __name__ == '__main__':
    """Download data for wind speed at 100m at given dates and times.
    To run this, you first need to create an account at Copernicus at:
    https://cds.climate.copernicus.eu/#!/home
    and save your UID and API key by running
    ```
    {
    echo 'url: https://cds.climate.copernicus.eu/api/v2'
    echo 'key: <UID>:<API key>'
    echo 'verify: 0'
    } > ~/.cdsapirc
    ```
    on terminal. More details in:
    https://towardsdatascience.com/read-era5-directly-into-memory-with-python-511a2740bba0
    """
    
    dates = ['2019-01-01']
    times = [
        '00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00',
        '09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00',
        '18:00','19:00','20:00','21:00','22:00','23:00'
    ]

    os.mkdir('era5_dataset') if not os.path.exists('era5_dataset') else None

    get_era5(dataset_name='reanalysis-era5-single-levels', 
             var='10m_u_component_of_wind', 
             dates=dates,
             times=times,
             download_flag=True,
             download_file='era5_dataset/10m_u_component_of_wind_2019_01_01.nc')

    get_era5(dataset_name='reanalysis-era5-single-levels', 
             var='10m_v_component_of_wind', 
             dates=dates,
             times=times,
             download_flag=True,
             download_file='era5_dataset/10m_v_component_of_wind_2019_01_01.nc')
    