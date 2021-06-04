# Instructions

## 1. Installing dependencies
To run the experiments, you need to first install the following packages
- netcdf4
```
pip install netcdf4
```
- xarray
```
pip install xarray
```
- skyfield
```
pip install skyfield
```
- dask
```
pip install dask
```
- xesmf
```
conda install -c conda-forge xesmf
```
- cartopy
```
conda install -c conda-forge cartopy
```
- cdsapi
```
pip install cdsapi
```

## 2. Downloading the data
The code to download the data are saved in `../../datasets`. Go there and run the following scripts
- Run `era5.py` to download the ERA5 wind reanalysis data (this may take up to an hour depending on the queue)
- Run `wind_dataset.py` to download the weatherbench historical wind data

## 3. Running the experiment
The main script to run the wind interpolation experiment in the paper is `gp_interpolation.py`. However before running this, you have to first compute the weekly historical average of the wind velocity field by running the script `climatology.py`, and then run `spatial_pretraining.py` to learn the length scale from the weatherbench data. In the latter, you have the option to specify the geometry of the base manifold by including the flag `-g`. So if you want to use a Euclidean kernel (which is the default), run
```
python spatial_pretraining.py -g r2
```
and if you want to use a spherical kernel, run
```
python spatial_pretraining.py -g s2
```

However, pretraining may take several hours to complete (this can be accelerated by using the GPU however, it still takes a few hours) so if you want to avoid this, you can set the (log) length scale manually by adding the flag `-l` and the desired value for the log length scale in the `gp_interpolation.py` script. A sensible value for the log length scale is -1.65.

The `-g` flag to specify the underlying geometry is also available in `gp_interpolation.py` as well as a `--plot-sphere/--no-plot-sphere` flag which plots the results on a sphere and the lat/lon map respectively.

Examples:
1. The code below fits the GP with log length scale -1.625 using the spherical Matern kernel and plots the result on a lat/lon map
```
python gp_interpolation.py -l -1.65 -g s2 --no-plot-sphere
```
1. The following fits a GP with pretrained log length scale using the Euclidean Matern kernel and plots the result on a sphere
```
python gp_interpolation.py -l -1.65 -g r2 --plot-sphere
```




