# Instructions

To reproduce the global wind interpolation results in the paper, simpy run
```
python gp_interpolation_clean.py
```
(maybe better to change the filename to `gp_wind_interpolation.py` or something and remember to add the necessary data to the log folder. Also add script to generate the observation data.).

In order to perform the experiment from scratch with different parameter settings and dates, follow the instructions below:

## 1. Installing additional dependencies
To run the experiments, you need to first install the package in `requirements_experiments.txt`. Additionally you will need to install `xesmf`, which is more tricky to install. The recomended method is via conda.

```
conda install -c conda-forge xesmf
```

## 2. Downloading the data
The code to download the data are saved in `../../datasets`.
- Run `python era5.py` to download the ERA5 wind reanalysis data (this may take up to an hour depending on the queue). You can change the date(s) of the wind reanalysis data to be retrieved in the script. The default is `2019-01-01`.
- Run `python wind_dataset.py` to download the weatherbench historical wind data. You can select the resolution of the wind data to be downloaded using the flag `--resolution`. The options are `1, 2` or `5` for 1.40625°, 2.8125° and 5.625° resolutions respectively.

## 3. Running the experiment
The main script to run the wind interpolation experiment in the paper is `gp_interpolation_clean.py` (reminder to change filename). However before running this, you have to first compute the weekly historical average of the wind velocity field by running the script `climatology.py`, and then run `spatial_pretraining.py` to learn the length scale from the weatherbench data. In the latter, you have the option to specify the geometry of the base manifold by including the flag `-g`. So if you want to use a Euclidean kernel (which is the default), run
```
python spatial_pretraining.py -g r2
```
and if you want to use a spherical kernel, run
```
python spatial_pretraining.py -g s2
```

However, pretraining may take several hours to complete (this can be accelerated by using the GPU however, it still may take a few hours) so if you want to avoid this, you can set the (log) length scale manually in the script `gp_interpolation_clean.py`.
