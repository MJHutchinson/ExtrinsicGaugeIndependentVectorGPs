# Instructions

To reproduce the global wind interpolation results in our paper, simpy run
```
python gp_wind_interpolation.py
```
The data used to reproduce the plots are saved in the `log` directory. Make sure to install all the packages in `requirements_experiments.txt` first and set the correct data paths in the script before running.

In order to perform the experiment from scratch with different configurations and parameter settings (e.g. different date, kernel, satellite), follow the instructions below:

## 1. Installing `xesmf`
In addition to the packages in `requirements_experiments.txt`, you will need the package `xesmf`, which is more tricky to install. The recomended method is via conda.

```
conda install -c conda-forge xesmf
```

## 2. Downloading the data
The code to download the wind data are saved in `../../datasets`.
- Run `python era5.py` to download the ERA5 wind reanalysis data (this may take up to an hour depending on the queue). You can change the date(s) of the wind reanalysis data to be retrieved in the script. The default is `2019-01-01`.
- Run `python wind_dataset.py` to download the weatherbench historical wind data. You can select the resolution of the wind data to be downloaded using the flag `--resolution`. The options are `1, 2` or `5` for 1.40625°, 2.8125° and 5.625° resolutions respectively (default is `5`).

## 3. Generating satellite observations and learning the length scale
You will need to run the following scripts in order:
- `generate_satellite_data.py` to generate the satellite observations from the ERA5 data, used to fit the GP.
- `spatial_pretraining.py` to learn the length scale from the weatherbench historic wind field data.
Here, you have the option to specify the geometry of the base manifold by including the flag `-g`. So if you want to use a Euclidean kernel (which is the default), run
```
python spatial_pretraining.py -g r2
```
and if you want to use a spherical kernel, run
```
python spatial_pretraining.py -g s2
```

Note that learning the lengthscale may take several hours to complete on GPU under the default settings. If you want faster results, you can reduce the number of training epochs and number of samples via the flags `-e` and `-s` respectively. The default is 800 and 150 resepectively. Example:
```
python spatial_pretraining.py -g s2 -e 400 -s 20
```

## 4. Fit the GP using the main script
Once the above is done, you can fit the GP on the generated observations and save the data for plotting by running `gp_wind_interpolation.py`.
