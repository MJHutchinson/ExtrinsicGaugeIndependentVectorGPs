# Instructions

The main script to perform the global wind interpolation is `gp_wind_interpolation.py`.
The data that we used to produce the plots in our paper are uploaded in the `log` directory, so you can run this script directly to reproduce our results without having to run anything else (provided that you set the correct data paths in the script).

### Requirements
You need to install the packages in `requirements.txt`.

## Producing the results from scratch
In order to produce the results from scratch (for example with different configurations and parameter settings), follow the instructions below:

### 1. Installing additional packages
Install the packages in `requirements_experiments.txt` in addition to the package `xesmf`, which is more tricky to install. The recomended method is via conda.

```
pip install -r requirements_experiments.txt
conda install -c conda-forge xesmf
```

### 2. Downloading the data
The code to download the wind data are saved in `../../datasets`.
- Run `python era5.py` to download the ERA5 wind reanalysis data (this may take up to an hour depending on the queue). You can change the date(s) of the wind reanalysis data to be retrieved in the script. The default is `2019-01-01`.
- Run `python wind_dataset.py` to download the weatherbench historical wind data. You can select the resolution of the wind data via the flag `--resolution`. The options are `1, 2` or `5` for 1.40625°, 2.8125° and 5.625° resolutions respectively (default is `5`).

### 3. Generating satellite observations and learning the length scale
Run the following scripts:
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

Note that computing the lengthscale may take several hours to complete on GPU under the default settings. If you want faster results, you can reduce the number of training epochs and number of samples via the flags `-e` and `-s` respectively. The default is 800 and 150 resepectively. Example:
```
python spatial_pretraining.py -g s2 -e 400 -s 20
```

### 4. Fit the GP using the main script
Once the above is done, fit the GP on the generated observations and save the data for plotting later by running `gp_wind_interpolation.py`.

## Space-time interpolation
It is also possible to perform space-time interpolation by considering a space-time product kernel.

TODO

