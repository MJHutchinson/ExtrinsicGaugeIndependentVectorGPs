import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel,
)
from riemannianvectorgp.utils import train_sparse_gp, GlobalRNG
from examples.wind_interpolation.utils import deg2rad, refresh_kernel
import matplotlib.pyplot as plt
import click
import pickle


def _get_v_cond(ds, date, climatology):
    u = ds.u10.sel(time=date)
    v = ds.v10.sel(time=date)
    week_number = u["time"].dt.isocalendar().week
    u_mean = climatology["u"][week_number - 1]
    v_mean = climatology["v"][week_number - 1]
    u_anomaly = u.values - u_mean
    v_anomaly = v.values - v_mean
    u_anomaly, v_anomaly = (
        u_anomaly.transpose().flatten(),
        v_anomaly.transpose().flatten(),
    )
    v_cond = np.stack([v_anomaly, u_anomaly], axis=-1)
    return v_cond


@click.command()
@click.option("--logdir", default="log", type=str)
@click.option("--samples", "-s", default=150, type=int)
@click.option("--epochs", "-e", default=800, type=int)
@click.option("--geometry", "-g", default="r2", type=click.Choice(["r2", "s2"]))
def main(logdir, samples, epochs, geometry):
    rng = GlobalRNG()

    # Load past reanalysis data
    ds = xr.open_mfdataset("../../datasets/weatherbench_wind_data/*.nc")
    total_length = ds.dims["time"]
    idxs = jnp.arange(total_length)
    idxs = jr.permutation(next(rng), idxs)  # Shuffle indices

    # Load climatology
    climatology = np.load("../../datasets/climatology/weekly_climatology.npz")

    # Get input locations
    lon = ds.isel(time=0).lon
    lat = ds.isel(time=0).lat
    lat, lon = jnp.meshgrid(
        deg2rad(lat, offset=jnp.pi / 2), deg2rad(lon)
    )  # Reparametrise as lat=(0, pi) and lon=(0, 2pi)
    lat = lat.flatten()
    lon = lon.flatten()
    m_cond = jnp.stack([lat, lon], axis=-1)

    # Set up kernel
    if geometry == "r2":
        # kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 2, 2))
        kernel = ScaledKernel(TFPKernel(tfk.MaternThreeHalves, 2, 2))
    elif geometry == "s2":
        S2 = EmbeddedS2(1.0)
        kernel = ScaledKernel(
            ManifoldProjectionVectorKernel(
                MaternCompactRiemannianManifoldKernel(3 / 2, S2, 144), S2
            )
        )  # 144 is the maximum number of basis functions we have implemented

    # Set up sparse GP
    num_points = 15
    sparse_gp = SparseGaussianProcess(
        kernel=kernel, num_inducing=num_points ** 2, num_basis=144, num_samples=10
    )

    # Set initial inducing locations on a regular grid
    lat_init = jnp.linspace(lat[0], lat[-1], num_points)
    lon_init = jnp.linspace(lon[0], lon[-1], num_points)
    phi_init, theta_init = jnp.meshgrid(lat_init, lon_init)
    phi_init, theta_init = phi_init.flatten(), theta_init.flatten()
    init_inducing_locations = jnp.stack([phi_init, theta_init], axis=-1)

    # Set initial length scale
    init_length_scale = 0.15
    init_log_length_scale = jnp.log(init_length_scale)

    log_length_scales, log_amplitudes = [], []
    for i, idx in enumerate(idxs[:samples]):
        date = ds.time[idx]
        print("Sample:", i, "Date:", date.values)

        # Initialise parameters and state
        params, state = sparse_gp.init_params_with_state(next(rng))
        kernel_params = refresh_kernel(
            next(rng), kernel, m_cond, geometry, init_log_length_scale
        )

        params = params._replace(kernel_params=kernel_params)
        params = params._replace(inducing_locations=init_inducing_locations)

        state = sparse_gp.resample_prior_basis(params, state, next(rng))
        state = sparse_gp.randomize(params, state, next(rng))

        # Get conditioning values
        v_cond = _get_v_cond(ds, date, climatology)

        # Train sparse GP
        params, state, _ = train_sparse_gp(
            sparse_gp, params, state, m_cond, v_cond, rng, epochs=epochs
        )

        if geometry == "r2":
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scales
            log_amplitude = params.kernel_params.log_amplitude
        elif geometry == "s2":
            log_length_scale = params.kernel_params.sub_kernel_params.log_length_scale
            log_amplitude = params.kernel_params.log_amplitude

        print("Log length scale:", log_length_scale, "Log amplitude:", log_amplitude)

        log_length_scales.append(log_length_scale)
        log_amplitudes.append(log_amplitude)

    log_length_scales = np.stack(log_length_scales)
    log_amplitudes = np.stack(log_amplitudes)

    np.save(logdir + "/" + geometry + "_log_length_scale.npy", log_length_scales.mean())

    # with open(logdir+'/'+geometry+'_params_pretrained.pickle', "wb") as f:
    #     pickle.dump({'log_length_scale': log_length_scales, 'log_amplitude': log_amplitudes}, f)


if __name__ == "__main__":
    main()
