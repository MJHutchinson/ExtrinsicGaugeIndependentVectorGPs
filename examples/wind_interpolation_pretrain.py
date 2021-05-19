import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
import optax
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from riemannianvectorgp.sparse_gp import SparseGaussianProcess, SparseGaussianProcessParameters
from riemannianvectorgp.kernel.scaled import ScaledKernel
from riemannianvectorgp.kernel.TFP import TFPKernel
from riemannianvectorgp.kernel.utils import train_sparse_gp, GlobalRNG
import cartopy
import cartopy.crs as ccrs
import xesmf as xe

class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self
    
    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key


def _deg2rad(x: np.ndarray, offset: float=0.):
    return (np.pi/180)*x + offset


def _rad2deg(x: np.ndarray, offset: float=0.):
    return (180/np.pi)*(x - offset)


if __name__ == "__main__":
    rng = GlobalRNG()
    date = '2018-01-01 00:00'

    # Load ERA5 data and regrid
    ds = xr.open_mfdataset('../datasets/weatherbench_wind_data/*.nc')

    lon = ds.isel(time=0).lon
    lat = ds.isel(time=0).lat
    lon_size = lon.shape[0]
    lat_size = lat.shape[0]
    mesh = np.meshgrid(lon, lat)

    phi, theta = jnp.meshgrid(_deg2rad(lat, offset=jnp.pi/2), _deg2rad(lon))
    phi = phi.flatten()
    theta = theta.flatten()
    m = jnp.stack([phi, theta], axis=-1)

    # Get inputs and outputs
    weekly_mean = np.load("../datasets/weekly_climatology.npz")

    u = ds.u10.sel(time=date)
    v = ds.v10.sel(time=date)

    week_number = u['time'].dt.isocalendar().week
    u_mean = weekly_mean['u'][week_number]
    v_mean = weekly_mean['v'][week_number]
    u_anomaly = u.values - u_mean
    v_anomaly = v.values - v_mean

    u_mean, v_mean = u_mean.transpose().flatten(), v_mean.transpose().flatten()
    u_anomaly, v_anomaly = u_anomaly.transpose().flatten(), v_anomaly.transpose().flatten()

    m_cond = m
    v_cond = np.stack([v_anomaly, u_anomaly], axis=-1)
    mean = np.stack([v_mean, u_mean], axis=-1)

    # Set up kernel
    ev_kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 2, 2))
    ev_kernel_params = ev_kernel.init_params(next(rng))
    sub_kernel_params = ev_kernel_params.sub_kernel_params
    sub_kernel_params = sub_kernel_params._replace(log_length_scales=jnp.log(0.2))
    ev_kernel_params = ev_kernel_params._replace(sub_kernel_params=sub_kernel_params)
    ev_kernel_params = ev_kernel_params._replace(
        log_amplitude=-jnp.log(ev_kernel.matrix(ev_kernel_params, m, m)[0, 0, 0, 0])
    )

    # Set up Euclidean Sparse GP
    num_points = 20
    ev_sparse_gp = SparseGaussianProcess(
                    kernel=ev_kernel,
                    num_inducing=num_points**2,
                    num_basis=144,
                    num_samples=10)

    params, state = ev_sparse_gp.init_params_with_state(next(rng))

    # Initialize inducing locations on a regular grid
    lat_init = jnp.linspace(0, jnp.pi, num_points)
    lon_init = jnp.linspace(0, 2*jnp.pi, num_points)
    phi_init, theta_init = jnp.meshgrid(lat_init, lon_init)
    phi_init, theta_init = phi_init.flatten(), theta_init.flatten()
    init_locations = jnp.stack([phi_init, theta_init], axis=-1)

    params = params._replace(kernel_params=ev_kernel_params)
    params = params._replace(inducing_locations=init_locations)

    state = ev_sparse_gp.resample_prior_basis(params, state, next(rng))
    state = ev_sparse_gp.randomize(params, state, next(rng))
    
    loss_before_optim = ev_sparse_gp.loss(
                        params,
                        state,
                        next(rng),
                        m_cond,
                        v_cond,
                        m_cond.shape[0])[0]

    print(f"Loss before optimizing: {loss_before_optim}")

    opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.01))
    opt_state = opt.init(params)

    for i in range(500):
        ((train_loss, state), grads) = jax.value_and_grad(
                                        ev_sparse_gp.loss,
                                        has_aux=True)(params,
                                        state,
                                        next(rng),
                                        m_cond,
                                        v_cond,
                                        m_cond.shape[0])
                                       
        (updates, opt_state) = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i <= 10 or i % 20 == 0: print(i, "Loss:", train_loss,
                                        "Log length scale:", params.kernel_params.sub_kernel_params.log_length_scales,
                                        "Log amplitude:", params.kernel_params.log_amplitude,
                                        "Error stddev:", jnp.exp(params.log_error_stddev).mean()
                                        )

    loss_after_optim = ev_sparse_gp.loss(
                        params,
                        state,
                        next(rng),
                        m_cond,
                        v_cond,
                        m_cond.shape[0])[0]

    print(f"Loss after optimizing: {loss_after_optim}")   

    # Plot predictions
    prediction = ev_sparse_gp(params, state, m)
    anomaly_mean = prediction.mean(axis=0)
    anomaly_std = jnp.sqrt(((prediction - mean)**2).mean(axis=0).sum(axis=-1))

    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    scale = 300

    plt.quiver(_rad2deg(m_cond[:,1]),
               _rad2deg(m_cond[:,0], offset=jnp.pi/2),
               anomaly_mean[:,1] + mean[:,1],
               anomaly_mean[:,0] + mean[:,0],
               alpha=0.5,
               color='blue',
               scale=scale,
               width=0.003,
               headwidth=3,
               zorder=2)

    plt.scatter(_rad2deg(params.inducing_locations[:, 1]),
                _rad2deg(params.inducing_locations[:, 0], offset=jnp.pi/2),
                color="red",
                zorder=3,
                s=8)

    plt.savefig("figs/wind_interpolation_sparse_gp_pretrain.png")

    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    scale = 300

    plt.quiver(_rad2deg(m_cond[:,1]),
               _rad2deg(m_cond[:,0], offset=jnp.pi/2),
               v_cond[:,1] + mean[:,1],
               v_cond[:,0] + mean[:,0],
               alpha=0.5,
               color='black',
               scale=scale,
               width=0.003,
               headwidth=3,
               zorder=2)

    plt.savefig("figs/ground_truth.png")


