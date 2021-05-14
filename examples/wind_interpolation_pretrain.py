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
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.kernel.scaled import ScaledKernel
from riemannianvectorgp.kernel.TFP import TFPKernel
from skyfield.api import wgs84, load, EarthSatellite
from scipy.interpolate import interp2d

class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self
    
    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key


def _deg2rad(x: np.ndarray):
    return (np.pi/180)*x


if __name__ == "__main__":
    rng = GlobalRNG()

    # Get wind data
    ds = xr.open_mfdataset('../datasets/global_wind_dataset/*.nc')
    lon = _deg2rad(ds.isel(time=0).lon.values)
    lat = _deg2rad(ds.isel(time=0).lat.values)
    mesh = np.meshgrid(lon, lat)

    phi, theta = jnp.meshgrid(lat, lon)
    phi = phi.flatten()
    theta = theta.flatten()
    m = jnp.stack(
        [phi, theta], axis=-1
    )

    u_2018_01_01 = ds.u10.sel(time='2018-01-01').values[0]
    v_2018_01_01 = ds.v10.sel(time='2018-01-01').values[0]
    u_2018_01_01, v_2018_01_01 = u_2018_01_01.flatten(), v_2018_01_01.flatten()

    m_cond = m
    v_cond = np.stack([u_2018_01_01, v_2018_01_01], axis=-1)

    # Set up kernel
    ev_kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 2, 2))
    ev_kernel_params = ev_kernel.init_params(next(rng))
    sub_kernel_params = ev_kernel_params.sub_kernel_params
    sub_kernel_params = sub_kernel_params._replace(log_length_scales=jnp.log(2))
    ev_kernel_params = ev_kernel_params._replace(sub_kernel_params=sub_kernel_params)
    ev_kernel_params = ev_kernel_params._replace(
        log_amplitude=-jnp.log(ev_kernel.matrix(ev_kernel_params, m, m)[0, 0, 0, 0])
    )

    # Set up Euclidean Sparse GP
    num_points = 20
    ev_sparse_gp = SparseGaussianProcess(
                    kernel=ev_kernel,
                    num_inducing=num_points**2,
                    num_basis=80,
                    num_samples=10)

    params, state = ev_sparse_gp.init_params_with_state(next(rng))

    lat_init = jnp.linspace(-jnp.pi/2, jnp.pi/2, num_points)
    lon_init = jnp.linspace(0, 2*jnp.pi, num_points)
    phi_init, theta_init = jnp.meshgrid(lat_init, lon_init)
    phi_init, theta_init = phi_init.flatten(), theta_init.flatten()
    init_locations = jnp.stack([phi_init, theta_init], axis=-1)

    params = params._replace(kernel_params=ev_kernel_params)
    params = params._replace(inducing_locations = init_locations)

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
        if i <= 10 or i % 20 == 0: print(i, "Loss:", train_loss)

    loss_after_optim = ev_sparse_gp.loss(
                        params,
                        state,
                        next(rng),
                        m_cond,
                        v_cond,
                        m_cond.shape[0])[0]

    print(f"Loss after optimizing: {loss_after_optim}")    

    # Plot predictions
    y_star = ev_sparse_gp(params, state, m_cond)

    constants = xr.open_dataset("../datasets/constants/constants.nc")
    lsm = constants.lsm.values

    plt.figure(figsize=(10, 5))
    scale = 300
    plt.scatter(params.inducing_locations[:, 1], params.inducing_locations[:, 0], color="blue", zorder=3, s=8)
    plt.contour(*mesh, lsm, zorder=1)
    for i in range(5):
        plt.quiver(m_cond[:,1], m_cond[:,0], y_star[i,:,0], y_star[i,:,1],
                   alpha=0.3,
                   color='black',
                   scale=scale,
                   width=0.003,
                   headwidth=3)
    plt.quiver(m_cond[:,1], m_cond[:,0], v_cond[:,0], v_cond[:,1], color='red', scale=scale, width=0.003, headwidth=3)

    plt.savefig("figs/wind_interpolation_sparse_gp.png")

