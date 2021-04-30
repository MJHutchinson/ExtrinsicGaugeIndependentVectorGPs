# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
%load_ext autoreload
%autoreload 2
# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
import optax
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
import sys

sys.path.insert(0, "..")
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.kernel import (
    SquaredExponentialCompactRiemannianManifoldKernel,
    MaternCompactRiemannianManifoldKernel,
    ScaledKernel,
    EigenBasisFunctionState,
)
from riemannianvectorgp.manifold import S1


from jax.config import config

config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)

# %%
def plot(x, y, f=None, samples=False):
    fig = plt.figure()
    # ax = fig.add_subplot(projection="polar")
    ax = fig.add_subplot()
    ax.scatter(x, y)
    # ax.set_rmin-(-4)

    if f is not None:
        m = jnp.mean(f, axis=0)
        u = jnp.quantile(f, 0.975, axis=0)
        l = jnp.quantile(f, 0.025, axis=0)

        ax.plot(x, m, linewidth=2)
        ax.fill_between(x, l, u, alpha=0.5)

        if samples:
            for i in range(f.shape[0]):
                ax.plot(x, f[i, :], color="gray", alpha=0.5)


def plot_gp(x, y, gp, params, state, samples=False):


    fig, axs = plt.subplots(3,1, figsize=(4, 9))

    state = gp.randomize(params, state, next(rng))
    inducing_mean = tf2jax.linalg.matvec(
        gp.kernel.matrix(params.kernel_params, params.inducing_locations, params.inducing_locations) + jax.vmap(jnp.diag)(jnp.exp(2 * params.inducing_pseudo_log_err_stddev)), 
        params.inducing_pseudo_mean
    )

    f = gp(params, state, x)[:, 0, :]
    f_prior = gp.prior(params.kernel_params, state.prior_state, x)[:, 0, :]

    k = kernel.matrix(params.kernel_params, x, x)[0, :, 0]

    x = x[:, 0]
    y = y[0, :]

    axs[0].scatter(x, y)
    # ax.set_rmin-(-4)

    m = jnp.mean(f, axis=0)
    u = jnp.quantile(f, 0.975, axis=0)
    l = jnp.quantile(f, 0.025, axis=0)

    axs[0].plot(x, m, linewidth=2)
    axs[0].fill_between(x, l, u, alpha=0.5)

    if samples:
        for i in range(f.shape[0]):
            axs[0].plot(x, f[i, :], color="gray", alpha=0.5)

    axs[0].scatter(params.inducing_locations[:,0], inducing_mean)
    axs[0].errorbar(
        params.inducing_locations[:,0],
        inducing_mean[0],
        yerr=jnp.exp(params.inducing_pseudo_log_err_stddev[0]),
        linestyle='none'
    )
    axs[0].set_title("Sparse GP")
    axs[1].plot(x, f_prior.T)
    axs[1].set_title("Prior samples")
    axs[2].plot(x, k)
    axs[2].set_title('Kernel')

    plt.tight_layout()

def plot_polar(x, y, f=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    # ax = fig.add_subplot()
    ax.scatter(x, y)
    ax.set_rmin(-4)

    if f is not None:
        m = jnp.mean(f, axis=0)
        u = jnp.quantile(f, 0.975, axis=0)
        l = jnp.quantile(f, 0.025, axis=0)

        ax.plot(x, m, linewidth=2)
        ax.fill_between(x, l, u, alpha=0.5)

        for i in range(f.shape[0]):
            ax.plot(x, f[i, :], color="gray", alpha=0.5)


class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key


# %%

rng = GlobalRNG()

# %%
x = jnp.expand_dims(jnp.linspace(0, jnp.pi * 2, 101), -1)
y = 2 * jnp.sin(x).T + jr.normal(next(rng), x.T.shape) / 10


# %%
plot(x, y)


# %%
s1 = S1(0.5)
kernel = ScaledKernel(SquaredExponentialCompactRiemannianManifoldKernel(s1, 100))
# kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(0.5,s1, 100))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.02))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, x, x)[0,0,0]))
k = kernel.matrix(kernel_params, x, x)
plt.plot(x[:, 0], k[0, :, 0])
# %%
rng = GlobalRNG()
gp = SparseGaussianProcess(kernel, 1, 1, 11, 67, 17)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(kernel_params=kernel_params)


# %%
params


# %%
x_ind = jnp.expand_dims(jnp.linspace(0, 2 * jnp.pi, 11), -1)
# y_ind = 2 * jnp.sin(x_ind * 4).T + jr.normal(next(rng), x_ind.T.shape) / 10
# params = params._replace(inducing_locations=x_ind, inducing_pseudo_mean=y_ind)
params = params._replace(inducing_locations=x_ind)


# %%
state = gp.resample_prior_basis(params, state, next(rng))
state = gp.randomize(params, state, next(rng))

# %%
state = gp.randomize(params, state, next(rng))

plot_gp(x, y, gp, params, state, samples=False)
print(gp.loss(params, state, next(rng), x, y, x.shape[0])[0])

# %%

state = gp.randomize(params, state, next(rng))
plot(x[:, 0], y[0, :], gp.prior(params.kernel_params, state.prior_state, x)[:, 0, :])
# %%
opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.01))
opt_state = opt.init(params)
# %%
for i in range(400):
    ((train_loss, state), grads) = jax.value_and_grad(gp.loss, has_aux=True)(
        params, state, next(rng), x, y, x.shape[0]
    )
    (updates, opt_state) = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if jnp.all(jnp.isnan(grads.kernel_params.sub_kernel_params.log_length_scale)):
        print("breaking for nan")
        break 
    if i <= 10 or i % 20 == 0:
        print(i, "Loss:", train_loss)

# %%

plot_gp(x, y, gp, params, state, samples=False)
print(gp.loss(params, state, next(rng), x, y, x.shape[0])[0])
# %%
