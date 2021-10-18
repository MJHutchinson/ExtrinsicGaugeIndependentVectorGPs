# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# %load_ext autoreload
# %autoreload 2
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
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
    TFPKernel,
    ManifoldProjectionVectorKernel,
)
from riemannianvectorgp.manifold import S1, EmbeddedS1
from einops import rearrange


from jax.config import config

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

# %%
def plot(x, y, gp, params, state, samples=False):
    fig = plt.figure()
    # ax = fig.add_subplot(projection="polar")
    ax = fig.add_subplot()

    state = gp.randomize(params, state, next(rng))

    K = gp.kernel.matrix(
        params.kernel_params, params.inducing_locations, params.inducing_locations
    )
    K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")

    M, OD = params.inducing_pseudo_log_err_stddev.shape
    Sigma = rearrange(
        jnp.exp(2 * params.inducing_pseudo_log_err_stddev), "M OD -> (M OD)"
    )
    Sigma = jnp.diag(Sigma)
    inducing_pseudo_mean = rearrange(params.inducing_pseudo_mean, "M OD -> (M OD)")

    inducing_mean = tf2jax.linalg.matvec(
        K + Sigma,
        inducing_pseudo_mean,
    )
    inducing_mean = rearrange(inducing_mean, "(M OD) -> M OD", M=M, OD=OD)
    f = gp(params, state, x)[:, :, 0]
    f_prior = gp.prior(params.kernel_params, state.prior_state, x)[:, :, 0]

    k = kernel.matrix(params.kernel_params, x, x)[0, :, 0]

    x = x[:, 0]
    y = y[:, 0]

    ax.scatter(x, y)

    m = jnp.mean(f, axis=0)
    u = jnp.quantile(f, 0.975, axis=0)
    l = jnp.quantile(f, 0.025, axis=0)

    ax.plot(x, m, linewidth=2)
    ax.fill_between(x, l, u, alpha=0.5)

    if samples:
        for i in range(f.shape[0]):
            ax.plot(x, f[i, :], color="gray", alpha=0.5)

    ax.scatter(params.inducing_locations[:, 0], inducing_mean, zorder=6)
    ax.errorbar(
        params.inducing_locations[:, 0],
        inducing_mean[:, 0],
        yerr=jnp.exp(params.inducing_pseudo_log_err_stddev[:, 0]),
        linestyle="none",
        zorder=5,
    )


def plot_gp(x, y, gp, params, state, samples=False):

    fig, axs = plt.subplots(3, 1, figsize=(4, 9))

    state = gp.randomize(params, state, next(rng))

    K = gp.kernel.matrix(
        params.kernel_params, params.inducing_locations, params.inducing_locations
    )
    K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")

    M, OD = params.inducing_pseudo_log_err_stddev.shape
    Sigma = rearrange(
        jnp.exp(2 * params.inducing_pseudo_log_err_stddev), "M OD -> (M OD)"
    )
    Sigma = jnp.diag(Sigma)
    inducing_pseudo_mean = rearrange(params.inducing_pseudo_mean, "M OD -> (M OD)")

    inducing_mean = tf2jax.linalg.matvec(
        K + Sigma,
        inducing_pseudo_mean,
    )
    inducing_mean = rearrange(inducing_mean, "(M OD) -> M OD", M=M, OD=OD)
    f = gp(params, state, x)[:, :, 0]
    f_prior = gp.prior(params.kernel_params, state.prior_state, x)[:, :, 0]

    k = kernel.matrix(params.kernel_params, x, x)[0, :, 0]

    x = x[:, 0]
    y = y[:, 0]

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

    axs[0].scatter(params.inducing_locations[:, 0], inducing_mean, zorder=6)
    axs[0].errorbar(
        params.inducing_locations[:, 0],
        inducing_mean[:, 0],
        yerr=jnp.exp(params.inducing_pseudo_log_err_stddev[:, 0]),
        linestyle="none",
        zorder=5,
    )
    axs[0].set_title("Sparse GP")
    axs[1].plot(x, f_prior.T)
    axs[1].set_title("Prior samples")
    axs[2].plot(x, k)
    axs[2].set_title("Kernel")

    plt.tight_layout()


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
y = 2 * jnp.sin(x) + jr.normal(next(rng), x.shape) / 10


# %%
kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 1, 1))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(jnp.array(0.5)))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, x, x)[0, 0, 0])
)
k = kernel.matrix(kernel_params, x, x)
plt.plot(x[:, 0], k[0, :, 0, 0], label="EQ - euclidean")

s1 = S1(0.5)
kernel = ScaledKernel(SquaredExponentialCompactRiemannianManifoldKernel(s1, 100))
# kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(4.5,s1, 100))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.15))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, x, x)[0, 0, 0])
)
k = kernel.matrix(kernel_params, x, x)
plt.plot(x[:, 0], k[0, :, 0, 0], label="EQ")
for nu in [0.5, 1.5, 2.5, 3.5, 4.5]:
    kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(nu, s1, 100))
    kernel_params = kernel.init_params(next(rng))
    sub_kernel_params = kernel_params.sub_kernel_params
    sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.15))
    kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
    kernel_params = kernel_params._replace(
        log_amplitude=-jnp.log(kernel.matrix(kernel_params, x, x)[0, 0, 0])
    )
    k = kernel.matrix(kernel_params, x, x)
    plt.plot(x[:, 0], k[0, :, 0, 0], label=f"{nu}")
plt.legend()
plt.title(f"LS: {jnp.exp(kernel_params.sub_kernel_params.log_length_scale):0.3f}")
# %%
s1 = S1(1.0)
kernel = ScaledKernel(SquaredExponentialCompactRiemannianManifoldKernel(s1, 100))
# kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(1.5, s1, 100))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, x, x)[0, 0, 0])
)
k = kernel.matrix(kernel_params, x, x)

print(k.shape)
plt.plot(x[:, 0], k[0, :, 0, 0], label="EQ")
# %%
s1 = EmbeddedS1(1.0)
kernel = ScaledKernel(
    ManifoldProjectionVectorKernel(
        # SquaredExponentialCompactRiemannianManifoldKernel(s1, 100)
        MaternCompactRiemannianManifoldKernel(0.5, s1, 100),
        s1,
    )
)
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, x, x)[0, 0, 0, 0])
)
k = kernel.matrix(kernel_params, x, x)
print(k.shape)
plt.plot(x[:, 0], k[0, :, 0, 0], label="EQ")
# %%
rng = GlobalRNG()
gp = SparseGaussianProcess(kernel, 11, 67, 100)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(kernel_params=kernel_params)


# %%
params


# %%
x_ind = jnp.expand_dims(jnp.linspace(0, 2 * jnp.pi, 11), -1)
y_ind = 2 * jnp.sin(x_ind) + jr.normal(next(rng), x_ind.shape) / 10
y_ind = jnp.zeros_like(x_ind)

params = gp.set_inducing_points(params, x_ind, y_ind, jnp.ones_like(y_ind) * 0.01)


# %%
state = gp.resample_prior_basis(params, state, next(rng))
state = gp.randomize(params, state, next(rng))

# %%

plot_gp(x, y, gp, params, state, samples=True)

# %%
opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.01))
opt_state = opt.init(params)

# %%
debug_params = [params]
debug_states = [state]
debug_keys = [rng.key]

# %%
for i in range(600):
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
    debug_params.append(params)
    debug_states.append(state)
    debug_keys.append(rng.key)

# %%

plot_gp(x, y, gp, debug_params[-1], debug_states[-1], samples=True)

# %%
jax.value_and_grad(gp.loss, has_aux=True)(
    debug_params[-1], debug_states[-1], debug_keys[-1], x, y, x.shape[0]
)

# %%
config.update("jax_debug_nans", True)

# %%
jax.grad(gp.loss, has_aux=True)(
    debug_params[-1], debug_states[-1], debug_keys[-1], x, y, x.shape[0]
)

# %%
debug_params[-1]

# %%

plot_gp(x, y, gp, params, state, samples=False)
print(gp.loss(params, state, next(rng), x, y, x.shape[0])[0])
# %%
