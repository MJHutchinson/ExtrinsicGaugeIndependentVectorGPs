# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
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
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
import sys

sys.path.insert(0, "..")
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.kernel import (
    SquaredExponentialCompactRiemannianManifoldKernel,
    EigenBasisFunctionState,
)
from riemannianvectorgp.manifold import S1


# %%
def plot(x, y, f=None):
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

        for i in range(f.shape[0]):
            ax.plot(x, f[i, :], color="gray", alpha=0.5)


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
y = 2 * jnp.sin(x * 4).T + jr.normal(next(rng), x.T.shape) / 10


# %%
plot(x, y)


# %%
s1 = S1(0.5)
kernel = SquaredExponentialCompactRiemannianManifoldKernel(s1, 100)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(log_length_scale=jnp.log(0.05))
k = kernel.matrix(kernel_params, x, x)
plt.plot(x[:, 0], k[0, :, 0])
# %%

gp = SparseGaussianProcess(kernel, 1, 1, 11, 67, 17)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(kernel_params=kernel_params)


# %%
params


# %%
params = params._replace(
    inducing_locations=jnp.expand_dims(jnp.linspace(0, 2 * jnp.pi, 11), -1)
)


# %%
state = gp.resample_prior_basis(params, state, next(rng))
state = gp.randomize(params, state, next(rng))


# %%
params.inducing_locations.shape


# %%
state = gp.randomize(params, state, next(rng))
plot(x[:, 0], y[0, :], gp(params, state, x)[:, 0, :])

# %%

state = gp.randomize(params, state, next(rng))
plot(x[:, 0], y[0, :], gp.prior(params.kernel_params, state.prior_state, x)[:, 0, :])
# %%
opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.01))
opt_state = opt.init(params)
# %%
for i in range(200):
    ((train_loss, state), grads) = jax.value_and_grad(gp.loss, has_aux=True)(
        params, state, next(rng), x, y, x.shape[0]
    )
    (updates, opt_state) = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i <= 10 or i % 20 == 0:
        print(i, "Loss:", train_loss)

# %%
state = gp.randomize(params, state, next(rng))
plot(x[:, 0], y[0, :], gp(params, state, x)[:, 0, :])
plt.scatter(params.inducing_locations, params.inducing_pseudo_mean)
plt.errorbar(
    params.inducing_locations[:, 0],
    params.inducing_pseudo_mean[0],
    yerr=params.inducing_pseudo_log_err_stddev[0],
)
# %%
n = jnp.arange(10)
l = s1.laplacian_eigenvalue(n)
plt.scatter(n, l)

# %%
x = jnp.expand_dims(jnp.linspace(0, jnp.pi * 2, 101), -1)
n = jnp.arange(10)
eigs = s1.laplacian_eigenfunction(n, x)

n_ = n  # [np.newaxis, :]
x_ = x  # [..., np.newaxis]
freq = n_ // 4
phase = (jnp.pi / 2) * (n_ % 2)
neg = -((((n_ // 2) + 1) % 2) * 2 - 1)
phase_ = phase * neg
eig_ = jnp.cos(x_ * freq + phase_)[..., np.newaxis, :, :]

# fig = plt.figure()
# # ax = fig.add_subplot(projection="polar")
# ax = fig.add_subplot()
# # ax.set_rorigin(-3)

# for i in range(eigs.shape[-1]):
#     ax.plot(x[0, :], eigs[0, :, i], alpha=0.5, label=f"{i}")
# plt.legend()

# %%
kernel = SquaredExponentialCompactRiemannianManifoldKernel(s1, 100)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(log_length_scale=jnp.array([-3]))
print(kernel.matrix(kernel_params, jnp.array([0]), jnp.array([0])))
# %%
for l in np.linspace(-7, 0, num=20):
    kernel_params = kernel_params._replace(log_length_scale=jnp.array([l]))
    k = kernel.matrix(kernel_params, x, x)
    plt.plot(x[:, 0], k[0, :, 0], label=f"{np.exp(l):0.4f}")

plt.legend()
# %%
for l in np.linspace(-7, 0, num=20):
    kernel_params = kernel_params._replace(log_length_scale=jnp.array([l]))
    k = kernel.matrix(kernel_params, x, x)
    plt.plot(
        x[:, 0], k[0, :, 0] / k[0, 0, 0], label=f"{np.exp(l):0.4f}" + " " + f"{l:0.2f}"
    )

plt.legend()
# %%
truncation = 100
eigenindicies = jnp.arange(truncation)
x1 = x
x2 = x
lengthscale = jnp.exp(kernel_params.log_length_scale)
fx1 = kernel.basis_functions(kernel_params, EigenBasisFunctionState(eigenindicies), x1)
fx2 = kernel.basis_functions(kernel_params, EigenBasisFunctionState(eigenindicies), x2)
lam = kernel.manifold.laplacian_eigenvalue(eigenindicies)
spectrum = jnp.exp(-(jnp.power(lengthscale, 2) * lam / 2))
k = jnp.sum(fx1[..., np.newaxis, :] * fx2[..., np.newaxis, :, :] * spectrum, axis=-1)

# %%
fig = plt.figure()
# ax = fig.add_subplot(projection="polar")
ax = fig.add_subplot()
# ax.set_rorigin(-3)

for i in range(fx1.shape[-1]):
    ax.plot(x[0, :], jnp.sqrt(spectrum[i]) * fx1[0, :, i], alpha=0.5, label=f"{i}")
plt.legend()
# %%

plt.scatter(eigenindicies, spectrum)
# %%
k = kernel.matrix(kernel_params, x, x)
plt.plot(x[:, 0], k[0, :, 0])
# %%

# %%
