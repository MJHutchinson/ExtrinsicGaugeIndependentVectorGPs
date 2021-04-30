# %%
import itertools
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

from riemannianvectorgp.kernel import (
    SquaredExponentialCompactRiemannianManifoldKernel,
    MaternCompactRiemannianManifoldKernel,
    TFPKernel,
    ScaledKernel,
    FourierFeatures,
)
from riemannianvectorgp.manifold import S1


class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key


def eval_kernel(
    kernel,
    kernel_params,
    x,
    key,
    num_samples=10000,
    num_basis_functions=10000,
    mean_tol=0.05,
    max_tol=0.2,
):
    ff = FourierFeatures(kernel, num_basis_functions)
    state = ff.init_state(kernel_params, num_samples, key)
    f = ff(kernel_params, state, x)

    k = kernel.matrix(kernel_params, x, x)
    m_ff = jnp.mean(f, axis=0)
    k_ff = jnp.mean(f[..., :, np.newaxis] * f[..., np.newaxis, :], axis=0)

    # k = k / k[0,0,0]
    # m_ff = m_ff / k[0,0,0]
    # k_ff = k_ff / k[0,0,0]

    m_err = -m_ff
    m_mean_err = jnp.mean(jnp.abs(m_err))
    m_max_err = jnp.max(jnp.abs(m_err))

    k_err = (k - k_ff) / k[0, 0, 0]  # normalise by the kernel scale
    k_mean_err = jnp.mean(jnp.abs(k_err))
    k_max_err = jnp.max(jnp.abs(k_err))

    # print(k[0,0,0])
    # print(m_mean_err, m_max_err, k_mean_err, k_max_err)

    assert (
        m_mean_err < mean_tol
    ), f"mean mean error: {m_mean_err}, tolerance: {mean_tol}. Kernel {kernel} with params {kernel_params}"
    assert (
        m_max_err < max_tol
    ), f"mean max error: {m_max_err}, tolerance: {max_tol}. Kernel {kernel} with params {kernel_params}"
    assert (
        k_mean_err < mean_tol
    ), f"covariance mean error: {k_mean_err}, tolerance: {mean_tol}. Kernel {kernel} with params {kernel_params}"
    assert (
        k_max_err < max_tol
    ), f"covariance max error: {k_max_err}, tolerance: {max_tol}. Kernel {kernel} with params {kernel_params}"


def plot(x, y=None, f=None, samples=False):
    fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    ax = fig.add_subplot()

    if y is not None:
        ax.scatter(x, y, zorder=4)
    if f is not None:
        m = jnp.mean(f, axis=0)
        u = jnp.quantile(f, 0.975, axis=0)
        l = jnp.quantile(f, 0.025, axis=0)

        # if samples:
        #     for i in range(f.shape[0]):
        #         ax.plot(x,f[i,:], color="gray",alpha=0.5, zorder=1)

        ax.plot(x, m, linewidth=2, zorder=2)
        ax.fill_between(x, l, u, alpha=0.5, zorder=3)


def test_ExponentialQuadratic():
    rng = GlobalRNG()
    for ls in jnp.linspace(0.01, 10, 20):
        kernel = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
        kernel_params = kernel.init_params(next(rng))
        kernel_params = kernel_params._replace(log_length_scales=jnp.log(ls))
        x = jnp.linspace(-5, 5, 101)[:, np.newaxis]
        key = next(rng)
        eval_kernel(kernel, kernel_params, x, key)


def test_S1ExponentialQuadratic():
    rng = GlobalRNG()
    s1 = S1(0.5)
    for ls in jnp.exp(jnp.linspace(-4.5, 0, 10)):
        kernel = SquaredExponentialCompactRiemannianManifoldKernel(s1, truncation=10000)
        kernel_params = kernel.init_params(next(rng))
        kernel_params = kernel_params._replace(log_length_scale=jnp.log(ls))
        x = jnp.linspace(0, 2 * 2 * jnp.pi * kernel.manifold.radius, 101)[:, np.newaxis]
        key = next(rng)

        eval_kernel(kernel, kernel_params, x, key)


def test_S1Matern():
    rng = GlobalRNG()
    s1 = S1(0.5)
    for ls, nu in itertools.product(
        np.exp(jnp.linspace(-4.5, 0, 10)), jnp.linspace(0.5, 2.5, 3)
    ):
        kernel = MaternCompactRiemannianManifoldKernel(nu, s1, truncation=10000)
        kernel_params = kernel.init_params(next(rng))
        kernel_params = kernel_params._replace(log_length_scale=jnp.log(ls))
        x = jnp.linspace(0, 2 * 2 * jnp.pi * kernel.manifold.radius, 101)[:, np.newaxis]
        key = next(rng)

        eval_kernel(kernel, kernel_params, x, key)


# %%
if __name__ == "__main__":
    test_ExponentialQuadratic()
    test_S1ExponentialQuadratic()
    test_S1Matern()
