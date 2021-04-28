# %%
%load_ext autoreload
%autoreload 2
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

from riemannianvectorgp.kernel import SquaredExponentialCompactRiemannianManifoldKernel, TFPKernel, ScaledKernel, FourierFeatures
from riemannianvectorgp.manifold import S1
# %%
import matplotlib.pyplot as plt
def plot(x,y=None,f=None, samples=False):    
    fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    ax = fig.add_subplot()

    if y is not None:
        ax.scatter(x,y, zorder=4)
    if f is not None:
        m = jnp.mean(f, axis=0)
        u = jnp.quantile(f, 0.975, axis=0)
        l = jnp.quantile(f, 0.025, axis=0)
        
        # if samples:
        #     for i in range(f.shape[0]):
        #         ax.plot(x,f[i,:], color="gray",alpha=0.5, zorder=1)

        ax.plot(x,m,linewidth=2, zorder=2)
        ax.fill_between(x, l, u, alpha=0.5, zorder=3)
        

class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self
    
    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key

rng = GlobalRNG()
# %%

def test_kernel(kernel, kernel_params, x, key, num_samples=1000, num_basis_functions=1000):
    ff = FourierFeatures(kernel, num_basis_functions)
    state = ff.init_state(kernel_params, num_samples, key)
    f = ff(kernel_params, state, x)

    k = kernel.matrix(kernel_params, x, x)
    m_ff = jnp.mean(f,axis=0)
    k_ff = jnp.mean(f[..., :, np.newaxis] * f[..., np.newaxis, :], axis=0)

    m_err = - m_ff
    m_mean_err = jnp.mean(m_err)
    m_max_err = jnp.max(jnp.abs(m_err))

    k_err = k - k_ff
    k_mean_err = jnp.mean(k_err)
    k_max_err = jnp.max(jnp.abs(k_err))

    return m_mean_err, m_max_err, k_mean_err, k_max_err


# %%
kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 1, 1))
kernel_params = kernel.init_params(next(rng))
x = jnp.linspace(-5, 5, 101)[:, np.newaxis]
num_samples = 1000
num_basis_functions = 1000
key = next(rng)
# %%

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)
# %%

k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis] * f[..., np.newaxis, :], axis=0)
# %%
plot(x[:, 0], k[0,0,:], k_ff[:,0,:], samples=False)
# %%

k_err = k - k_ff
mean_err = jnp.mean(k_err)
max_err = jnp.max(jnp.abs(k_err))

print(mean_err, max_err)
# %%

print(test_kernel(kernel, kernel_params, x, next(rng)))

# %%

kernel = SquaredExponentialCompactRiemannianManifoldKernel(S1(0.5), truncation = 500)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(log_length_scale=jnp.log(0.1 * kernel.manifold.radius))
x = jnp.linspace(0, 2*2*jnp.pi*kernel.manifold.radius, 101)[:, np.newaxis]
num_samples = 10000
num_basis_functions = 10000
key = next(rng)

# %%
ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)

# %%

k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis] * f[..., np.newaxis, :], axis=0)
# %%
plot(x[:, 0], k[0,0,:], k_ff[:,0,:], samples=False)
# %%
