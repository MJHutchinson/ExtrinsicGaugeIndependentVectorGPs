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

import sys
sys.path.insert(0, "..")
from riemannianvectorgp.manifold import S1, EmbeddedS1, ProductManifold, EmbeddedProductManifold
from riemannianvectorgp.kernel import (
    ScaledKernel, 
    ManifoldProjectionVectorKernel, 
    MaternCompactRiemannianManifoldKernel, 
    SquaredExponentialCompactRiemannianManifoldKernel,
    ProductKernel,
    FourierFeatures,
    TFPKernel
)

from jax.config import config
config.update("jax_enable_x64", True)
# %%
import matplotlib.pyplot as plt       

class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self
    
    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key

rng = GlobalRNG()

def eval_kernel(kernel, kernel_params, x, key, num_samples=1000, num_basis_functions=1000):
    ff = FourierFeatures(kernel, num_basis_functions)
    state = ff.init_state(kernel_params, num_samples, key)
    f = ff(kernel_params, state, x)

    k = kernel.matrix(kernel_params, x, x)
    m_ff = jnp.mean(f,axis=0)
    k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)

    m_err = - m_ff
    m_mean_err = jnp.mean(m_err)
    m_max_err = jnp.max(jnp.abs(m_err))

    k_err = k - k_ff
    k_mean_err = jnp.mean(k_err)
    k_max_err = jnp.max(jnp.abs(k_err))

    return m_mean_err, m_max_err, k_mean_err, k_max_err


# %%
def plot_kernel_approx(x,mu,K,mu_approx, K_approx, samples=False):    
    fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    ax = fig.add_subplot()

    ax.plot(x,K_approx, zorder=2, color='blue', label='approx')
    # ax.plot(x,mu_approx, zorder=2, color='blue')
    # ax.fill_between(x, mu_approx+K_approx, mu_approx-K_approx, alpha=0.5, zorder=3, color='blue')

    ax.plot(x,K,linewidth=2, zorder=4, color='orange', label='actual')
    plt.legend()
    # ax.plot(x,mu,linewidth=2, zorder=4, color='orange')
    # ax.fill_between(x, mu+K, mu-K, alpha=0.5, zorder=3, color='orange')

# %%
kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 1, 1))
kernel_params = kernel.init_params(next(rng))
x = jnp.linspace(-5, 5, 101)[:, np.newaxis]
num_samples = 20000
num_basis_functions = 10000
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)

k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[0,:,0,0], m_ff[:,0], k_ff[0,:,0,0])
# plt.title(f"Square exponential, samples: {num_samples}, basis functions: {num_basis_functions}")
# %%
i = 50
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[i,:,0,0], m_ff[:,0], k_ff[i,:,0,0])

# %%

k_err = k - k_ff
mean_err = jnp.mean(k_err)
max_err = jnp.max(jnp.abs(k_err))

print(mean_err, max_err)
# %%

print(eval_kernel(kernel, kernel_params, x, next(rng)))
# %%
kernel = ScaledKernel(TFPKernel(tfk.MaternOneHalf, 1, 1))
kernel_params = kernel.init_params(next(rng))
x = jnp.linspace(-5, 5, 101)[:, np.newaxis]
num_samples = 20000
num_basis_functions = 10000
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)

k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[0,:,0,0], m_ff[:,0], k_ff[0,:,0,0])
# plt.title(f"Square exponential, samples: {num_samples}, basis functions: {num_basis_functions}")
# %%
i = 50
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[i,:,0,0], m_ff[:,0], k_ff[i,:,0,0])

# %%

k_err = k - k_ff
mean_err = jnp.mean(k_err)
max_err = jnp.max(jnp.abs(k_err))

print(mean_err, max_err)
# %%
kernel = TFPKernel(tfk.MaternThreeHalves, 1, 1)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(log_length_scale=jnp.log(3.0))
x = jnp.linspace(-5, 5, 101)[:, np.newaxis]
num_samples = 20000
num_basis_functions = 10000
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)

k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[0,:,0,0], m_ff[:,0], k_ff[0,:,0,0])
# plt.title(f"Square exponential, samples: {num_samples}, basis functions: {num_basis_functions}")
# %%
i = 50
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[i,:,0,0], m_ff[:,0], k_ff[i,:,0,0])

# %%
kernel = ScaledKernel(TFPKernel(tfk.MaternFiveHalves, 1, 1))
kernel_params = kernel.init_params(next(rng))
x = jnp.linspace(-5, 5, 101)[:, np.newaxis]
num_samples = 20000
num_basis_functions = 10000
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)

k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[0,:,0,0], m_ff[:,0], k_ff[0,:,0,0])
# plt.title(f"Square exponential, samples: {num_samples}, basis functions: {num_basis_functions}")
# %%
i = 50
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[i,:,0,0], m_ff[:,0], k_ff[i,:,0,0])

# %%

k_err = k - k_ff
mean_err = jnp.mean(k_err)
max_err = jnp.max(jnp.abs(k_err))

print(mean_err, max_err)
# %%

kernel = SquaredExponentialCompactRiemannianManifoldKernel(S1(0.5), truncation = 500)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(log_length_scale=jnp.log(0.1))
x = jnp.linspace(0, 2*2*jnp.pi*kernel.manifold.radius, 101)[:, np.newaxis]
num_samples = 20000
num_basis_functions = 10000
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)


k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[0,:,0,0], m_ff[:,0], k_ff[0,:,0,0])
plt.title(f"Circle Square exponential, samples: {num_samples}, basis functions: {num_basis_functions}")
# %%
nu=1.5
kernel = MaternCompactRiemannianManifoldKernel(nu, S1(0.5), truncation = 500)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(log_length_scale=jnp.log(0.2))
x = jnp.linspace(0, 2*2*jnp.pi*kernel.manifold.radius, 101)[:, np.newaxis]
num_samples = 20000
num_basis_functions = 10000
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, x)


k = kernel.matrix(kernel_params, x, x)
m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)
plot_kernel_approx(x[:,0], jnp.zeros_like(x[:,0]), k[0,:,0,0], m_ff[:,0], k_ff[0,:,0])
plt.title(f"Circle Matern {nu}, samples: {num_samples}, basis functions: {num_basis_functions}")

# %%

for nu in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    kernel = MaternCompactRiemannianManifoldKernel(nu, S1(0.5), truncation = 500)
    kernel_params = kernel.init_params(next(rng))
    kernel_params = kernel_params._replace(log_length_scale=jnp.log(0.3))
    k = kernel.matrix(kernel_params, x, x)
    plt.plot(x[:,0], k[0,:,0,0]/ k[0,0,0,0])
    # plt.plot(x[:,0], k[0,:,0,0])

# %%
s1 = EmbeddedS1(0.5) # produce base manifold
num_basis_functions = 1000
num_samples = 2000
man = EmbeddedProductManifold(s1,s1, num_eigenfunctions=num_basis_functions)

n_points = 25
m = np.linspace(0, jnp.pi*2, n_points+1)[:-1] % (2 * jnp.pi)
m = jnp.meshgrid(m,m)
m = jnp.stack([m_.flatten() for m_ in m], axis=-1)

kernel = ScaledKernel(ManifoldProjectionVectorKernel(SquaredExponentialCompactRiemannianManifoldKernel(man, num_basis_functions), man))
# kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(2.5,man, num_basis_functions))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m,m)[0,0,0,0]))
k = kernel.matrix(kernel_params, m, m)
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, m)

m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)

# %%
i = int(n_points**2 / 2)

fig, axs = plt.subplots(2,2, figsize=(8,8))

axs[0,0].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    m_ff[:,0].reshape((n_points, n_points)),
    50,
    vmin=0,
    vmax=1,
)
axs[0,0].set_aspect('equal')
axs[0,0].set_title("Fourier featuer mean")

axs[0,1].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    jnp.abs(k[i,:,0,0] - k_ff[i,:,0,0]).reshape((n_points, n_points)),
    20,
    vmin=0,
    vmax=1,
)
axs[0,1].set_aspect('equal')
axs[0,1].set_title("Kernel error")



axs[1,0].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    k_ff[i,:,0,0].reshape((n_points, n_points)),
    20,
    vmin=0,
    vmax=1,
)
axs[1,0].set_aspect('equal')
axs[1,0].set_title("Fourier feature")

cf = axs[1,1].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    k[i,:,0,0].reshape((n_points, n_points)),
    20,
    vmin=0,
    vmax=1,
)
axs[1,1].set_aspect('equal')
axs[1,1].set_title("Computed")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cf, cax=cbar_ax)
# %%
s1 = EmbeddedS1(0.5) # produce base manifold
num_basis_functions = 1000
num_samples = 2000
man = EmbeddedProductManifold(s1,s1, num_eigenfunctions=num_basis_functions)

n_points = 25
m = np.linspace(0, jnp.pi*2, n_points+1, dtype=jnp.float64)[:-1] % (2 * jnp.pi)
m = jnp.meshgrid(m,m)
m = jnp.stack([m_.flatten() for m_ in m], axis=-1)

kernel = ScaledKernel(ManifoldProjectionVectorKernel(MaternCompactRiemannianManifoldKernel(0.5,man, num_basis_functions), man))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m,m)[0,0,0,0]))
k = kernel.matrix(kernel_params, m, m)
key = next(rng)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, key)
f = ff(kernel_params, state, m)

m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)

# %%

scale=20

i = int(n_points**2 / 2)

fig, axs = plt.subplots(2,2, figsize=(8,8))

axs[0,0].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    m_ff[:,0].reshape((n_points, n_points)),
    50,
    vmin=0,
    vmax=1,
)
axs[0,0].set_aspect('equal')
axs[0,0].set_title("Fourier featuer mean")

axs[0,1].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    jnp.abs(k[i,:,0,0] - k_ff[i,:,0,0]).reshape((n_points, n_points)),
    20,
    vmin=0,
    vmax=1,
)
axs[0,1].set_aspect('equal')
axs[0,1].set_title("Kernel error")



axs[1,0].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    k_ff[i,:,0,0].reshape((n_points, n_points)),
    20,
    vmin=0,
    vmax=1,
)
axs[1,0].set_aspect('equal')
axs[1,0].set_title("Fourier feature")

cf = axs[1,1].contourf(
    m[:,0].reshape((n_points, n_points)), 
    m[:,1].reshape((n_points, n_points)), 
    k[i,:,0,0].reshape((n_points, n_points)),
    20,
    vmin=0,
    vmax=1,
)
axs[1,1].set_aspect('equal')
axs[1,1].set_title("Computed")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cf, cax=cbar_ax)
# %%
i=110
scale=50
plt.quiver(
    m[:,0],
    m[:,1],
    f[i,:,0],
    f[i,:,1],
    color='blue',
    scale=scale
)
plt.gca().set_aspect('equal')
# %%

num_basis_functions = 1000
num_samples = 2000

n_points = 50
r = jnp.linspace(-5, 5, n_points)
m = jnp.linspace(0, 2 * jnp.pi, n_points + 1)[:-1]
r, m = jnp.meshgrid(r, m)
m = jnp.stack([m.flatten(), r.flatten()], axis=-1)

s1 = S1(1.0)
k_s1 = MaternCompactRiemannianManifoldKernel(1.5, s1, 500)
k_s1_params = k_s1.init_params(next(rng))
k_s1_params = k_s1_params._replace(log_length_scale=jnp.log(0.3))

k_r1 = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
k_r1_params = k_r1.init_params(next(rng))

kernel = ProductKernel(k_s1, k_r1)
product_kernel_params = kernel.init_params(next(rng))
product_kernel_params = product_kernel_params._replace(sub_kernel_params=[k_s1_params, k_r1_params])

kernel = ScaledKernel(kernel)
kernel_params = kernel.init_params(next(rng))
kernel_params = kernel_params._replace(sub_kernel_params=product_kernel_params)
kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m,m)[0,0,0,0]))
k = kernel.matrix(kernel_params, m, m)

ff = FourierFeatures(kernel, num_basis_functions)
state = ff.init_state(kernel_params, num_samples, next(rng))
f = ff(kernel_params, state, m)

m_ff = jnp.mean(f,axis=0)
k_ff = jnp.mean(f[..., :, np.newaxis, :, np.newaxis] * f[..., np.newaxis, :, np.newaxis, :], axis=0)
# %%
k_err = k - k_ff
mean_err = jnp.mean(k_err)
max_err = jnp.max(jnp.abs(k_err))

print(mean_err, max_err)
# %%

print(eval_kernel(kernel, kernel_params, m, next(rng)))
# %%
