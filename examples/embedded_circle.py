# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
%load_ext autoreload
%autoreload 2
# %%
# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optax
from tensorflow_probability.python.internal.backend import jax as tf2jax
from einops import rearrange

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
import sys

from riemannianvectorgp.manifold import (
    S1,
    EmbeddedS1,
    ProductManifold,
    EmbeddedProductManifold,
)
from riemannianvectorgp.kernel import (
    ScaledKernel,
    ManifoldProjectionVectorKernel,
    MaternCompactRiemannianManifoldKernel,
    SquaredExponentialCompactRiemannianManifoldKernel,
)
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from jax.config import config

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

# %%

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


rng = GlobalRNG()
# %%

s1 = EmbeddedS1(0.5)
n_points=100
m = jnp.linspace(0, jnp.pi*2, n_points+1)[:-1, np.newaxis] % (2 * jnp.pi)
v = 2 * jnp.sin(m) + jr.normal(next(rng), m.shape) / 10
x, y = s1.project_to_e(m, v)

# %%

proj_mat = s1.projection_matrix(m)
Pi = s1.euclidean_projection_matrix(m)
# %%
plt.plot(x[:,0], x[:, 1])
plt.quiver(x[:,0], x[:, 1], proj_mat[:,0, 0], proj_mat[:, 1, 0])
# plt.quiver(x[:,0], x[:, 1], Pi[:,1, 0], Pi[:, 1, 1])

plt.gca().set_aspect('equal')
# %%
plt.plot(x[:,0], x[:, 1])
plt.quiver(x[:,0], x[:, 1], Pi[:,0, 0], Pi[:, 0, 1])
plt.quiver(x[:,0], x[:, 1], Pi[:,1, 0], Pi[:, 1, 1])

plt.gca().set_aspect('equal')
# %%
plt.plot(x[:,0], x[:, 1])
plt.quiver(x[:,0], x[:, 1], y[:,0], y[:, 1])
plt.gca().set_aspect('equal')

# %%
y = jr.normal(next(rng), (x.shape))
scale=10
plt.plot(x[:,0], x[:, 1], zorder=1, color='black')
plt.quiver(x[:,0], x[:, 1], y[:,0], y[:, 1], scale=scale, zorder=2, color='red')
plt.gca().set_aspect('equal')

proj_mat = s1.projection_matrix(m)
v_ = jnp.einsum("...em,...e->...m", proj_mat, y)
y_ = jnp.einsum("...em,...m->...e", proj_mat, v_)

# eig_val, eig_vec = jnp.linalg.eig(Pi)

# idx = eig_val.argsort()
# eig_val = np.take_along_axis(eig_val, idx, axis=1)
# eig_vec = np.take_along_axis(eig_vec, idx[..., np.newaxis], axis=1)

# y_ = s1.flatten_to_manifold(x, y)
plt.quiver(x[:,0], x[:, 1], y_[:,0], y_[:, 1], scale=scale, zorder=3, color='blue')
# plt.quiver(x[:,0], x[:, 1], eig_vec[:,0,0], eig_vec[:,1,0], zorder=3, color='blue')
# plt.quiver(x[:,0], x[:, 1], eig_vec[:,0,1], eig_vec[:,1,1], zorder=3, color='blue')

m_, v_ = s1.project_to_m(x, y)
_, y_ = s1.project_to_e(m_, v_)
plt.quiver(x[:,0], x[:, 1], y_[:,0], y_[:, 1], scale=scale, zorder=3, color='orange')

y_ = s1.flatten_to_manifold(x, y)
plt.quiver(x[:,0], x[:, 1], y_[:,0], y_[:, 1], scale=scale, zorder=3, color='green')

# y_ = jnp.einsum("...em,...fm,...f->...e", proj_mat, proj_mat, v_)
# plt.quiver(x[:,0], x[:, 1], y_[:,0], y_[:, 1], scale=scale, zorder=3, color='purple')

flatten_mat1 = tf2jax.linalg.matmul(proj_mat, jnp.swapaxes(proj_mat, -1, -2))
y_ = tf2jax.linalg.matvec(flatten_mat1, y)
plt.quiver(x[:,0], x[:, 1], y_[:,0], y_[:, 1], scale=scale, zorder=3, color='pink')

flatten_mat2 = jnp.einsum("...em,...fn->...ef", proj_mat, proj_mat)
y_ = jnp.einsum("...ef,...f->...e", flatten_mat2, y)
plt.quiver(x[:,0], x[:, 1], y_[:,0], y_[:, 1], scale=scale, zorder=3, color='yellow')
# %%

tp = s1.tanget_projection(m, m)

plt.plot(x[:,0], x[:,1])
plt.plot(x[:,0]*(1 + tp[0, :, 0, 0]), x[:,1]*(1 + tp[0, :, 0, 0]))
plt.gca().set_aspect('equal')
plt.scatter(x[0,0], x[0,1])

p1 = s1.projection_matrix(m)
p2 = s1.projection_matrix(m)
tp =  jnp.einsum("...iem,ef,...jfn->...jinm", p1, jnp.eye(2), p2)
plt.plot(x[:,0]*(1 + tp[0, :, 0, 0]), x[:,1]*(1 + tp[0, :, 0, 0]))

tp =  jnp.einsum("...iem,...jfn->...jinm", p1, p2)
plt.plot(x[:,0]*(1 + tp[0, :, 0, 0]), x[:,1]*(1 + tp[0, :, 0, 0]))
# %%


s1 = S1(0.5)
# kernel = ScaledKernel(SquaredExponentialCompactRiemannianManifoldKernel(s1, 100))
kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(0.5,s1, 100))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m,m)[0,0,0]))
k = kernel.matrix(kernel_params, m, m)
plt.plot(m[:, 0], k[0, :, 0, 0], label="EQ")

# %%
i = 20
plt.plot(x[:,0], x[:,1])
plt.scatter(x[i,0], x[i,1])
plt.plot(x[:,0]*(1 + k[i, :, 0, 0]), x[:,1]*(1 + k[i, :, 0, 0]))
plt.gca().set_aspect('equal')

# %%

s1 = EmbeddedS1(0.5)
# kernel = ScaledKernel(SquaredExponentialCompactRiemannianManifoldKernel(s1, 100))
kernel = ScaledKernel(
    ManifoldProjectionVectorKernel(
        MaternCompactRiemannianManifoldKernel(0.5,s1, 100),
        s1
    )
)
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(log_amplitude=-jnp.log(kernel.matrix(kernel_params, m,m)[0,0,0,0]))
k = kernel.matrix(kernel_params, m, m)
plt.plot(m[:, 0], k[0, :, 0, 0], label="EQ")
# %%
plt.plot(x[:,0], x[:,1])
plt.plot(x[:,0]*(1 + k[0, :, 0, 0]), x[:,1]*(1 + k[0, :, 0, 0]))
plt.gca().set_aspect('equal')
# %%
_, k_ = s1.project_to_e(m, k[0, :, :, 0])
plt.plot(x[:, 0], x[:, 1], zorder=0)
plt.quiver(x[:, 0], x[:, 1], k_[:, 0], k_[:, 1], zorder=1)
plt.gca().set_aspect("equal")
# %%

scale = 30 * 2
n_cond = 10
n_ind = 11

sparse_gp = SparseGaussianProcess(kernel, n_ind, 99, 20)
sparse_gp_params, sparse_gp_state = sparse_gp.init_params_with_state(next(rng))
sparse_gp_params = sparse_gp_params._replace(kernel_params=kernel_params)
sparse_gp_state = sparse_gp.randomize(sparse_gp_params, sparse_gp_state, next(rng))


m_ind = jnp.expand_dims(jnp.linspace(0, 2 * jnp.pi, n_ind), -1)
v_ind = 2 * jnp.sin(m_ind) + jr.normal(next(rng), m_ind.shape) / 10
# v_ind = jnp.zeros_like(m_ind)

sparse_gp_params = sparse_gp.set_inducing_points(
    sparse_gp_params,
    m_ind,
    v_ind,
    jnp.ones((n_ind, 1)) * 0.01,
)
# %%
plot_gp(m, v, sparse_gp, sparse_gp_params, sparse_gp_state)
# %%


inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)
inducing_locs_, inducing_means_ = s1.project_to_e(
    sparse_gp_params.inducing_locations, inducing_means
)

plt.plot(x[:, 0], x[:, 1], color="black", zorder=0)
plt.quiver(
    inducing_locs_[:, 0],
    inducing_locs_[:, 1],
    inducing_means_[:, 0],
    inducing_means_[:, 1],
    color="green",
    scale=scale,
    zorder=1,
)

posterior_samples = sparse_gp(sparse_gp_params, sparse_gp_state, m)
for i in range(posterior_samples.shape[0]):
    _, ps_ = s1.project_to_e(m, posterior_samples[i])
    plt.quiver(
        x[:, 0],
        x[:, 1],
        ps_[:, 0],
        ps_[:, 1],
        color="grey",
        scale=scale,
        alpha=0.3,
    )

plt.gca().set_aspect("equal")
# %%
opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), optax.scale(-0.01))
opt_state = opt.init(sparse_gp_params)

# %%
debug_params = [sparse_gp_params]
debug_states = [sparse_gp_state]
debug_keys = [rng.key]

# %%
for i in range(600):
    ((train_loss, sparse_gp_state), grads) = jax.value_and_grad(sparse_gp.loss, has_aux=True)(
        sparse_gp_params, sparse_gp_state, next(rng), m, v, m.shape[0]
    )
    (updates, opt_state) = opt.update(grads, opt_state)
    sparse_gp_params = optax.apply_updates(sparse_gp_params, updates)
    if jnp.all(jnp.isnan(grads.kernel_params.sub_kernel_params.log_length_scale)):
        print("breaking for nan")
        break
    if i <= 10 or i % 20 == 0:
        print(i, "Loss:", train_loss)
    debug_params.append(sparse_gp_params)
    debug_states.append(sparse_gp_state)
    debug_keys.append(rng.key)

# %%

plot_gp(m, v, sparse_gp, sparse_gp_params, sparse_gp_state)
# %%

prior, data = sparse_gp.sample_parts(sparse_gp_params, sparse_gp_state, m)
# %%
i = 0
plt.plot(m[:,0], prior[i, :,0], color='orange')
plt.plot(m[:,0], data[i, :,0], color='purple')
# %%
