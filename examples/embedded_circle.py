# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %load_ext autoreload
# %autoreload 2
# %%
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from tensorflow_probability.python.internal.backend import jax as tf2jax

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
from jax.config import config

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

# %%
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

m = jnp.linspace(0, jnp.pi * 2, 26)[:-1, np.newaxis] % (2 * jnp.pi)
v = jnp.ones_like(m)
x, y = s1.project_to_e(m, v)

# %%

proj_mat = s1.projection_matrix(m)
Pi = s1.euclidean_projection_matrix(m)
# %%
plt.plot(x[:, 0], x[:, 1])
plt.quiver(x[:, 0], x[:, 1], proj_mat[:, 0, 0], proj_mat[:, 1, 0])
# plt.quiver(x[:,0], x[:, 1], Pi[:,1, 0], Pi[:, 1, 1])

plt.gca().set_aspect("equal")
# %%
plt.plot(x[:, 0], x[:, 1])
plt.quiver(x[:, 0], x[:, 1], Pi[:, 0, 0], Pi[:, 0, 1])
plt.quiver(x[:, 0], x[:, 1], Pi[:, 1, 0], Pi[:, 1, 1])

plt.gca().set_aspect("equal")
# %%
plt.plot(x[:, 0], x[:, 1])
plt.quiver(x[:, 0], x[:, 1], y[:, 0], y[:, 1])
plt.gca().set_aspect("equal")

# %%
y = jr.normal(next(rng), (x.shape))
scale = 10
plt.plot(x[:, 0], x[:, 1], zorder=1, color="black")
plt.quiver(x[:, 0], x[:, 1], y[:, 0], y[:, 1], scale=scale, zorder=2, color="red")
plt.gca().set_aspect("equal")

proj_mat = s1.projection_matrix(m)
v_ = jnp.einsum("...em,...e->...m", proj_mat, y)
y_ = jnp.einsum("...em,...m->...e", proj_mat, v_)

# eig_val, eig_vec = jnp.linalg.eig(Pi)

# idx = eig_val.argsort()
# eig_val = np.take_along_axis(eig_val, idx, axis=1)
# eig_vec = np.take_along_axis(eig_vec, idx[..., np.newaxis], axis=1)

# y_ = s1.flatten_to_manifold(x, y)
plt.quiver(x[:, 0], x[:, 1], y_[:, 0], y_[:, 1], scale=scale, zorder=3, color="blue")
# plt.quiver(x[:,0], x[:, 1], eig_vec[:,0,0], eig_vec[:,1,0], zorder=3, color='blue')
# plt.quiver(x[:,0], x[:, 1], eig_vec[:,0,1], eig_vec[:,1,1], zorder=3, color='blue')

m_, v_ = s1.project_to_m(x, y)
_, y_ = s1.project_to_e(m_, v_)
plt.quiver(x[:, 0], x[:, 1], y_[:, 0], y_[:, 1], scale=scale, zorder=3, color="orange")

y_ = s1.flatten_to_manifold(x, y)
plt.quiver(x[:, 0], x[:, 1], y_[:, 0], y_[:, 1], scale=scale, zorder=3, color="green")

# y_ = jnp.einsum("...em,...fm,...f->...e", proj_mat, proj_mat, v_)
# plt.quiver(x[:,0], x[:, 1], y_[:,0], y_[:, 1], scale=scale, zorder=3, color='purple')

flatten_mat1 = tf2jax.linalg.matmul(proj_mat, jnp.swapaxes(proj_mat, -1, -2))
y_ = tf2jax.linalg.matvec(flatten_mat1, y)
plt.quiver(x[:, 0], x[:, 1], y_[:, 0], y_[:, 1], scale=scale, zorder=3, color="pink")

flatten_mat2 = jnp.einsum("...em,...fn->...ef", proj_mat, proj_mat)
y_ = jnp.einsum("...ef,...f->...e", flatten_mat2, y)
plt.quiver(x[:, 0], x[:, 1], y_[:, 0], y_[:, 1], scale=scale, zorder=3, color="yellow")
# %%

tp = s1.tanget_projection(m, m)

plt.plot(x[:, 0], x[:, 1])
plt.plot(x[:, 0] * (1 + tp[0, :, 0, 0]), x[:, 1] * (1 + tp[0, :, 0, 0]))
plt.gca().set_aspect("equal")
plt.scatter(x[0, 0], x[0, 1])

p1 = s1.projection_matrix(m)
p2 = s1.projection_matrix(m)
tp = jnp.einsum("...iem,ef,...jfn->...jinm", p1, jnp.eye(2), p2)
plt.plot(x[:, 0] * (1 + tp[0, :, 0, 0]), x[:, 1] * (1 + tp[0, :, 0, 0]))

tp = jnp.einsum("...iem,...jfn->...jinm", p1, p2)
plt.plot(x[:, 0] * (1 + tp[0, :, 0, 0]), x[:, 1] * (1 + tp[0, :, 0, 0]))
# %%

m_, v_ = s1.project_to_m(x, y)

print(jnp.mean(m - m_), jnp.mean(v - v_))
# %%

M = jnp.concatenate([m, m], axis=-1)
jnp.split(M, [1], axis=-1)
# %%

s_2 = EmbeddedProductManifold(EmbeddedS1(0.5), EmbeddedS1(0.5))


# %%
m = jnp.linspace(0, jnp.pi * 2, 30) % (2 * jnp.pi)
m = jnp.meshgrid(m, m)
m = jnp.stack([m_.flatten() for m_ in m], axis=-1)
v = jnp.ones_like(m)

x, y = s_2.project_to_e(m, v)
plt.plot(x[:, 0], x[:, 1], zorder=0)
plt.quiver(x[:, 0], x[:, 1], y[:, 0], y[:, 1], zorder=1)
plt.gca().set_aspect("equal")
# %%
s_2.projection_matrix(m).shape

# %%

s1 = S1(0.5)
# kernel = ScaledKernel(SquaredExponentialCompactRiemannianManifoldKernel(s1, 100))
kernel = ScaledKernel(MaternCompactRiemannianManifoldKernel(0.5, s1, 100))
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0])
)
k = kernel.matrix(kernel_params, m, m)
plt.plot(m[:, 0], k[0, :, 0, 0], label="EQ")

# %%
i = 20
plt.plot(x[:, 0], x[:, 1])
plt.scatter(x[i, 0], x[i, 1])
plt.plot(x[:, 0] * (1 + k[i, :, 0, 0]), x[:, 1] * (1 + k[i, :, 0, 0]))
plt.gca().set_aspect("equal")

# %%

s1 = EmbeddedS1(0.5)
# kernel = ScaledKernel(SquaredExponentialCompactRiemannianManifoldKernel(s1, 100))
kernel = ScaledKernel(
    ManifoldProjectionVectorKernel(
        MaternCompactRiemannianManifoldKernel(0.5, s1, 100), s1
    )
)
kernel_params = kernel.init_params(next(rng))
sub_kernel_params = kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.1))
kernel_params = kernel_params._replace(sub_kernel_params=sub_kernel_params)
kernel_params = kernel_params._replace(
    log_amplitude=-jnp.log(kernel.matrix(kernel_params, m, m)[0, 0, 0, 0])
)
k = kernel.matrix(kernel_params, m, m)
plt.plot(m[:, 0], k[0, :, 0, 0], label="EQ")
# %%
plt.plot(x[:, 0], x[:, 1])
plt.plot(x[:, 0] * (1 + k[0, :, 0, 0]), x[:, 1] * (1 + k[0, :, 0, 0]))
plt.gca().set_aspect("equal")
# %%
_, k_ = s1.project_to_e(m, k[0, :, :, 0])
plt.plot(x[:, 0], x[:, 1], zorder=0)
plt.quiver(x[:, 0], x[:, 1], k_[:, 0], k_[:, 1], zorder=1)
plt.gca().set_aspect("equal")
# %%
