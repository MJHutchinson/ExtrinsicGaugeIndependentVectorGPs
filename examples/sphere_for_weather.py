# %%
#get_ipython().run_line_magic("load_ext", "autoreload")
#get_ipython().run_line_magic("autoreload", "2")

import math
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import optax

import jax
from jax import jit
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

import matplotlib.pyplot as plt

#from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    TFPKernel,
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
)
import sys

class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self
    
    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key

rng = GlobalRNG()

import polyscope as ps

# %%
#ps.init()

# %%
S2 = EmbeddedS2(1.0)

num_points = 30
phi = jnp.linspace(0, jnp.pi, num_points)
theta = jnp.linspace(0, 2 * jnp.pi, num_points + 1)[1:]
phi, theta = jnp.meshgrid(phi, theta)
phi = phi.flatten()
theta = theta.flatten()
m = jnp.stack(
    [phi, theta], axis=-1
)  ### NOTE this ordering, I can change it but it'll be a pain, its latitude, longitude
# x = S2.m_to_e(m)

# sphere_mesh = ps.register_surface_mesh(
#     "Sphere",
#     *mesh_to_polyscope(x.reshape((num_points, num_points, 3)), wrap_y=False),
#     color=(0.8, 0.8, 0.8)
# )
# sphere_mesh.set_vertex_tangent_basisX(S2.projection_matrix(m)[..., 0])

# %% ## Euclidean Scalar Kernel ##
es_kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 1, 1))
es_kernel_params = es_kernel.init_params(next(rng))
sub_kernel_params = es_kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scales=jnp.log(1))
es_kernel_params = es_kernel_params._replace(sub_kernel_params=sub_kernel_params)
es_kernel_params = es_kernel_params._replace(
    log_amplitude=-jnp.log(es_kernel.matrix(es_kernel_params, m, m)[0, 0, 0, 0])
)
es_k = es_kernel.matrix(es_kernel_params, m, m)

i = int((num_points ** 2 + num_points) / 2)
plt.contourf(
    m[:, 0].reshape((num_points, num_points)),
    m[:, 1].reshape((num_points, num_points)),
    es_k[i, :, 0, 0].reshape((num_points, num_points)),
    50,
)
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title("Scalar Euclidean EQ kernel")

# %% Manifold Scalar Kernel ##
ms_kernel = ScaledKernel(
    MaternCompactRiemannianManifoldKernel(1.5, S2, 144)
)  # 144 is the maximum number of basis functions we have implemented
ms_kernel_params = ms_kernel.init_params(next(rng))
sub_kernel_params = ms_kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(1))
ms_kernel_params = ms_kernel_params._replace(sub_kernel_params=sub_kernel_params)
ms_kernel_params = ms_kernel_params._replace(
    log_amplitude=-jnp.log(ms_kernel.matrix(ms_kernel_params, m, m)[0, 0, 0, 0])
)
ms_k = ms_kernel.matrix(ms_kernel_params, m, m)

i = int((num_points ** 2 + num_points) / 2)
plt.contourf(
    m[:, 0].reshape((num_points, num_points)),
    m[:, 1].reshape((num_points, num_points)),
    ms_k[i, :, 0, 0].reshape((num_points, num_points)),
    50,
)
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title("Scalar S2 Matern 3/2 kernel")

# %% ## Euclidean Vector Kernel ##
ev_kernel = ScaledKernel(TFPKernel(tfk.ExponentiatedQuadratic, 2, 2))
ev_kernel_params = ev_kernel.init_params(next(rng))
sub_kernel_params = ev_kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scales=jnp.log(0.5))
ev_kernel_params = ev_kernel_params._replace(sub_kernel_params=sub_kernel_params)
ev_kernel_params = ev_kernel_params._replace(
    log_amplitude=-jnp.log(ev_kernel.matrix(ev_kernel_params, m, m)[0, 0, 0, 0])
)
ev_k = ev_kernel.matrix(ev_kernel_params, m, m) * jnp.eye(
    2
)  # Make it the proper kernel

i = int((num_points ** 2 + num_points) / 2)
vec = jnp.array([0, 1])
operator = ev_k[:, i] @ vec
plt.quiver(m[:, 0], m[:, 1], operator[:, 0], operator[:, 1], color="blue")
plt.quiver(m[i, 0], m[i, 1], vec[0], vec[1], color="red")
plt.gca().set_aspect("equal")
# plt.colorbar()
plt.title("Vector Euclidean EQ kernel")

# %% Manifold Vector Kernel ##
mv_kernel = ScaledKernel(
    ManifoldProjectionVectorKernel(
        MaternCompactRiemannianManifoldKernel(1.5, S2, 144), S2
    )
)  # 144 is the maximum number of basis functions we have implemented
mv_kernel_params = mv_kernel.init_params(next(rng))
sub_kernel_params = mv_kernel_params.sub_kernel_params
sub_kernel_params = sub_kernel_params._replace(log_length_scale=jnp.log(0.5))
mv_kernel_params = mv_kernel_params._replace(sub_kernel_params=sub_kernel_params)
mv_kernel_params = mv_kernel_params._replace(
    log_amplitude=-jnp.log(mv_kernel.matrix(mv_kernel_params, m, m)[0, 0, 0, 0])
)
mv_k = mv_kernel.matrix(mv_kernel_params, m, m)

i = int((num_points ** 2 + num_points) / 2)
vec = jnp.array([0, 1])
operator = mv_k[:, i] @ vec
plt.quiver(m[:, 0], m[:, 1], operator[:, 0], operator[:, 1], color="blue")
plt.quiver(m[i, 0], m[i, 1], vec[0], vec[1], color="red")
plt.gca().set_aspect("equal")
# plt.colorbar()
plt.title("Vector S2 Matern 3/2 kernel")

# %% Draw a set to condition on
n_cond = 10

m_cond_ind = jr.permutation(next(rng), jnp.arange(m.shape[0]))[:n_cond]
m_cond = m[m_cond_ind]
v_cond = jr.normal(next(rng), (n_cond, 2))
noises_cond = jnp.ones_like(v_cond) * 0.01

plt.quiver(m_cond[:, 0], m_cond[:, 1], v_cond[:, 0], v_cond[:, 1], color="red")
plt.gca().set_aspect("equal")
plt.xlim(0, jnp.pi)
plt.ylim(0, 2 * jnp.pi)

# %% Euclidean vector GP
ev_gp = GaussianProcess(ev_kernel)
ev_gp_params, ev_gp_state = ev_gp.init_params_with_state(next(rng))
ev_gp_params = ev_gp_params._replace(kernel_params=ev_kernel_params)

ev_gp_state = ev_gp.condition(ev_gp_params, m_cond, v_cond, noises_cond)

scale = 50
fig = plt.figure(figsize=(6, 6))
mean, K = ev_gp(ev_gp_params, ev_gp_state, m)

plt.quiver(
    m_cond[:, 1],
    m_cond[:, 0],
    v_cond[:, 0],
    v_cond[:, 1],
    color="red",
    scale=scale,
    width=0.003,
    headwidth=2,
    zorder=5,
)

plt.quiver(
    m[:, 1],
    m[:, 0],
    mean[:, 0],
    mean[:, 1],
    color="blue",
    scale=scale,
    width=0.003,
    headwidth=2,
    zorder=4
)
plt.gca().set_aspect("equal")
#plt.xlim(0, jnp.pi)
#plt.ylim(0, 2 * jnp.pi)
plt.title("Vector Euclidean EQ kernel - Mean")

plt.savefig("figs/example_plot.png")

sys.exit()

# %%
samples = 10
obs_noise = 1e-3

posterior_samples = ev_gp.sample(
    ev_gp_params, ev_gp_state, m, samples, next(rng), obs_noise=obs_noise
)
scale = 50
fig = plt.figure(figsize=(6, 6))
plt.quiver(
    m_cond[:, 0],
    m_cond[:, 1],
    v_cond[:, 0],
    v_cond[:, 1],
    color="red",
    scale=scale,
    zorder=3,
)

plt.quiver(
    m[:, 0], m[:, 1], mean[:, 0], mean[:, 1], color="blue", scale=scale, zorder=2
)

for i in range(samples):
    plt.quiver(
        m[:, 0],
        m[:, 1],
        posterior_samples[i, :, 0],
        posterior_samples[i, :, 1],
        color="black",
        alpha=0.3,
        scale=scale,
        zorder=1,
    )

plt.gca().set_aspect("equal")
plt.xlim(0, jnp.pi)
plt.ylim(0, 2 * jnp.pi)
plt.title("Vector Euclidean EQ kernel - Samples")

# %% Manifold vector GP
mv_gp = GaussianProcess(mv_kernel)
mv_gp_params, mv_gp_state = mv_gp.init_params_with_state(next(rng))
mv_gp_params = mv_gp_params._replace(kernel_params=mv_kernel_params)

mv_gp_state = mv_gp.condition(mv_gp_params, m_cond, v_cond, noises_cond)

scale = 50
fig = plt.figure(figsize=(6, 6))
mean, K = mv_gp(mv_gp_params, mv_gp_state, m)
plt.quiver(
    m_cond[:, 0],
    m_cond[:, 1],
    v_cond[:, 0],
    v_cond[:, 1],
    color="red",
    scale=scale,
    zorder=5,
)
plt.quiver(
    m[:, 0], m[:, 1], mean[:, 0], mean[:, 1], color="blue", scale=scale, zorder=4
)
plt.gca().set_aspect("equal")
plt.xlim(0, jnp.pi)
plt.ylim(0, 2 * jnp.pi)
plt.title("Vector S2 Matern 3/2 kernel - Mean")
# %%
samples = 10
obs_noise = 1e-3

posterior_samples = mv_gp.sample(
    mv_gp_params, mv_gp_state, m, samples, next(rng), obs_noise=obs_noise
)
scale = 50
fig = plt.figure(figsize=(6, 6))
plt.quiver(
    m_cond[:, 0],
    m_cond[:, 1],
    v_cond[:, 0],
    v_cond[:, 1],
    color="red",
    scale=scale,
    zorder=3,
)

plt.quiver(
    m[:, 0], m[:, 1], mean[:, 0], mean[:, 1], color="blue", scale=scale, zorder=2
)

for i in range(samples):
    plt.quiver(
        m[:, 0],
        m[:, 1],
        posterior_samples[i, :, 0],
        posterior_samples[i, :, 1],
        color="black",
        alpha=0.3,
        scale=scale,
        zorder=1,
    )

plt.gca().set_aspect("equal")
plt.xlim(0, jnp.pi)
plt.ylim(0, 2 * jnp.pi)
plt.title("Vector S2 Matern 3/2 kernel - Samples")
