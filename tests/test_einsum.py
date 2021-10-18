# %%
from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial
import tensorflow_probability
import timeit
from tensorflow_probability.python.internal.backend import jax as tf2jax
from einops import rearrange

# %%

S = 10
PD = 3
OD = 2
N1 = 13
N2 = 15
M = 17

# %%

# %%
@jit
def K_naive(p1, K, p2):
    return jnp.swapaxes(p1, -1, -2)[..., :, np.newaxis, :, :] @ Ke @ p2[..., np.newaxis, :, :, :]

@jit
def K_einsum(p1, K, p2):
    return jnp.einsum("...neo,...nmef,...mfp->...mnop", p1, Ke, p2)

# @jit
# def K_einops(p1, K, p2):
#     return jnp.einsum("...neo,...nmef,...mfp->...mnop", p1, Ke, p2)


Ke = jnp.ones((N1, N2, PD, PD))
p1 = jnp.ones((N1, PD, OD))
p2 = jnp.ones((N2, PD, OD))
print(K_naive(p1, Ke, p2).shape)
print(K_einsum(p1, Ke, p2).shape)
# %%
Ke = jnp.ones((N1, N2, PD, PD))
p1 = jnp.ones((N1, PD, OD))
p2 = jnp.ones((N2, PD, OD))
p1t = jnp.swapaxes(p1, -1, -2)

print("Full kernel")
%timeit K_naive(p1, Ke, p2)
%timeit K_einsum(p1, Ke, p2)
# %%
Ke = jnp.ones((N1, N2, 1, 1))
p1 = jnp.ones((N1, PD, OD))
p2 = jnp.ones((N2, PD, OD))
p1t = jnp.swapaxes(p1, -1, -2)

print("scalar kernel")
try:
    %timeit K_naive(p1, Ke, p2)
except:
    print("failed to run - can't auto broadcast along 1 dim")
%timeit K_einsum(p1, Ke, p2)
# %%
Ke = jnp.ones((S, N1, N2, PD, PD))
p1 = jnp.ones((S, N1, PD, OD))
p2 = jnp.ones((S, N2, PD, OD))
p1t = jnp.swapaxes(p1, -1, -2)

print("batched full")
%timeit K_naive(p1, Ke, p2)
%timeit K_einsum(p1, Ke, p2)
# %%
Ke = jnp.ones((S, N1, N2, 1, 1))
p1 = jnp.ones((S, N1, PD, OD))
p2 = jnp.ones((S, N2, PD, OD))
p1t = jnp.swapaxes(p1, -1, -2)

print("batched partial")
try:
    %timeit K_naive(p1, Ke, p2)
except:
    print("failed to run - can't auto broadcast along 1 dim")
%timeit K_einsum(p1, Ke, p2)
# %%
@jax.jit
def ind_naive(K, W):
    n1, n2, d1, d2 = K.shape
    s,_,_ = W.shape
    K = rearrange(K, "n1 n2 d1 d2 -> (n1 d1) (n2 d2)")[np.newaxis, ...]
    W = rearrange(W, "s n2 d2 -> s (n2 d2)")
    out = tf2jax.linalg.matvec(K, W)
    return rearrange(out, "s (n1 d1) -> s n1 d1", n1=n1, d1=d1)

@jax.jit
def ind_einsum(K, W):
    return jnp.einsum("...mnop,...snp->...smo", K, W)

K = jnp.zeros((11, M, M, OD, OD))
inducing_weights = jnp.zeros((11, S, M, OD))


# print(ind_naive(K, inducing_weights).shape)
print(ind_einsum(K, inducing_weights).shape)
# %%
%timeit ind_naive(K, inducing_weights)
%timeit ind_einsum(K, inducing_weights)
# %%
