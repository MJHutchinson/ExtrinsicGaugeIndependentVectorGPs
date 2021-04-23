from typing import NamedTuple, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial
import tensorflow_probability

from .kernel import AbstractKernel
from .utils import pairwise_dimension_distance

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels


class CircularSquaredExponentialKernelParameters(NamedTuple):
    log_amplitude: jnp.ndarray
    log_length_scale: jnp.ndarray


class CircularSquaredExponentialKernel(AbstractKernel):
    def __init__(
        self,
        dim: int,
        truncation_level: int,
        reference_length_scale: int = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.truncation_level = truncation_level
        self.reference_length_scale = reference_length_scale

    def init_params(
        self, key: jnp.ndarray
    ) -> CircularSquaredExponentialKernelParameters:
        log_amplitude = jnp.zeros((1))
        log_length_scale = jnp.zeros((1))
        return CircularSquaredExponentialKernelParameters(
            log_amplitude, log_length_scale
        )

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: CircularSquaredExponentialKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        n, d = x1.shape
        m, d = x2.shape
        print(
            jnp.arange(-self.truncation_level, self.truncation_level + 1, step=1).shape
        )
        loop = (
            2
            * jnp.pi
            * jnp.arange(-self.truncation_level, self.truncation_level + 1, step=1)
        )
        loop = self.dim * [loop]
        loop = jnp.stack(jnp.meshgrid(*loop), axis=-1)
        loop = jnp.reshape(loop, (1, 1, -1, d))
        dist = pairwise_dimension_distance(x1, x2)
        dist = jnp.reshape(dist, (n, m, 1, d))
        dist = (dist + loop) / jnp.reshape(
            jnp.exp(params.log_length_scale), (1, 1, 1, -1)
        )
        dist = jnp.power(dist, 2) / 2
        kernel = jnp.exp(params.log_amplitude) * jnp.sum(
            jnp.exp(-jnp.sum(dist, axis=-1)), axis=-1
        )

        return kernel

    def kernel(
        self,
        params: NamedTuple,
    ):
        pass

    @partial(jit, static_argnums=(0, 2))
    def standard_spectral_measure(
        self, key: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=(0,))
    def spectral_weights(
        self,
        params: CircularSquaredExponentialKernelParameters,
        frequency: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass