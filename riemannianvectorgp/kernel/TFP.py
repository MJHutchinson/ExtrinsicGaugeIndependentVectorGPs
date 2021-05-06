from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

from .kernel import AbstractKernel, RandomBasisFunctionState


class TFPKernelParameters(NamedTuple):
    log_length_scales: jnp.ndarray


class TFPKernel(AbstractKernel):
    """A kernel with learned amplitude and length scale parameters."""

    def __init__(
        self,
        tfp_class: ABCMeta,
        input_dimension: int,
        output_dimension: int,
    ):
        """Scales the given kernel input by length scales and output by amplitudes.

        Args:
            tfp_class: the TensorFlow Probability class of the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
        """
        super().__init__()
        self.tfp_class = tfp_class
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def init_params(
        self,
        key: jnp.ndarray,
    ) -> TFPKernelParameters:
        if self.input_dimension == 1:
            log_length_scales = jnp.zeros(())
        else:
            log_length_scales = jnp.zeros((self.input_dimension))
        return TFPKernelParameters(log_length_scales)

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: TFPKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """Assemble the kernel matrix.

        Args:
            x1: the first input.
            x2: the second input.
        """
        length_scales = jnp.exp(params.log_length_scales)
        tfp_kernel = self.tfp_class(amplitude=None, length_scale=length_scales)
        return tfp_kernel.matrix(x1, x2)[..., np.newaxis, np.newaxis]

    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        (k1, k2) = jr.split(key)
        if self.tfp_class == tfk.ExponentiatedQuadratic:
            frequency = jr.normal(
                k1,
                (
                    num_samples,
                    self.output_dimension,
                    self.input_dimension,
                ),
            )
            phase = (
                2
                * jnp.pi
                * jr.uniform(
                    k2,
                    (
                        num_samples,
                        self.output_dimension,
                    ),
                )
            )
        else:
            raise NotImplementedError(
                f"Fourier features not implemented for {self.tfp_class}"
            )
        return RandomBasisFunctionState(frequency, phase)

    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:
        L = state.frequency.shape[-3]
        return jnp.ones((L))

    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        length_scales = jnp.exp(params.log_length_scales)

        frequency = state.frequency
        phase = state.phase

        L = frequency.shape[0]

        # print(frequency.shape)
        # print(phase.shape)
        # print(x.shape)

        rescaled_x = x / length_scales
        basis_fn = jnp.sqrt(2 / L) * jnp.cos(
            jnp.einsum("ni,boi->nbo", rescaled_x, frequency) + phase
        )
        # print(frequency.shape)
        # print(phase.shape)
        # print(basis_fn.shape)
        return basis_fn
