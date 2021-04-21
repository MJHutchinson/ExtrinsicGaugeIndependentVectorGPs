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


class BasisFunctionState(NamedTuple):
    pass


class EigenBasisFunctionState(NamedTuple):
    eigen_index: jnp.ndarray


class RandomBasisFunctionState(NamedTuple):
    frequency: jnp.ndarray
    phase: jnp.ndarray


class AbstractKernel(ABC):
    @abstractmethod
    def init_params(
        self,
        key: jnp.ndarray,
    ) -> NamedTuple:
        pass

    @abstractmethod
    def matrix(
        self,
        params: NamedTuple,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        pass

    @abstractmethod
    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        pass


class FourierFeatureState(NamedTuple):
    basis_function_state: NamedTuple
    weights: jnp.ndarray


class FourierFeatures:
    def __init__(
        self,
        kernel: AbstractKernel,
        num_basis_functions: int,
    ):
        self.kernel = kernel
        self.num_basis_functions = num_basis_functions

    def init_state(
        self,
        params: NamedTuple,
        num_samples: int,
        key: jnp.ndarray,
    ) -> NamedTuple:
        basis_function_state = self.kernel.sample_fourier_features(
            params, key, self.num_basis_functions
        )
        weight_variance = self.kernel.weight_variance(params, basis_function_state)
        weights = weight_variance * jr.normal(
            key, (num_samples, *weight_variance.shape)
        )
        return FourierFeatureState(basis_function_state, weights)

    def resample_weights(
        self,
        params: NamedTuple,
        state: NamedTuple,
        num_samples: int,
        key: jnp.ndarray,
    ) -> NamedTuple:
        weight_variance = self.kernel.weight_variance(
            params, state.basis_function_state
        )
        weights = weight_variance * jr.normal(
            key, (num_samples, *weight_variance.shape)
        )
        return FourierFeatureState(state.basis_function_state, weights)

    def resample_basis(
        self,
        params: NamedTuple,
        state: NamedTuple,
        key: jnp.ndarray,
    ) -> NamedTuple:
        basis_function_state = self.kernel.sample_fourier_features(
            params, key, self.num_basis_functions
        )
        return FourierFeatureState(basis_function_state, state.weights)

    def __call__(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        basis_functions = self.kernel.basis_functions(
            params,
            state.basis_function_state,
            x,
        )
        basis_functions = basis_functions[np.newaxis, ...]
        weights = state.weights[..., np.newaxis, :]
        return tf2jax.linalg.matvec(basis_functions, weights)


class ScaledKernelParameters(NamedTuple):
    log_amplitudes: jnp.ndarray
    log_length_scales: jnp.ndarray


class ScaledTFPKernel(AbstractKernel):
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
    ) -> ScaledKernelParameters:
        log_amplitudes = jnp.zeros((self.output_dimension))
        log_length_scales = jnp.zeros((self.input_dimension))
        return ScaledKernelParameters(log_amplitudes, log_length_scales)

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: ScaledKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """Assemble the kernel matrix.

        Args:
            x1: the first input.
            x2: the second input.
        """
        amplitudes = jnp.exp(params.log_amplitudes)
        length_scales = jnp.exp(params.log_length_scales)
        tfp_kernel = self.tfp_class(amplitude=amplitudes, length_scale=length_scales)
        return tfp_kernel.matrix(x1, x2)

    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        (k1, k2) = jr.split(key)
        frequency = jr.normal(
            k1, (self.output_dimension, self.input_dimension, num_samples)
        )
        phase = 2 * jnp.pi * jr.uniform(k2, (self.output_dimension, num_samples))
        return RandomBasisFunctionState(frequency, phase)

    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:
        L = state.frequency.shape[-1]
        return jnp.ones((L))

    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        amplitudes = jnp.exp(params.log_amplitudes)
        length_scales = jnp.exp(params.log_length_scales)

        frequency = state.frequency
        phase = state.phase

        L = frequency.shape[-1]

        rescaled_x = x / length_scales
        basis_fn = jnp.sqrt(2 / L) * jnp.cos(
            rescaled_x @ frequency + jnp.expand_dims(phase, -2)
        )
        return basis_fn