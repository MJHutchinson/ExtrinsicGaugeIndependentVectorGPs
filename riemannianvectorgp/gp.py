from typing import NamedTuple, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from functools import partial
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from abc import ABC, abstractmethod
from .kernel import AbstractKernel, FourierFeatures


class GaussianProcessParameters(NamedTuple):
    kernel_params: NamedTuple


class GaussianProcessState(NamedTuple):
    locations: jnp.ndarray
    values: jnp.ndarray
    noises: jnp.ndarray
    K_inv: jnp.ndarray


class GaussianProcess:
    """A Gaussian process, implemented as a Haiku module."""

    def __init__(
        self,
        kernel: AbstractKernel,
        input_dimension: int,
        output_dimension: int,
        num_samples: int,
    ):
        """Initializes the GP.

        Args:
            kernel: the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
            num_samples: the number of samples stored in the GP.
        """
        self.kernel = kernel
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_samples = num_samples

    def init_params_with_state(
        self,
        key: jnp.ndarray,
    ) -> Tuple[GaussianProcessParameters, GaussianProcessState]:
        (k1, k2) = jr.split(key)
        kernel_params = self.kernel.init_params(k1)

        (S, OD, ID) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
        )
        params = GaussianProcessParameters(
            kernel_params,
        )

        locations = jnp.array([])
        values = jnp.array([])
        noises = jnp.array([])
        K_inv = jnp.array([])
        state = GaussianProcessState(locations, values, noises, K_inv)

        return (params, state)

    @partial(jax.jit, static_argnums=(0,))
    def condition(
        self,
        params: GaussianProcessParameters,
        locations: jnp.ndarray,
        values: jnp.ndarray,
        noises: jnp.ndarray,
        key: jnp.ndarray,
    ):
        (S, OD, ID, N) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            locations.shape[0],
        )
        K = self.kernel.matrix(params.kernel_params, locations, locations) + jnp.diag(
            noises
        )
        K_inv = tf2jax.linalg.inv(K)
        return GaussianProcessState(locations, values, noises, K_inv)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: GaussianProcessParameters,
        state: GaussianProcessState,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (S, OD, ID, M) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            x.shape[0],
        )
        K_inv = state.K_inv
        kernel_params = params.kernel_params

        K_sn = self.kernel.matrix(kernel_params, x, state.locations)
        K_ns = self.kernel.matrix(kernel_params, state.locations, x)
        K_ss = self.kernel.matrix(kernel_params, x, x)

        m = tf2jax.linalg.matvec(K_sn, tf2jax.linalg.matvec(K_inv, state.values))
        K = K_ss - tf2jax.linalg.matmul(tf2jax.linalg.matmul(K_sn, K_inv), K_ns)

        return m, K

    def sample(
        self,
        params: GaussianProcessParameters,
        state: GaussianProcessState,
        x: jnp.ndarray,
        key: jnp.ndarray,
        obs_noise: float = 1e-6,
    ) -> jnp.ndarray:
        m, K = self(params, state, x)
        cholesky = jsp.linalg.cho_factor(
            K + jnp.identity(K.shape[-1]) * obs_noise, lower=True
        )[0]
        (S, N) = self.num_samples, x.shape[0]
        sample_noise = jr.normal(key, (S, N))
        return m + tf2jax.linalg.matvec(cholesky, sample_noise)[:, np.newaxis, :]

    # @partial(jax.jit, static_argnums=(0,))
    # def randomize(
    #     self,
    #     params: GaussianProcessParameters,
    #     state: GaussianProcessState,
    #     key: jnp.ndarray,
    # ) -> GaussianProcessState:
    #     """Samples a new set of random functions from the GP."""
    #     (S, N) = (
    #         self.num_samples,
    #         state.locations.shape[0],
    #     )
    #     sample_noise = jr.normal(key, (S, N))
    #     state = state._replace(sample_noise=sample_noise)
    #     return state

    # @partial(jax.jit, static_argnums=(0,))
    # def prior_kl(
    #     self,
    #     params: SparseGaussianProcessParameters,
    #     state: SparseGaussianProcessState,
    # ) -> jnp.ndarray:
    #     """Evaluates the prior KL term in the sparse VI objective."""
    #     (OD, ID, M) = (self.output_dimension, self.input_dimension, self.num_inducing)
    #     inducing_locations = params.inducing_locations
    #     inducing_pseudo_mean = params.inducing_pseudo_mean
    #     inducing_pseudo_log_err_stddev = params.inducing_pseudo_log_err_stddev
    #     kernel_params = params.kernel_params
    #     cholesky = state.cholesky

    #     logdet_term = 2 * jnp.sum(jnp.log(jax.vmap(jnp.diag)(cholesky))) - 2 * jnp.sum(
    #         inducing_pseudo_log_err_stddev
    #     )
    #     kernel_matrix = self.kernel.matrix(
    #         kernel_params, inducing_locations, inducing_locations
    #     )
    #     cholesky_inv = tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(
    #         tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(jnp.eye(M)),
    #         adjoint=True,
    #     )
    #     trace_term = jnp.sum(cholesky_inv * kernel_matrix)
    #     reparameterized_quadratic_form_term = jnp.sum(
    #         inducing_pseudo_mean
    #         * tf2jax.linalg.matvec(kernel_matrix, inducing_pseudo_mean)
    #     )
    #     return (
    #         logdet_term
    #         - (OD * ID * M)
    #         + trace_term
    #         + reparameterized_quadratic_form_term
    #     ) / 2

    # @partial(jax.jit, static_argnums=(0,))
    # def hyperprior(
    #     self,
    #     params: SparseGaussianProcessParameters,
    #     state: SparseGaussianProcessState,
    # ) -> jnp.ndarray:
    #     """Returns the log hyperprior regularization term of the GP."""
    #     return jnp.zeros(())  # temporary

    # @partial(jax.jit, static_argnums=(0,))
    # def loss(
    #     self,
    #     params: SparseGaussianProcessParameters,
    #     state: SparseGaussianProcessState,
    #     key: jnp.ndarray,
    #     x: jnp.ndarray,
    #     y: jnp.ndarray,
    #     n_data: int,
    # ) -> Tuple[jnp.ndarray, SparseGaussianProcessState]:
    #     state = self.randomize(params, state, key)

    #     kl = self.prior_kl(params, state)

    #     f = self(params, state, x)
    #     s = jnp.exp(params.log_error_stddev)
    #     (n_samples, _, n_batch) = f.shape
    #     c = n_data / (n_batch * n_samples * 2)
    #     l = n_data * jnp.sum(jnp.log(s)) + c * jnp.sum(((y - f) / s) ** 2)

    #     r = self.hyperprior(params, state)
    #     return (kl + l + r, state)
