from typing import NamedTuple, Tuple
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


class SparseGaussianProcessParameters(NamedTuple):
    log_error_stddev: jnp.ndarray
    inducing_locations: jnp.ndarray
    inducing_pseudo_mean: jnp.ndarray
    inducing_pseudo_log_err_stddev: jnp.ndarray
    kernel_params: NamedTuple


class SparseGaussianProcessState(NamedTuple):
    inducing_weights: jnp.ndarray
    cholesky: jnp.ndarray
    prior_state: NamedTuple


class SparseGaussianProcess:
    """A sparse Gaussian process, implemented as a Haiku module."""

    def __init__(
        self,
        kernel: AbstractKernel,
        input_dimension: int,
        output_dimension: int,
        num_inducing: int,
        num_basis: int,
        num_samples: int,
    ):
        """Initializes the sparse GP.

        Args:
            kernel: the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
            num_inducing: the number of inducing points per input dimension.
            num_basis: the number of prior basis functions.
            num_samples: the number of samples stored in the GP.
        """
        self.kernel = kernel
        self.prior = FourierFeatures(kernel, num_basis)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_inducing = num_inducing
        self.num_basis = num_basis
        self.num_samples = num_samples

    def init_params_with_state(
        self,
        key: jnp.ndarray,
    ) -> Tuple[SparseGaussianProcessParameters, SparseGaussianProcessState]:
        (k1, k2, k3) = jr.split(key, 3)
        kernel_params = self.kernel.init_params(k1)

        (S, OD, ID, M, L) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            self.num_inducing,
            self.num_basis,
        )
        log_error_stddev = jnp.zeros((OD))
        inducing_locations = jr.uniform(k2, (M, ID))
        inducing_pseudo_mean = jnp.zeros((OD, M))
        inducing_pseudo_log_err_stddev = jnp.zeros((OD, M))
        params = SparseGaussianProcessParameters(
            log_error_stddev,
            inducing_locations,
            inducing_pseudo_mean,
            inducing_pseudo_log_err_stddev,
            kernel_params,
        )

        inducing_weights = jnp.zeros((S, OD, M))
        cholesky = jnp.zeros((OD, M, M))
        prior_state = self.prior.init_state(params.kernel_params, self.num_samples, k3)
        state = SparseGaussianProcessState(inducing_weights, cholesky, prior_state)

        return (params, state)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
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
            self.num_inducing,
        )
        inducing_locations = params.inducing_locations
        kernel_params = params.kernel_params
        inducing_weights = state.inducing_weights
        prior_state = state.prior_state

        f_prior = self.prior(kernel_params, prior_state, x)
        f_data = tf2jax.linalg.matvec(
            self.kernel.matrix(kernel_params, x, inducing_locations), inducing_weights
        )  # non-batched

        return f_prior + f_data

    @partial(jax.jit, static_argnums=(0,))
    def randomize(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        key: jnp.ndarray,
    ) -> SparseGaussianProcessState:
        """Samples a new set of random functions from the GP."""
        (S, OD, ID, M, L) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            self.num_inducing,
            self.num_basis,
        )
        inducing_locations = params.inducing_locations
        inducing_pseudo_mean = params.inducing_pseudo_mean
        inducing_pseudo_log_err_stddev = params.inducing_pseudo_log_err_stddev
        kernel_params = params.kernel_params
        prior_state = state.prior_state

        (k1, k2) = jr.split(key)
        prior_state = self.prior.resample_weights(
            kernel_params, prior_state, self.num_samples, k1
        )
        state = SparseGaussianProcessState(
            state.inducing_weights, state.cholesky, prior_state
        )

        (cholesky, _) = jsp.linalg.cho_factor(
            self.kernel.matrix(kernel_params, inducing_locations, inducing_locations)
            + jax.vmap(jnp.diag)(jnp.exp(inducing_pseudo_log_err_stddev * 2)),
            lower=True,
        )
        residual = self.prior(
            params.kernel_params, state.prior_state, inducing_locations
        ) + jnp.exp(inducing_pseudo_log_err_stddev) * jr.normal(k2, (S, OD, M))
        inducing_weights = (
            inducing_pseudo_mean
            - tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(
                tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(
                    residual
                ),
                adjoint=True,
            )
        )  # mean-reparameterized v = \mu + (K + V)^{-1}(-f - \eps)

        return SparseGaussianProcessState(inducing_weights, cholesky, prior_state)

    @partial(jax.jit, static_argnums=(0,))
    def resample_prior_basis(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        key: jnp.ndarray,
    ):
        """Resamples the frequency and phase of the prior random feature basis."""
        prior_state = self.prior.resample_basis(
            params.kernel_params, state.prior_state, key
        )

        return SparseGaussianProcessState(
            state.inducing_weights, state.cholesky, prior_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def prior_kl(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective."""
        (OD, ID, M) = (self.output_dimension, self.input_dimension, self.num_inducing)
        inducing_locations = params.inducing_locations
        inducing_pseudo_mean = params.inducing_pseudo_mean
        inducing_pseudo_log_err_stddev = params.inducing_pseudo_log_err_stddev
        kernel_params = params.kernel_params
        cholesky = state.cholesky

        logdet_term = 2 * jnp.sum(jnp.log(jax.vmap(jnp.diag)(cholesky))) - 2 * jnp.sum(
            inducing_pseudo_log_err_stddev
        )
        kernel_matrix = self.kernel.matrix(
            kernel_params, inducing_locations, inducing_locations
        )
        cholesky_inv = tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(
            tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(jnp.eye(M)),
            adjoint=True,
        )
        trace_term = jnp.sum(cholesky_inv * kernel_matrix)
        reparameterized_quadratic_form_term = jnp.sum(
            inducing_pseudo_mean
            * tf2jax.linalg.matvec(kernel_matrix, inducing_pseudo_mean)
        )
        return (
            logdet_term
            - (OD * ID * M)
            + trace_term
            + reparameterized_quadratic_form_term
        ) / 2

    @partial(jax.jit, static_argnums=(0,))
    def hyperprior(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
    ) -> jnp.ndarray:
        """Returns the log hyperprior regularization term of the GP."""
        return jnp.zeros(())  # temporary

    @partial(jax.jit, static_argnums=(0,))
    def loss(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        key: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, SparseGaussianProcessState]:
        state = self.randomize(params, state, key)

        kl = self.prior_kl(params, state)

        f = self(params, state, x)
        s = jnp.exp(params.log_error_stddev)
        (n_samples, _, n_batch) = f.shape
        c = n_data / (n_batch * n_samples * 2)
        l = n_data * jnp.sum(jnp.log(s)) + c * jnp.sum(((y - f) / s) ** 2)

        r = self.hyperprior(params, state)

        return (kl + l + r, state)