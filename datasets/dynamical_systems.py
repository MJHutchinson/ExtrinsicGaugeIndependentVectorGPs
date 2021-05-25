from abc import ABC, abstractmethod
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial

from riemannianvectorgp.sparse_gp import (
    SparseGaussianProcessParameters,
    SparseGaussianProcessState,
)
from riemannianvectorgp.utils import circle_distance


class DynamicalSystem(ABC):
    @partial(jax.jit, static_argnums=(0, 2))
    def rollout(
        self,
        initial_state: jnp.ndarray,
        num_steps: int,
    ):
        def unvectozied_scan(initial_state: jnp.array):
            def scan_function(
                states: jnp.array, ignored_incoming_array_values: jnp.array
            ):
                next_states = self.step(states)
                return (next_states, next_states)

            (_, output_array) = jax.lax.scan(
                scan_function, initial_state, None, length=num_steps
            )
            return output_array

        return jnp.vectorize(unvectozied_scan, signature="(n)->(m,k)")(initial_state)

    @abstractmethod
    def dynamics_gradient_field(self, position: jnp.ndarray, momentum: jnp.ndarray):
        pass


class HamiltonianSystem(DynamicalSystem):
    @abstractmethod
    def hamiltonian(self, position: jnp.ndarray, momentum: jnp.ndarray):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(self, states: jnp.ndarray):
        (position, momentum) = jnp.split(states, 2, axis=-1)
        momentum = momentum - self.step_size / 2 * jnp.vectorize(
            jax.grad(self.hamiltonian, argnums=0), signature="(n),(n)->(n)"
        )(position, momentum)
        position = position + self.step_size * jnp.vectorize(
            jax.grad(self.hamiltonian, argnums=1), signature="(n),(n)->(n)"
        )(position, momentum)
        momentum = momentum - self.step_size / 2 * jnp.vectorize(
            jax.grad(self.hamiltonian, argnums=0), signature="(n),(n)->(n)"
        )(position, momentum)
        return jnp.concatenate((position, momentum), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def hamiltonian_gradient_field(self, position, momentum):
        position_delta = jnp.vectorize(
            jax.grad(self.hamiltonian, argnums=1), signature="(n),(n)->(n)"
        )(position, momentum)
        momentum_delta = -jnp.vectorize(
            jax.grad(self.hamiltonian, argnums=0), signature="(n),(n)->(n)"
        )(position, momentum)
        return jnp.concatenate((position_delta, momentum_delta), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def dynamics_gradient_field(self, position: jnp.ndarray, momentum: jnp.ndarray):
        return self.hamiltonian_gradient_field(position, momentum)


class NonConservativeHamiltonianSystem(DynamicalSystem):
    @abstractmethod
    def hamiltonian(self, position: jnp.ndarray, momentum: jnp.ndarray):
        pass

    @abstractmethod
    def non_conservative_field(self, position: jnp.ndarray, momentum: jnp.ndarray):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(self, states: jnp.ndarray):
        (position, momentum) = jnp.split(states, 2, axis=-1)
        momentum = momentum - self.step_size / 2 * (
            jnp.vectorize(
                jax.grad(self.hamiltonian, argnums=0), signature="(n),(n)->(n)"
            )(position, momentum)
            - jnp.vectorize(self.non_conservative_field, signature="(n),(n)->(m)")(
                position, momentum
            )[..., 1]
        )
        position = position + self.step_size * (
            jnp.vectorize(
                jax.grad(self.hamiltonian, argnums=1), signature="(n),(n)->(n)"
            )(position, momentum)
            + jnp.vectorize(self.non_conservative_field, signature="(n),(n)->(m)")(
                position, momentum
            )[..., 0]
        )
        momentum = momentum - self.step_size / 2 * (
            jnp.vectorize(
                jax.grad(self.hamiltonian, argnums=0), signature="(n),(n)->(n)"
            )(position, momentum)
            - jnp.vectorize(self.non_conservative_field, signature="(n),(n)->(m)")(
                position, momentum
            )[..., 1]
        )
        return jnp.concatenate((position, momentum), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def hamiltonian_gradient_field(self, position, momentum):
        position_delta = jnp.vectorize(
            jax.grad(self.hamiltonian, argnums=1), signature="(n),(n)->(n)"
        )(position, momentum)
        momentum_delta = -jnp.vectorize(
            jax.grad(self.hamiltonian, argnums=0), signature="(n),(n)->(n)"
        )(position, momentum)
        return jnp.concatenate((position_delta, momentum_delta), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def dynamics_gradient_field(self, position: jnp.ndarray, momentum: jnp.ndarray):
        return self.hamiltonian_gradient_field(
            position, momentum
        ) + self.non_conservative_field(position, momentum)


class PendulumSystem(HamiltonianSystem):
    def __init__(
        self,
        mass: float = 1.0,
        length: float = 2.0,
        gravity: float = 9.8,
        step_size: float = 0.01,
    ):
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.step_size = step_size

    @partial(jax.jit, static_argnums=(0,))
    def hamiltonian(
        self,
        theta: jnp.ndarray,
        p_theta: jnp.ndarray,
    ):
        return jnp.squeeze(
            p_theta ** 2 / (2 * self.mass * self.length ** 2)
            + self.mass * self.gravity * self.length * (1 - jnp.cos(theta)),
            axis=-1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, states: jnp.ndarray):
        (theta, p_theta) = jnp.split(super().step(states), 2, axis=-1)
        theta = theta % (2 * jnp.pi)
        return jnp.concatenate((theta, p_theta), axis=-1)


class FrictionPendulumSystem(NonConservativeHamiltonianSystem):
    def __init__(
        self,
        mass: float = 1.0,
        length: float = 2.0,
        gravity: float = 9.8,
        friction: float = 1.0,
        step_size: float = 0.01,
    ):
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.friction = friction
        self.step_size = step_size

    @partial(jax.jit, static_argnums=(0,))
    def hamiltonian(
        self,
        theta: jnp.ndarray,
        p_theta: jnp.ndarray,
    ):
        return jnp.squeeze(
            p_theta ** 2 / (2 * self.mass * self.length ** 2)
            + self.mass * self.gravity * self.length * (1 - jnp.cos(theta)),
            axis=-1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def non_conservative_field(self, position: jnp.ndarray, momentum: jnp.ndarray):
        return jnp.concatenate(
            [jnp.zeros_like(position), -self.friction * momentum / self.mass], axis=-1
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, states: jnp.ndarray):
        (theta, p_theta) = jnp.split(super().step(states), 2, axis=-1)
        theta = theta % (2 * jnp.pi)
        return jnp.concatenate((theta, p_theta), axis=-1)


class GPDynamicalSystem(DynamicalSystem):
    def __init__(
        self,
        gp,
        step_size: float = 0.01,
    ):
        self.gp = gp
        self.step_size = step_size

    @partial(jax.jit, static_argnums=(0,))
    def step(self, gp_params, gp_state, states: jnp.ndarray):
        D = int(states.shape[-1] / 2)

        deltas = self.gp(gp_params, gp_state, jnp.expand_dims(states, 1))
        deltas = jnp.squeeze(deltas, 1)
        (position, momentum) = jnp.split(states, 2, axis=-1)
        momentum = momentum + self.step_size / 2 * deltas[..., D:]
        states = jnp.concatenate([position, momentum], axis=-1)

        deltas = self.gp(gp_params, gp_state, jnp.expand_dims(states, 1))
        deltas = jnp.squeeze(deltas, 1)
        (position, momentum) = jnp.split(states, 2, axis=-1)
        position = position + self.step_size * deltas[..., :D]
        states = jnp.concatenate([position, momentum], axis=-1)

        deltas = self.gp(gp_params, gp_state, jnp.expand_dims(states, 1))
        deltas = jnp.squeeze(deltas, 1)
        (position, momentum) = jnp.split(states, 2, axis=-1)
        momentum = momentum + self.step_size / 2 * deltas[..., D:]
        states = jnp.concatenate([position, momentum], axis=-1)

        return states

    def dynamics_gradient_field(
        self, gp_params, gp_state, position: jnp.ndarray, momentum: jnp.ndarray
    ):
        states = jnp.expand_dims(jnp.stack([position, momentum], axis=-1), 1)
        return jnp.squeeze(self.gp(gp_params, gp_state, jnp.expand_dims(states, 1)), 1)

    @partial(jax.jit, static_argnums=(0, 4))
    def rollout(
        self,
        gp_params,
        gp_state,
        initial_state: jnp.ndarray,
        num_steps: int,
    ):
        def unvectozied_scan(initial_state: jnp.array):
            def scan_function(
                states: jnp.array, ignored_incoming_array_values: jnp.array
            ):
                next_states = self.step(gp_params, gp_state, states)
                return (next_states, next_states)

            (_, output_array) = jax.lax.scan(
                scan_function, initial_state, None, length=num_steps
            )
            (theta, p_theta) = jnp.split(output_array, 2, axis=-1)
            theta = theta % (2 * jnp.pi)
            output_array = jnp.concatenate((theta, p_theta), axis=-1)
            return jnp.swapaxes(output_array, 0, 1)

        return jnp.vectorize(unvectozied_scan, signature="(s,n)->(s,m,k)")(
            initial_state
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def trajectory_to_trajectory_loss(
        self,
        gp_params: SparseGaussianProcessParameters,
        gp_state: SparseGaussianProcessState,
        rollout_length: int,
        initial_states: jnp.ndarray,
        ground_truth: jnp.ndarray,
        key: jnp.ndarray,
        n_data: int,
    ):

        gp_state = self.gp.randomize(gp_params, gp_state, key)

        gp_rollouts = self.rollout(
            gp_params,
            gp_state,
            initial_states,
            rollout_length,
        )[..., 0][..., np.newaxis]

        kl = self.gp.prior_kl(gp_params, gp_state)

        s = jnp.exp(gp_params.log_error_stddev)
        (n_samples, n_batch, _) = gp_rollouts.shape
        c = n_data / (n_batch * n_samples * 2)
        dist = circle_distance(ground_truth, gp_rollouts)
        l = n_data * jnp.sum(jnp.log(s)) + c * jnp.sum((dist / s) ** 2)

        r = self.gp.hyperprior(gp_params, gp_state)

        return (kl + l + r, gp_state)

    @partial(jax.jit, static_argnums=(0, 3))
    def trajectory_to_trajectory_loss_parts(
        self,
        gp_params: SparseGaussianProcessParameters,
        gp_state: SparseGaussianProcessState,
        rollout_length: int,
        initial_states: jnp.ndarray,
        ground_truth: jnp.ndarray,
        key: jnp.ndarray,
        n_data: int,
    ):

        # gp_state = self.gp.randomize(gp_params, gp_state, key)

        gp_rollouts = self.rollout(
            gp_params,
            gp_state,
            initial_states,
            rollout_length,
        )[..., 0][..., np.newaxis]

        kl = self.gp.prior_kl(gp_params, gp_state)

        s = jnp.exp(gp_params.log_error_stddev)
        (n_samples, n_batch, _) = gp_rollouts.shape
        c = n_data / (n_batch * n_samples * 2)
        dist = circle_distance(ground_truth, gp_rollouts)
        l = n_data * jnp.sum(jnp.log(s)) + c * jnp.sum((dist / s) ** 2)

        r = self.gp.hyperprior(gp_params, gp_state)

        return kl, l, r
