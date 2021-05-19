from abc import ABC, abstractmethod
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial


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
        print(f"{states.shape=}")
        deltas = self.gp(gp_params, gp_state, jnp.expand_dims(states, 1))
        deltas = jnp.squeeze(deltas, 1)
        print(f"{deltas.shape=}")
        return states + self.step_size * deltas

    @partial(jax.jit, static_argnums=(0, 4))
    def rollout(
        self,
        gp_params,
        gp_state,
        initial_state: jnp.ndarray,
        num_steps: int,
    ):
        # def unvectozied_scan(initial_state: jnp.array):
        #     def scan_function(
        #         states: jnp.array, ignored_incoming_array_values: jnp.array
        #     ):
        #         next_states = self.step(gp_params, gp_state, states)
        #         return (next_states, next_states)

        #     (_, output_array) = jax.lax.scan(
        #         scan_function, initial_state, None, length=num_steps
        #     )
        #     return output_array

        # return jnp.vectorize(unvectozied_scan, signature="(n)->(m,k)")(initial_state)

        def scan_function(states: jnp.array, ignored_incoming_array_values: jnp.array):
            next_states = self.step(gp_params, gp_state, states)
            return (next_states, next_states)

        (_, output_array) = jax.lax.scan(
            scan_function, initial_state, None, length=num_steps
        )
        return jnp.swapaxes(output_array, 0, 1)
