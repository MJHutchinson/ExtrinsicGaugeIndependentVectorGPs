from typing import NamedTuple

import jax
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

import optax

from riemannianvectorgp.sparse_gp import SparseGaussianProcessParameters
from riemannianvectorgp.manifold import (
    S1,
    EmbeddedS1,
    EmbeddedR,
    ProductManifold,
    EmbeddedProductManifold,
)
from riemannianvectorgp.kernel import (
    FourierFeatures,
    ScaledKernel,
    ProductKernel,
    TFPKernel,
    ManifoldProjectionVectorKernel,
    MaternCompactRiemannianManifoldKernel,
    SquaredExponentialCompactRiemannianManifoldKernel,
)

from riemannianvectorgp.utils import circle_distance


def zero_grad_named_tuple(named_tuple):
    fields = []
    for sub in named_tuple:
        if isinstance(sub, jnp.DeviceArray):
            fields.append(jnp.zeros_like(sub))
        else:
            fields.append(zero_grad_named_tuple(sub))
    if isinstance(named_tuple, list):
        return fields
    else:
        return named_tuple.__class__(*fields)


def reverse_eular_integrate_rollouts(
    rollouts, system, estimate_momentum=False, thinning=1, chuck_factor=10
):

    if estimate_momentum:
        positions = rollouts[..., :, 0]
        momentums = (
            (positions[..., 1:] - positions[..., :-1] / system.step_size)
            * system.length
            * system.mass
            / 2
        )
        positions = positions[..., :-1]
        rollouts = jnp.stack([positions, momentums], axis=-1)

    deltas = (rollouts[..., 1:, :] - rollouts[..., :-1, :]) / system.step_size
    rollouts = rollouts[..., :-1, :]

    deltas = deltas[..., ::thinning, :]
    rollouts = rollouts[..., ::thinning, :]

    rollouts = rollouts.reshape((-1, rollouts.shape[-1]))
    deltas = deltas.reshape((-1, rollouts.shape[-1]))
    # Sketchy, chuckout the big delats that are fake...
    delta_norm = jnp.linalg.norm(deltas, axis=-1)
    delta_mean = jnp.mean(jnp.linalg.norm(deltas, axis=-1))

    chuck_inds = delta_norm > delta_mean * chuck_factor

    return rollouts[~chuck_inds], deltas[~chuck_inds]


def pendulum_statespace_kernel(rng, r_lengthscale=1, s1_lengthscale=0.3):
    s1 = EmbeddedS1(1.0)
    r1 = EmbeddedR(1)
    k_s1 = MaternCompactRiemannianManifoldKernel(2.5, s1, 100)
    k_s1_params = k_s1.init_params(next(rng))
    k_s1_params = k_s1_params._replace(log_length_scale=jnp.log(s1_lengthscale))

    k_r1 = TFPKernel(tfk.MaternFiveHalves, 1, 1)
    k_r1_params = k_r1.init_params(next(rng))
    k_r1_params = k_r1_params._replace(log_length_scales=jnp.log(r_lengthscale))

    kernel = ProductKernel(k_s1, k_r1)
    product_kernel_params = kernel.init_params(next(rng))
    product_kernel_params = product_kernel_params._replace(
        sub_kernel_params=[k_s1_params, k_r1_params]
    )
    kernel = ManifoldProjectionVectorKernel(kernel, s1 * r1)
    manifold_kernel_params = kernel.init_params(next(rng))
    manifold_kernel_params = product_kernel_params

    kernel = ScaledKernel(kernel)
    scaled_kernel_params = kernel.init_params(next(rng))
    scaled_kernel_params = scaled_kernel_params._replace(
        sub_kernel_params=manifold_kernel_params
    )
    kernel_params = scaled_kernel_params

    return kernel, kernel_params


def pendulum_statespace_kernel_2(rng, r_lengthscale=1, s1_lengthscale=0.3):
    s1 = EmbeddedS1(1.0)
    r1 = EmbeddedR(1)
    k_s1 = SquaredExponentialCompactRiemannianManifoldKernel(s1, 100)
    k_s1_params = k_s1.init_params(next(rng))
    k_s1_params = k_s1_params._replace(log_length_scale=jnp.log(s1_lengthscale))

    k_r1 = TFPKernel(tfk.ExponentiatedQuadratic, 1, 1)
    k_r1_params = k_r1.init_params(next(rng))
    k_r1_params = k_r1_params._replace(log_length_scale=jnp.log(r_lengthscale))

    kernel = ProductKernel(k_s1, k_r1)
    product_kernel_params = kernel.init_params(next(rng))
    product_kernel_params = product_kernel_params._replace(
        sub_kernel_params=[k_s1_params, k_r1_params]
    )
    kernel = ManifoldProjectionVectorKernel(kernel, s1 * r1)
    manifold_kernel_params = kernel.init_params(next(rng))
    manifold_kernel_params = product_kernel_params

    kernel = ScaledKernel(kernel)
    scaled_kernel_params = kernel.init_params(next(rng))
    scaled_kernel_params = scaled_kernel_params._replace(
        sub_kernel_params=manifold_kernel_params
    )
    kernel_params = scaled_kernel_params

    return kernel, kernel_params


def pendulum_statespace_euclidean_kernel(rng, r_lengthscale=1.0, s1_lengthscale=1.0):
    k = TFPKernel(tfk.ExponentiatedQuadratic, 2, 2)
    k_params = k.init_params(next(rng))
    k_params = k_params._replace(log_length_scale=jnp.log(s1_lengthscale))

    kernel = ScaledKernel(k)
    scaled_kernel_params = kernel.init_params(next(rng))
    scaled_kernel_params = scaled_kernel_params._replace(sub_kernel_params=k_params)
    kernel_params = scaled_kernel_params

    return kernel, kernel_params


def setup_sparse_gp_training(
    sparse_gp_params, sparse_gp_state, rng, b1=0.9, b2=0.999, eps=1e-8, lr=0.003
):
    opt = optax.chain(optax.scale_by_adam(b1=b1, b2=b2, eps=eps), optax.scale(-lr))
    opt_state = opt.init(sparse_gp_params)
    debug_params = [sparse_gp_params]
    debug_states = [sparse_gp_state]
    debug_keys = [rng.key]
    losses = []

    return opt, opt_state, debug_params, debug_states, debug_keys, losses


def train_eular_integration_sparse_gp(
    m_cond,
    v_cond,
    sparse_gp,
    sparse_gp_params,
    sparse_gp_state,
    steps,
    opt,
    opt_state,
    rng,
    debug_params=None,
    debug_states=None,
    debug_keys=None,
    losses=None,
    fix_inducing_points=False,
):
    for i in range(steps):
        ((train_loss, sparse_gp_state), grads) = jax.value_and_grad(
            sparse_gp.loss, has_aux=True
        )(sparse_gp_params, sparse_gp_state, next(rng), m_cond, v_cond, m_cond.shape[0])

        if fix_inducing_points:
            grads = SparseGaussianProcessParameters(
                log_error_stddev=grads.log_error_stddev,
                inducing_locations=jnp.zeros_like(grads.inducing_locations),
                inducing_pseudo_mean=jnp.zeros_like(grads.inducing_pseudo_mean),
                inducing_pseudo_log_err_stddev=jnp.zeros_like(
                    grads.inducing_pseudo_log_err_stddev
                ),
                kernel_params=grads.kernel_params,
            )

        (updates, opt_state) = opt.update(grads, opt_state)
        sparse_gp_params = optax.apply_updates(sparse_gp_params, updates)
        if i <= 10 or i % 20 == 0:
            print(i, "Loss:", train_loss)
        if losses is not None:
            losses.append(train_loss)
            debug_params.append(sparse_gp_params)
            debug_states.append(sparse_gp_state)
            debug_keys.append(rng.key)

    return (
        sparse_gp_params,
        sparse_gp_state,
        opt_state,
        debug_params,
        debug_states,
        debug_keys,
        losses,
    )


def train_trajectory_to_trajectory_sparse_gp(
    real_rollouts,
    system,
    gp_system,
    sparse_gp_params,
    sparse_gp_state,
    steps,
    rollout_length,
    opt,
    opt_state,
    rng,
    debug_params=None,
    debug_states=None,
    debug_keys=None,
    losses=None,
    fix_inducing_points=False,
):
    num_rollouts = real_rollouts.shape[0]
    max_rollout_start = real_rollouts.shape[1] - rollout_length
    for i in range(steps):
        rollout_starts = jr.randint(
            next(rng), (num_rollouts,), minval=0, maxval=max_rollout_start
        )
        initial_states = real_rollouts[jnp.arange(num_rollouts), rollout_starts]
        ground_truth = system.rollout(initial_states, rollout_length)[..., 0][
            ..., np.newaxis
        ]
        gp_rollout = gp_system.rollout(
            sparse_gp_params, sparse_gp_state, initial_states, rollout_length
        )
        mse = circle_distance(ground_truth, gp_rollout).mean()
        print(f"{mse=}")
        parts = gp_system.trajectory_to_trajectory_loss_parts(
            sparse_gp_params,
            sparse_gp_state,
            rollout_length,
            initial_states,
            ground_truth,
            next(rng),
            ground_truth.shape[0] * ground_truth.shape[1],
        )
        print(f"{parts=}")
        ((train_loss, sparse_gp_state), grads) = jax.value_and_grad(
            gp_system.trajectory_to_trajectory_loss, has_aux=True
        )(
            sparse_gp_params,
            sparse_gp_state,
            rollout_length,
            initial_states,
            ground_truth,
            next(rng),
            ground_truth.shape[0] * ground_truth.shape[1],
        )

        if fix_inducing_points:
            grads = SparseGaussianProcessParameters(
                log_error_stddev=grads.log_error_stddev,
                inducing_locations=jnp.zeros_like(grads.inducing_locations),
                inducing_pseudo_mean=jnp.zeros_like(grads.inducing_pseudo_mean),
                inducing_pseudo_log_err_stddev=jnp.zeros_like(
                    grads.inducing_pseudo_log_err_stddev
                ),
                kernel_params=grads.kernel_params,
            )

        (updates, opt_state) = opt.update(grads, opt_state)
        sparse_gp_params = optax.apply_updates(sparse_gp_params, updates)
        if i <= 10 or i % 20 == 0:
            print(i, "Loss:", train_loss)
        if losses is not None:
            losses.append(train_loss)
            debug_params.append(sparse_gp_params)
            debug_states.append(sparse_gp_state)
            debug_keys.append(rng.key)

    return (
        sparse_gp_params,
        sparse_gp_state,
        opt_state,
        debug_params,
        debug_states,
        debug_keys,
        losses,
    )
