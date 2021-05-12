from functools import partial

import numpy as np
from jax import jit
import jax.numpy as jnp

from riemannianvectorgp.utils import _spherical_harmonics
from .manifold import AbstractRiemannianMainfold
from .embedded_manifold import AbstractEmbeddedRiemannianManifold


class S2(AbstractRiemannianMainfold):
    dimension = 2
    compact = True

    def __init__(
        self,
        radius: float,
        max_l: int = 11,
    ):
        self.radius = radius
        self.max_l = max_l

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenvalue(
        self,
        n: int,
    ) -> jnp.ndarray:
        l = jnp.floor(jnp.sqrt(n))
        return l * (l + 1)

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenfunction(
        self,
        n: int,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        phi = x[..., 0]
        theta = x[..., 1]

        x = jnp.cos(theta) * jnp.sin(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(phi)

        e = jnp.stack([x, y, z], axis=-1)

        return _spherical_harmonics(self.max_l, e)[..., n, np.newaxis]

    def __repr__(
        self,
    ):
        if self.radius == 1.0:
            return f"\U0001D4E2\u2082"
        else:
            return f"\U0001D4E2\u2082({self.radius})"


class EmbeddedS2(AbstractEmbeddedRiemannianManifold, S2):
    embedded_dimension = 3

    def __init__(
        self,
        radius: float,
        max_l: int = 11,
    ):
        super().__init__(radius, max_l)

    @partial(jit, static_argnums=(0,))
    def m_to_e(self, M):
        phi = M[..., 0]
        theta = M[..., 1]

        return jnp.stack(
            [
                jnp.sin(phi) * jnp.cos(theta),
                jnp.sin(phi) * jnp.sin(theta),
                jnp.cos(phi),
            ],
            axis=-1,
        )

    @partial(jit, static_argnums=(0,))
    def e_to_m(self, E):
        x = E[..., 0]
        y = E[..., 1]
        z = E[..., 2]
        return jnp.stack([jnp.arccos(z), jnp.arctan2(y, x)], axis=-1)

    @partial(jit, static_argnums=(0,))
    def projection_matrix(self, M):
        phi = M[..., 0]
        theta = M[..., 1]

        e1 = jnp.stack(
            [
                jnp.cos(phi) * jnp.cos(theta),
                jnp.cos(phi) * jnp.sin(theta),
                -jnp.sin(phi),
            ],
            axis=-1,
        )
        e2 = -jnp.stack([-jnp.sin(theta), jnp.cos(theta), jnp.zeros_like(phi)], axis=-1)

        return jnp.stack(
            [
                e1,
                e2,
            ],
            axis=-1,
        )
