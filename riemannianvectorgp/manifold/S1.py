import numpy as np
import jax.numpy as jnp

from .manifold import AbstractRiemannianMainfold
from .embedded_manifold import AbstractEmbeddedRiemannianManifold


class S1(AbstractRiemannianMainfold):
    dimension = 1

    def __init__(
        self,
        radius: float,
    ):
        self.radius = radius

    def laplacian_eigenvalue(
        self,
        n: int,
    ) -> jnp.ndarray:
        freq = (n + 2) // 4
        return jnp.power(2 * jnp.pi * freq / self.radius, 2)

    def laplacian_eigenfunction(
        self,
        n: int,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        n = n + 2
        freq = n // 4
        phase = (jnp.pi / 2) * (n % 2)
        neg = -((((n // 2) + 1) % 2) * 2 - 1)
        phase = phase * neg
        return jnp.cos(x * freq + phase)[..., np.newaxis, :, :]


class EmbeddedS1(AbstractEmbeddedRiemannianManifold, S1):
    embedded_dimension = 2

    def __init__(
        self,
        radius: float,
    ):
        super().__init__(radius)

    def m_to_e(self, M):
        pass

    def e_to_m(self, E):
        pass

    def projection_matrix(self, M):
        pass