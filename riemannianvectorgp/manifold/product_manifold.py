from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import jit
from functools import partial

from .manifold import AbstractRiemannianMainfold
from .embedded_manifold import AbstractEmbeddedRiemannianManifold


class ProductManifold(AbstractRiemannianMainfold):
    def __init__(self, *sub_manifolds):
        self.dimension = sum([m.dimension for m in sub_manifolds])
        self.sub_manifolds = sub_manifolds

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenvalue(
        self,
        n: int,
    ) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenfunction(
        self,
        n: int,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        pass

    def __repr__(
        self,
    ):
        return "×".join([str(m) for m in self.sub_manifolds])


class EmbeddedProductManifold(AbstractEmbeddedRiemannianManifold, ProductManifold):
    def __init__(self, *sub_manifolds):
        self.sub_manifolds = sub_manifolds
        self.sub_dimensions = [sm.dimension for sm in self.sub_manifolds]
        self.dimension = sum(self.sub_dimensions)
        self.sub_embedded_dimensions = [
            sm.embedded_dimension for sm in self.sub_manifolds
        ]
        self.embedded_dimension = sum(self.sub_embedded_dimensions)

    @partial(jit, static_argnums=(0,))
    def m_to_e(self, M):
        sub_M = jnp.split(M, self.sub_dimensions[:-1], axis=-1)
        sub_X = [
            self.sub_manifolds[i].m_to_e(sub_M[i])
            for i in range(len(self.sub_manifolds))
        ]
        return jnp.concatenate(sub_X, axis=-1)

    @partial(jit, static_argnums=(0,))
    def e_to_m(self, E):
        sub_E = jnp.split(E, self.sub_dimensions[:-1], axis=-1)
        sub_M = [
            self.sub_manifolds[i].e_to_m(sub_E[i])
            for i in range(len(self.sub_manifolds))
        ]
        return jnp.concatenate(sub_M, axis=-1)

    @partial(jit, static_argnums=(0,))
    def projection_matrix(self, M):
        sub_M = jnp.split(M, self.sub_dimensions[:-1], axis=-1)
        sub_projection_matricies = [
            self.sub_manifolds[i].projection_matrix(sub_M[i])
            for i in range(len(self.sub_manifolds))
        ]
        return jax.vmap(jsp.linalg.block_diag)(*sub_projection_matricies)
