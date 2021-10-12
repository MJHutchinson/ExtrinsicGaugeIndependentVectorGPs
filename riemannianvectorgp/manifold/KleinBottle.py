from functools import partial
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
from jax import jit
import jax.numpy as jnp

from riemannianvectorgp.utils import _spherical_harmonics
from .manifold import AbstractRiemannianMainfold
from .embedded_manifold import AbstractEmbeddedRiemannianManifold


class NonOrientableAbstractRiemannianManifold(AbstractRiemannianMainfold):
    double_cover = None
    test_points = None
    double_cover = None

    def __init__(num_eigenfunctions=1000, tol=1e-3):
        self.num_eigenfunctions = num_eigenfunctions
        self.tol = tol

    def precompute_eigenfunctions(num_eigenfunction):
        diff = self.double_cover.laplacian_eigenfunctions(
            jnp.arange(num_eigenfunction), self.test_points
        ) - self.double_cover.laplacian_eigenfunctions(
            jnp.arange(num_eigenfunction), self.identification(self.test_points)
        )
        test = jnp.mean(jnp.abs(diff), axis=(0, 2))



    @abstractmethod
    def identification(M: jnp.array):
        pass







class KleinBottle(AbstractRiemannianMainfold):

    



class EmbeddedKleinBottle(AbstractEmbeddedRiemannianManifold, S2):
