from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

from .embedded_manifold import EmbeddedManifold


class ProductManifold(EmbeddedManifold):
    def __init__(self, *submanifolds):
        self.submanifolds = submanifolds
        self.sub_dimensions = [sm.dimension for sm in self.submanifolds]
        self.dimension = sum(self.sub_dimensions)
        self.sub_embedded_dimensions = [sm.embedded_dimension for sm in self.submanifolds]
        self.embedded_dimension = sum(self.sub_embedded_dimensions)

    @abstractmethod
    def m_to_e(self, M):
        """Maps points from intrinsic coordinates to Euclidean coordinates

        Parameters
        ----------
        M : jnp.array
            array of intrinsic coordinates
        """
        coords = 

    @abstractmethod
    def e_to_m(self, E):
        """Maps points from Euclidean coordinates to intrinsic coordinates

        Parameters
        ----------
        E : jnp.array
        """
        pass

    @abstractmethod
    def projection_matrix(self, M):
        """Matrix function that projects euclidean vectors into the tangent
        space of the given point.

        Implicitly encodes a choice of gauge.

        Parameters
        ----------
        M : jnp.array
        """
        pass

    @abstractmethod
    def mesh(self, n_points):
        """Generates an intrinsic coordinate mesh of the manifold with some
        discretisation level.

        Parameters
        ----------
        n_points : int.
            discretisation level. Usually (n_points + 1)^d vertices
        """
        pass

    def projection_matrix_to_3d(self, M):
        """Projection matrix to project tangent space vectors to 3d vectors
        for visualisation

        Parameters
        ----------
        M : jnp.ndarray
        """
        raise NotImplementedError()

    def m_to_3d(self, M):
        """Projects intrinsic coordinates to 3D coordinates for visualisation

        Parameters
        ----------
        M : jnp.ndarray
        """
        raise NotImplementedError()

    def project_to_m(self, X, Y):
        """Projects a set of Euclidean locations and vectors to the intrinsic
        coordinates and tangent spaces.

        Parameters
        ----------
        X : jnp.ndarray
            Euclidean coordinates
        Y : jnp.ndarray
            Euclidean vectors

        Returns
        -------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tangent vectors in gauge defined by self.projection_matrix
        """
        M = self.e_to_m(X)
        return M, (self.projection_matrix(M) @ Y[..., np.newaxis]).squeeze(-1)

    def project_to_e(self, M, V):
        """Projects a set of intrinsic coordinates and tangent vectors to
        euclidean space.

        Parameters
        ----------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tangent vectors

        Returns
        -------
        X : jnp.ndarray
            Euclidean coordinates
        Y : jnp.ndarray
            Euclidean vectors
        """
        return (
            self.m_to_e(M),
            (
                jnp.swapaxes(self.projection_matrix(M), -1, -2) @ V[..., np.newaxis]
            ).squeeze(-1),
        )

    def rotate_points(self, M, V, R):
        """Rotate a set of intrinsic coordinates and tangent vectors by a rotation R

        Parameters
        ----------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarry
            Tangent vectors
        R : jnp.ndarray
            Rotation matrix

        Returns
        -------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tangent vectors in gauge defined by self.projection_matrix
        """
        X = self.m_to_e(M)
        X_ = (R @ X[..., np.newaxis]).squeeze(-1)
        M_ = self.e_to_m(X_)
        vector_transport_map = (
            self.projection_matrix(M_)
            @ R
            @ jnp.swapaxes(self.projection_matrix(M), -1, -2)
        )
        V_ = (vector_transport_map @ V[..., np.newaxis]).squeeze(-1)
        return M_, V_

    def project_to_3d(self, M, V):
        """Projects intrinsic coordinates and tangent vectors to 3D for visualisation

        Parameters
        ----------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tanget vectors

        Returns
        -------
        X : jnp.ndarray
            3D coordinates
        Y : jnp.ndarray
            3D vectors
        """
        X = self.m_to_3d(M)
        Y = (
            jnp.swapaxes(self.projection_matrix_to_3d(M), -1, -2) @ V[..., np.newaxis]
        ).squeeze()
        return X, Y 
