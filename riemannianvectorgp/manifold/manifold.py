from abc import ABC, ABCMeta, abstractmethod
import jax.numpy as jnp


class AbstractRiemannianMainfold(ABC):
    dimension = None

    @abstractmethod
    def laplacian_eigenvalue(
        self,
        n: jnp.ndarray,
    ):
        pass

    @abstractmethod
    def laplacian_eigenfunction(
        self,
        n: jnp.ndarray,
        x: jnp.ndarray,
    ):
        pass