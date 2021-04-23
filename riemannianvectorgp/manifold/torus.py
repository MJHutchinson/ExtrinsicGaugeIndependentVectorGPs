from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

from .embedded_manifold import EmbeddedManifold
