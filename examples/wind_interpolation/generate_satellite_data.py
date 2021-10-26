import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from riemannianvectorgp.gp import GaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
    ManifoldProjectionVectorKernel,
    ScaledKernel,
    TFPKernel,
    ProductKernel
)
from riemannianvectorgp.utils import GlobalRNG
from examples.wind_interpolation.utils import deg2rad, rad2deg, GetDataAlongSatelliteTrack
from skyfield.api import load, EarthSatellite
import click
import pickle
import xesmf as xe

