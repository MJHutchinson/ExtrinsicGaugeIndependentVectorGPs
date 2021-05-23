import numpy as np
import jax.numpy as jnp
from examples.wind_interpolation.utils import deg2rad, rad2deg, refresh_kernel, GetDataAlongSatelliteTrack
from copy import deepcopy
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from examples.wind_interpolation.utils import Hyperprior, SparseGaussianProcessWithHyperprior
import click
import pickle
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

def spatial_plot(fname1, fname2, gp, params, state, m, m_cond, v_cond, mesh, lon_size, lat_size):
    prediction = gp(params, state, m)

    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    scale = 250
    
    mean = prediction.mean(axis=0)
    std = jnp.sqrt(((prediction - mean)**2).mean(axis=0).sum(axis=-1))
    plt.quiver(rad2deg(m[:,1]),
               rad2deg(m[:,0], offset=jnp.pi/2),
               mean[:,1],
               mean[:,0],
               alpha=0.5,
               color='blue',
               scale=scale,
               width=0.003,
               headwidth=3,
               zorder=2)

    plt.quiver(rad2deg(m_cond[:,1]),
               rad2deg(m_cond[:,0], offset=jnp.pi/2),
               v_cond[:,1],
               v_cond[:,0],
               color='red',
               scale=scale,
               width=0.003,
               headwidth=3,
               zorder=3)

    # Plot satellite trajectories (we split it in two parts to respect periodicity)
    def _where_is_jump(x):
        for i in range(1,len(x)):
            if np.abs(x[i-1] - x[i]) > 180:
                return i

    idx = _where_is_jump(rad2deg(m_cond[:, 1]))

    x1 = deepcopy(rad2deg(m_cond[:idx+1, 1]))
    y1 = rad2deg(m_cond[:idx+1, 0], offset=jnp.pi/2)
    x1[idx] = x1[idx] - 360

    x2 = deepcopy(rad2deg(m_cond[idx-1:, 1]))
    y2 = rad2deg(m_cond[idx-1:, 0], offset=jnp.pi/2)
    x2[0] = x2[0] + 360

    plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2)
    plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2)

    plt.savefig(fname1)

    std = std.reshape(lon_size, lat_size).transpose()
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.contourf(*mesh, std, levels=30, zorder=1)

    plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2)
    plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2)

    plt.title("posterior std")
    plt.savefig(fname2)


def main(logdir, geometry):
    with open(logdir+"/"+geometry+"_params_and_state_for_spatial_interpolation.pickle", "rb") as f:
        sparse_gp = pickle.load(f)

    with open(logdir+"/"+geometry+"_sparse_gp.pickle", "rb") as f:
        sparse_gp = pickle.load(f)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main("log", "r2")