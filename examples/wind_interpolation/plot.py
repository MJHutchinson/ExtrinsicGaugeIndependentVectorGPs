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

def spatial_plot(
    fname1,
    fname2,
    m,
    prediction,
    m_cond,
    v_cond,
    mesh,
    lon_size,
    lat_size
    ):
    
    m = np.asarray(m)
    m_cond = np.asarray(m_cond)
    v_cond = np.asarray(v_cond)
    prediction = np.asarray(prediction)

    plt.figure(figsize=(10, 5))
    # ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0-(5.625/2)))
    # crs = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.Orthographic(130, 50))
    crs = ccrs.RotatedPole()
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    ax.set_global()
    ax.gridlines()
    scale = 100
    
    mean = prediction.mean(axis=0)
    std = jnp.sqrt(((prediction - mean)**2).mean(axis=0).sum(axis=-1))

    ax.quiver(rad2deg(m[:,1]),
              rad2deg(m[:,0], offset=jnp.pi/2),
              mean[:,1],
              mean[:,0],
              alpha=1.,
              color='blue',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=2,
              transform=crs)

    ax.quiver(rad2deg(m_cond[:,1]),
              rad2deg(m_cond[:,0], offset=jnp.pi/2),
              v_cond[:,1],
              v_cond[:,0],
              color='red',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=3,
              transform=crs)

    std = std.reshape(lon_size, lat_size).transpose()
    plt.contourf(*mesh, std, levels=30, zorder=1, transform=crs)

    plt.savefig(fname1)


def spatial_plot_2(
    fname1,
    fname2,
    m,
    prediction,
    m_cond,
    v_cond,
    mesh,
    lon_size,
    lat_size
    ):
    
    m = np.asarray(m)
    m_cond = np.asarray(m_cond)
    v_cond = np.asarray(v_cond)
    prediction = np.asarray(prediction)

    plt.figure(figsize=(10, 5))
    # ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0-(5.625/2)))
    # crs = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.Orthographic(-15, 45))
    crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    ax.set_global()
    scale = 250
    
    mean = prediction.mean(axis=0)
    std = jnp.sqrt(((prediction - mean)**2).mean(axis=0).sum(axis=-1))

    ax.quiver(rad2deg(m[:,1]),
              rad2deg(m[:,0], offset=jnp.pi/2),
              mean[:,1],
              mean[:,0],
              alpha=0.5,
              color='bl',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=2,
              transform=crs)

    ax.quiver(rad2deg(m_cond[:,1]),
              rad2deg(m_cond[:,0], offset=jnp.pi/2),
              v_cond[:,1],
              v_cond[:,0],
              color='red',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=3,
              transform=crs)

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

    ax.plot(x1, y1, c='r', alpha=0.5, linewidth=2, transform=crs)
    ax.plot(x2, y2, c='r', alpha=0.5, linewidth=2, transform=crs)

    std = std.reshape(lon_size, lat_size).transpose()
    # fig = plt.figure(figsize=(10,5))
    # ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0-(5.625/2)))
    # ax.coastlines()
    plt.contourf(*mesh, std, levels=30, zorder=1, transform=crs)

    plt.savefig(fname1)

    # plt.plot(x1, y1, c='r', alpha=0.5, linewidth=2, transform=ccrs.PlateCarree())
    # plt.plot(x2, y2, c='r', alpha=0.5, linewidth=2, transform=ccrs.PlateCarree())

    # plt.title("posterior std")
    # plt.savefig(fname2)


def space_time_plot(
    fname1,
    fname2,
    m,
    prediction,
    m_cond,
    v_cond,
    num_hours,
    ):
    m = np.asarray(m)
    m_cond = np.asarray(m_cond)
    v_cond = np.asarray(v_cond)
    prediction = np.asarray(prediction)

    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0-(5.625/2)))
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()
    scale = 250
    
    mean = prediction.mean(axis=0)
    std = jnp.sqrt(((prediction - mean)**2).mean(axis=0).sum(axis=-1))

    num_points = m[:,1].shape[0] // num_hours
    ax.quiver(rad2deg(m[:num_points,1]),
              rad2deg(m[:num_points,0], offset=jnp.pi/2),
              mean[:num_points,1],
              mean[:num_points,0],
              alpha=0.5,
              color='blue',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=2,
              transform=ccrs.PlateCarree())

    num_points = m_cond[:,1].shape[0] // num_hours
    ax.quiver(rad2deg(m_cond[:num_points,1]),
              rad2deg(m_cond[:num_points,0], offset=jnp.pi/2),
              v_cond[:num_points,1],
              v_cond[:num_points,0],
              color='red',
              scale=scale,
              width=0.003,
              headwidth=3,
              zorder=3,
              transform=ccrs.PlateCarree())

    plt.savefig(fname1)


def sample_data(shape=(20, 30)):
    """
    Returns ``(x, y, u, v, crs)`` of some vector data
    computed mathematically. The returned crs will be a rotated
    pole CRS, meaning that the vectors will be unevenly spaced in
    regular PlateCarree space.

    """
    # crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
    crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)

    x = np.linspace(311.9, 391.1, shape[1])
    y = np.linspace(-23.6, 24.8, shape[0])

    x2d, y2d = np.meshgrid(x, y)
    u = 10 * (2 * np.cos(2 * np.deg2rad(x2d) + 3 * np.deg2rad(y2d + 30)) ** 2)
    v = 20 * np.cos(6 * np.deg2rad(x2d))

    return x, y, u, v, crs


def main():
    ax = plt.axes(projection=ccrs.Orthographic(-10, 45))
    #ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')

    ax.set_global()
    ax.gridlines()

    x, y, u, v, vector_crs = sample_data()
    ax.quiver(x, y, u, v, transform=vector_crs)

    plt.savefig("test.png")


if __name__=="__main__":
    main()
