import numpy as np
import jax.numpy as jnp
from examples.wind_interpolation.utils import deg2rad, rad2deg
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from celluloid import Camera

def plot(
    fname,
    mean,
    K,
    m,
    m_cond,
    v_cond,
    num_hours,
    mesh,
    lon_size,
    lat_size,
    plot_sphere
    ):
    m = np.asarray(m)
    m_cond = np.asarray(m_cond)
    v_cond = np.asarray(v_cond)
    mean = np.asarray(mean)

    plt.figure(figsize=(10, 5))

    if plot_sphere:
        scale = 200
        width = 0.006
        headwidth = 3
        ax = plt.axes(projection=ccrs.Orthographic(-45, 45))
        crs = ccrs.RotatedPole(pole_longitude=180)
        ax.set_global()
        ax.gridlines()
    else:
        scale = 250
        width = 0.003
        headwidth = 3
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
        crs = ccrs.PlateCarree()

    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.coastlines()

    t = 4
    ax.quiver(rad2deg(m[t,:,1]),
              rad2deg(m[t,:,0], offset=jnp.pi/2),
              mean[t,:,1],
              mean[t,:,0],
              alpha=0.5,
              color='chocolate',
              scale=scale,
              width=width,
              headwidth=headwidth,
              zorder=2,
              transform=crs)

    ax.quiver(rad2deg(m_cond[t,:,1]),
              rad2deg(m_cond[t,:,0], offset=jnp.pi/2),
              v_cond[t,:,1],
              v_cond[t,:,0],
              color='white',
              scale=scale,
              width=width,
              headwidth=headwidth,
              zorder=3,
              transform=crs)

    spatial_size = lon_size*lat_size
    var_norm = jnp.diag(jnp.trace(K[t*spatial_size:(t+1)*spatial_size, t*spatial_size:(t+1)*spatial_size], axis1=2, axis2=3)).reshape(lon_size, lat_size)
    std_norm = jnp.sqrt(var_norm).transpose()
    CS = plt.contourf(*mesh, std_norm, levels=np.linspace(0.5, 4.5, 30), zorder=1, transform=crs)

    n_points = 200
    x = 1.*np.ones(n_points)
    y = np.linspace(-89, 89, n_points)
    plt.plot(x, y,
            color='deeppink', linewidth=3, #linestyle='--',
            zorder=2,
            transform=crs,
            )
    
    if not plot_sphere:
        cbar = plt.colorbar(CS, shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        x = 359.*np.ones(n_points)
        plt.plot(x, y,
                color='deeppink', linewidth=3, #linestyle='--',
                zorder=2,
                transform=crs,
                )

    plt.savefig(fname, transparent=True, bbox_inches='tight')


def animate(
    fname,
    mean,
    K,
    m,
    m_cond,
    v_cond,
    num_hours,
    mesh,
    lon_size,
    lat_size
    ):
    m = np.asarray(m)
    m_cond = np.asarray(m_cond)
    v_cond = np.asarray(v_cond)
    mean = np.asarray(mean)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
    crs = ccrs.PlateCarree()
    camera = Camera(fig)
    scale = 250
    width = 0.003
    headwidth = 3

    for t in range(num_hours):

        ax.add_feature(cartopy.feature.LAND, zorder=0)
        ax.coastlines()

        ax.quiver(rad2deg(m[t,:,1]),
                  rad2deg(m[t,:,0], offset=jnp.pi/2),
                  mean[t,:,1],
                  mean[t,:,0],
                  alpha=0.8,
                  color='orangered',
                  scale=scale,
                  width=width,
                  headwidth=headwidth,
                  zorder=2,
                  transform=crs)

        ax.quiver(rad2deg(m_cond[t,:,1]),
                  rad2deg(m_cond[t,:,0], offset=jnp.pi/2),
                  v_cond[t,:,1],
                  v_cond[t,:,0],
                  color='white',
                  scale=scale,
                  width=width,
                  headwidth=headwidth,
                  zorder=3,
                  transform=crs)

        spatial_size = lon_size*lat_size
        var_norm = jnp.diag(jnp.trace(K[t*spatial_size:(t+1)*spatial_size, t*spatial_size:(t+1)*spatial_size], axis1=2, axis2=3)).reshape(lon_size, lat_size)
        std_norm = jnp.sqrt(var_norm).transpose()
        plt.contourf(*mesh, std_norm, levels=np.linspace(0.5, 4.5, 30), zorder=1, transform=crs)

        ax.text(0.45, 1.03, f'time = {t+1}', transform=ax.transAxes, fontsize=16)

        camera.snap()

    animation = camera.animate(interval=1000, repeat=True)
    animation.save(fname)


def spacetime_plot(
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
