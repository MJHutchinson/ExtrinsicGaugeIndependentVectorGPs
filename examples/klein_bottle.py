# %%

import jax
import jax.numpy as jnp
from riemannianvectorgp.manifold import S1
import numpy as np

from riemannianvectorgp.utils import (
    klein_bottle_m_to_3d,
    klein_fig8_m_to_3d,
    klein_fig8_double_m_to_3d,
)
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope


# %%
import polyscope as ps

ps.init()
ps.set_up_dir("z_up")

# %%
num_points = 30
u = np.linspace(0, np.pi, num_points + 1)[1:]
v = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
u, v = np.meshgrid(u, v, indexing="ij")
u = u.flatten()
v = v.flatten()
m = np.stack([u, v], axis=-1)

u = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
v = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
u, v = np.meshgrid(u, v, indexing="ij")
u = u.flatten()
v = v.flatten()
m2 = np.stack([u, v], axis=-1)


def _2d_to_3d(m):
    return jnp.stack([m[..., 0], m[..., 1], jnp.zeros_like(m[..., 0])], axis=-1)


klein_mesh = ps.register_surface_mesh(
    f"klein_8_double_surface",
    *mesh_to_polyscope(
        klein_fig8_double_m_to_3d(m, delta=0.1).reshape((num_points, num_points, 3)),
        wrap_x=True,
        wrap_y=True,
        reverse_x=False,
    ),
    # color=(28/255,99/255,227/255),
    color=(1, 1, 1),
    # color=(39/255,119/255,177/255), # matplotlib blue
    # color=(252/255,128/255,43/255), # matplotlib orange
    # color=(51/255, 159/255, 54/255), # matplotlib green
    # color=(217/255, 95/255, 2/255), # colorbrewer orange
    # color=(231/255, 41/255, 139/255), # colorbrewer magenta
    smooth_shade=True,
    material="wax",
)
klein_mesh = ps.register_surface_mesh(
    f"klein_8_surface",
    *mesh_to_polyscope(
        klein_fig8_m_to_3d(m).reshape((num_points, num_points, 3)),
        wrap_x=True,
        wrap_y=True,
        reverse_x=False,
    ),
    # color=(28/255,99/255,227/255),
    color=(1, 1, 1),
    # color=(39/255,119/255,177/255), # matplotlib blue
    # color=(252/255,128/255,43/255), # matplotlib orange
    # color=(51/255, 159/255, 54/255), # matplotlib green
    # color=(217/255, 95/255, 2/255), # colorbrewer orange
    # color=(231/255, 41/255, 139/255), # colorbrewer magenta
    smooth_shade=True,
    material="wax",
)
klein_rect = ps.register_surface_mesh(
    "plane_klein",
    *mesh_to_polyscope(
        _2d_to_3d(m).reshape((num_points, num_points, 3)), wrap_x=False, wrap_y=False
    ),
)
torus_square = ps.register_surface_mesh(
    "plane_torus",
    *mesh_to_polyscope(
        _2d_to_3d(m2).reshape((num_points, num_points, 3)), wrap_x=False, wrap_y=False
    ),
)
# %%
def identifiction(M):
    x, y = M[..., 0], M[..., 1]
    return jnp.stack([x + jnp.pi, 2 * jnp.pi - y], axis=-1)


n = 20
test_points = jnp.stack([jnp.zeros(n), jnp.linspace(0, 2 * jnp.pi, n)], axis=-1)
identifiction_points = ps.register_point_cloud(
    "points",
    _2d_to_3d(jnp.concatenate([test_points, identifiction(test_points)], axis=0)),
)
identifiction_points.add_scalar_quantity(
    "index", jnp.concatenate([jnp.arange(n), jnp.arange(n)])
)
# %%

T2 = S1(0.5) * S1(0.5)

# %%
eig_funcs = T2.laplacian_eigenfunction(jnp.arange(20), m2)
for i in range(eig_funcs.shape[1]):
    torus_square.add_scalar_quantity(f"eigen_{i}", eig_funcs[:, i, 0])
# %%
n_eigs = 200
eig_inds = jnp.arange(n_eigs)
diff = T2.laplacian_eigenfunction(eig_inds, m2) - T2.laplacian_eigenfunction(
    eig_inds, identifiction(m2)
)
test = jnp.mean(jnp.abs(diff), axis=(0, 2))
inds = test < 1e-5
cover_eigs = eig_inds[inds]
non_cover_eigs = eig_inds[~inds]

j_plus = eig_funcs = T2.laplacian_eigenfunction(cover_eigs, m2)
j_minus = eig_funcs = T2.laplacian_eigenfunction(non_cover_eigs, m2)


for i in range(j_plus.shape[0]):
    torus_square.add_scalar_quantity(f"j+{i}", j_plus[:, i, 0])

for i in range(j_minus.shape[0]):
    torus_square.add_scalar_quantity(f"j-{i}", j_minus[:, i, 0])

# %%
