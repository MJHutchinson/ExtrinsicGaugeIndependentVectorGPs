# %%
%load_ext autoreload
%autoreload 2
import os

os.chdir("..")

# %%
import numpy as np
import jax.numpy as jnp
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
)
from riemannianvectorgp.utils import (
    mesh_to_obj,
    save_obj,
    make_faces_from_vectors,
    regular_square_mesh_to_obj,
    make_scalar_texture,
    square_mesh_to_obj,
    export_vec_field,
    import_obj
)
import potpourri3d as pp3d

import polyscope as ps

# %%

ps.init()
ps.set_up_dir("z_up")

color1 = (36 / 255, 132 / 255, 141 / 255)
color2 = (114 / 255, 0 / 255, 168 / 255)
color3 = (255 / 255, 112 / 255, 0 / 255)


# %%

S2 = EmbeddedS2(1.0)
num_points = 30
phi = np.linspace(0, np.pi, num_points)
theta = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
phi, theta = np.meshgrid(phi, theta, indexing="ij")
phi = phi.flatten()
theta = theta.flatten()
m = np.stack(
    [phi, theta], axis=-1
)

sphere_mesh = ps.register_surface_mesh(
    "Sphere",
    *mesh_to_polyscope(S2.m_to_e(m).reshape((num_points, num_points, 3)), wrap_x=False),
    color=(1, 1, 1),
    smooth_shade=True,
    material="wax",
)
sphere_mesh.set_vertex_tangent_basisX(S2.projection_matrix(m)[..., 0])


rng = GlobalRNG(0)
kernel = MaternCompactRiemannianManifoldKernel(1.5, S2, 144)
gp = SparseGaussianProcess(kernel, 1, 144, 3)
(params, state) = gp.init_params_with_state(next(rng))
params = params._replace(
    kernel_params=params.kernel_params._replace(log_length_scale=jnp.log(0.1))
)
state = gp.randomize(params, state, next(rng))

# %%
from skyfield.api import wgs84, load, EarthSatellite

satellite = EarthSatellite(
    "1 3600U 18066A   21289.57131473  .00052668  00000-0  20943-3 0  9995",
    "2 43600  96.7161 294.5087 0007995 112.6080 247.6039 15.86489541182450"
)

ts = load.timescale()

hours_ = [14,15,16,17, 18, 19]
minutes_ = list(range(60))

minutes = np.array(minutes_ * len(hours_))
hours = np.array(list(np.repeat(hours_, len(minutes_))))

time_span = ts.utc(2021, 10, 16, hours, minutes)

offset = 165
time_span = time_span[offset:offset+90]

geocentric = satellite.at(time_span)
subpoint = wgs84.subpoint(geocentric)

lon_location = subpoint.longitude.radians # Range: [-pi, pi]
lon_location = np.where(lon_location > 0, lon_location, 2*np.pi + lon_location) # Range: [0, 2pi]
lat_location = subpoint.latitude.radians + jnp.pi/2 # Range: [0, pi]

track_angles = np.stack([lat_location, lon_location], axis=-1)
track_points = S2.m_to_e(track_angles)
track_points_intrinsic, track_vecs_intrinsic = S2.project_to_m(track_points, gp.prior(params.kernel_params, state.prior_state, track_angles)[..., 0].T)
track_vecs = S2.project_to_e(track_points_intrinsic, track_vecs_intrinsic)[1]

track_cloud = ps.register_point_cloud('track', track_points)
track_cloud.add_vector_quantity('samples', track_vecs)

# V, F = mesh_to_polyscope(S2.m_to_e(m).reshape((num_points, num_points, 3)), wrap_x=False)
# save_obj(mesh_to_obj(V, F, uv_coords=m / jnp.array([jnp.pi, 2 * jnp.pi])), f"blender/satellite_tracks/earth.obj")

for i in range(len(track_angles)):
    export_vec_field(track_points[i:i+1, :], track_vecs[i:i+1, :], f"blender/satellite_tracks/vec_{i}.csv")

np.savetxt(f"blender/satellite_tracks/track.csv", track_points, delimiter=',')
np.savetxt(f"blender/satellite_tracks/track_angles.csv", track_angles, delimiter=',')
np.savetxt(f"blender/satellite_tracks/track_vecs.csv", track_vecs, delimiter=',')
np.savetxt(f"blender/satellite_tracks/track_intrinsic_vecs.csv", track_vecs_intrinsic, delimiter=',')

# %%
