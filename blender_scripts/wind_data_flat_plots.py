import bpy
import bmesh
import math
from functools import partial
import numpy as np
import os
from mathutils import Vector
from mathutils import Euler

# directory = os.getcwd()
base_dir = os.path.expanduser("~/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/")
scripts_dir = os.path.join(base_dir, "blender_scripts")
data_dir = os.path.join(base_dir, "blender")
texture_path = os.path.join(base_dir, "textures")
col_dir = os.path.join(base_dir, "col")

render_name = 'wind_plots'
os.makedirs(os.path.join(data_dir, render_name, 'renders'), exist_ok=True)

with open(os.path.join(scripts_dir, "render.py")) as file:
    exec(file.read())

reset_scene()
set_renderer_settings(num_samples = 2048 if bpy.app.background else 8)
setup_layers()
setup_compositor(
    mask_center = (0.5,0.5), 
    mask_size = (1.5,1.0), 
    shadow_color_correction_exponent = 2.75
)
(cam_axis, cam_obj) = setup_camera(
    distance=15.0,
    angle=(0, 0, 0),
    lens=85,
    height=1500,
)
setup_lighting(shifts = (-10,-10,10), sizes = (9,18,15), energies = (1500,150,1125),
               horizontal_angles = (-np.pi/6, np.pi/3, np.pi/3), vertical_angles = (-np.pi/3, -np.pi/6, np.pi/4))
set_resolution(1080, aspect=(2, 1), crop=(0,1,0,1))

bd_obj = create_backdrop(location=(0, 0, -1.6), scale=(10, 5, 5))
arr_obj = create_vector_arrow(color=(0.7, 0.7, 0.7, 1))

bm = import_bmesh(os.path.join(data_dir, render_name, "earth_flat.obj"))
# bm = import_bmesh(os.path.join(data_dir, "kernels", "s2.obj"))
# import_color(bm, name='white', color = (1,1,1,1))
import_color(bm, data_file = os.path.join(data_dir, render_name, "s_s2.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot_shift.png"))

line_obj = import_curve(os.path.join(data_dir, render_name, "earth_flat_line.csv"), name="Line")
add_line_color(line_obj, (1,0,0,1))
line_obj2 = import_curve(os.path.join(data_dir, render_name, "earth_flat_line2.csv"), name="Line")
add_line_color(line_obj2, (1,0,0,1))

# VECTOR FIELD
scale = 0.002

vf_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"tracks_flat.csv")
)
# vf_bm = import_vector_field(
#     os.path.join(data_dir, render_name, f"tracks_full_flat.csv")
# )
vf_obj = add_vector_field(
    vf_bm, arr_obj, scale=scale, name="observations"
)

grey_arr_obj = create_vector_arrow(color=(0,0,0, 1.0)) # (0.0, 0.75, 1.0, 1)
mean_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"mean_s2_flat.csv")
)
# mean_bm = import_vector_field(
#     os.path.join(data_dir, render_name, f"mean_full_s2_flat.csv")
# )
mean_obj = add_vector_field(
    mean_bm, grey_arr_obj, scale=scale, name="means"
)

bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
empty = bpy.context.selected_objects[0]
earth_obj.parent = empty
vf_obj.parent = empty
mean_obj.parent = empty
line_obj.parent = empty
line_obj2.parent = empty

set_object_collections(backdrop = [bd_obj], object=[earth_obj, vf_obj, mean_obj, line_obj,line_obj2] , instancing = [grey_arr_obj, arr_obj])


# bpy.context.scene.render.filepath = os.path.join(
#     data_dir, render_name, "renders", 's2_flat.png'
# )
bpy.context.scene.render.filepath = os.path.join(
    data_dir, render_name, "renders", 's2_flat.png'
)
empty.rotation_euler = Euler((0,math.radians(180),math.radians(90)), "XYZ")
# bpy.ops.render.render(use_viewport=True, write_still=True)

cleanup(objects=[earth_obj, vf_obj, mean_obj], materials=[earth_mat], force=True)

bm = import_bmesh(os.path.join(data_dir, render_name, "earth_flat.obj"))
# bm = import_bmesh(os.path.join(data_dir, "kernels", "s2.obj"))
# import_color(bm, name='white', color = (1,1,1,1))
import_color(bm, data_file = os.path.join(data_dir, render_name, "s_s2.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot_shift.png"))

# VECTOR FIELD
scale = 0.002

vf_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"tracks_flat.csv")
)
# vf_bm = import_vector_field(
#     os.path.join(data_dir, render_name, f"tracks_full_flat.csv")
# )
vf_obj = add_vector_field(
    vf_bm, arr_obj, scale=scale, name="observations"
)

grey_arr_obj = create_vector_arrow(color=(0,0,0, 1.0)) # (0.0, 0.75, 1.0, 1)
mean_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"mean_r2_flat.csv")
)
# mean_bm = import_vector_field(
#     os.path.join(data_dir, render_name, f"mean_full_s2_flat.csv")
# )
mean_obj = add_vector_field(
    mean_bm, grey_arr_obj, scale=scale, name="means"
)

bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
empty = bpy.context.selected_objects[0]
earth_obj.parent = empty
vf_obj.parent = empty
mean_obj.parent = empty
line_obj.parent = empty
line_obj2.parent = empty

set_object_collections( object=[earth_obj, vf_obj, mean_obj])


# bpy.context.scene.render.filepath = os.path.join(
#     data_dir, render_name, "renders", 's2_flat.png'
# )
bpy.context.scene.render.filepath = os.path.join(
    data_dir, render_name, "renders", 'r2_flat.png'
)
empty.rotation_euler = Euler((0,math.radians(180),math.radians(90)), "XYZ")