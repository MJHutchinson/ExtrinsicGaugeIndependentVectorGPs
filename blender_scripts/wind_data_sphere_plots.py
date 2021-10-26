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
set_renderer_settings(num_samples = 2048 if bpy.app.background else 1)
setup_layers()
setup_compositor(mask_center = (0.5,0.3125), mask_size = (0.675,0.325), shadow_color_correction_exponent = 2.75)
(cam_axis, cam_obj) = setup_camera(distance = 9.125, angle = (-np.pi/16, 0, 0), lens = 85, height = 2560, crop = (1/5,9/10,0,10/11))
setup_lighting(shifts = (-10,-10,10), sizes = (9,18,15), energies = (1500,150,1125),
               horizontal_angles = (-np.pi/6, np.pi/3, np.pi/3), vertical_angles = (-np.pi/3, -np.pi/6, np.pi/4))
set_resolution(580, aspect=(3,2))

bd_obj = create_backdrop(location=(0, 0, -1), scale=(10, 5, 5))
arr_obj = create_vector_arrow(color=(0.7,0.7,0.7, 1))
grey_arr_obj = create_vector_arrow(color=(0,0,0, 1.0)) # (0.0, 0.75, 1.0, 1)
ell_obj = create_elliptical_torus(line_thickness = 0.1, vertical_thickness = 0.5)

set_object_collections(backdrop = [bd_obj], instancing = [grey_arr_obj, arr_obj, ell_obj])


bm = import_bmesh(os.path.join(data_dir, render_name, "earth.obj"))
# bm = import_bmesh(os.path.join(data_dir, "kernels", "r2.obj"))
# import_color(bm, name='white', color = (1,1,1,1))
import_color(bm, data_file = os.path.join(data_dir, render_name, "s_r2.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot_shift.png"))

line_obj = import_curve(os.path.join(data_dir, render_name, "earth_line.csv"), name="Line")
add_line_color(line_obj, (1,0,0,1))

# VECTOR FIELD
width = 7
scale = 0.0015
ellipse_scale = 8

vf_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"tracks.csv")
)
vf_obj = add_vector_field(
    vf_bm, arr_obj, scale=scale, name="observations"
)

mean_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"mean_r2.csv")
)
mean_obj = add_vector_field(
    mean_bm, grey_arr_obj, scale=scale, name="means"
)

vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].input_type_x = "FLOAT"
vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].inputs[2].default_value = width
vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].input_type_z = "FLOAT"
vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].inputs[6].default_value = width

# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].input_type_x = "FLOAT"
# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].inputs[2].default_value = width
# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].input_type_z = "FLOAT"
# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].inputs[6].default_value = width

cc_bm = import_vector_field(os.path.join(data_dir, render_name, f"covariance_r2.csv"))
cc_obj = add_vector_field(cc_bm, ell_obj, scale = scale * ellipse_scale, name='cc')

bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
empty = bpy.context.selected_objects[0]
earth_obj.parent = empty
vf_obj.parent = empty
mean_obj.parent = empty
line_obj.parent = empty
cc_obj.parent = empty


set_object_collections(object=[earth_obj, vf_obj, mean_obj, line_obj, cc_obj])


empty.rotation_euler = Euler((math.radians(160),math.radians(-25),math.radians(125)), "XYZ")
empty.scale = (-1,1,1)

bpy.context.scene.render.filepath = os.path.join(
    data_dir, render_name, "renders", 'r2_sphere.png'
)
if bpy.app.background:
    bpy.ops.render.render(use_viewport=True, write_still=True)

cleanup(objects=[earth_obj, vf_obj, mean_obj, cc_obj], materials=[earth_mat], force=True)


bm = import_bmesh(os.path.join(data_dir, render_name, "earth.obj"))
# bm = import_bmesh(os.path.join(data_dir, "kernels", "r2.obj"))
# import_color(bm, name='white', color = (1,1,1,1))
import_color(bm, data_file = os.path.join(data_dir, render_name, "s_s2.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot_shift.png"))

# VECTOR FIELD
vf_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"tracks.csv")
)
vf_obj = add_vector_field(
    vf_bm, arr_obj, scale=scale, name="observations"
)

grey_arr_obj = create_vector_arrow(color=(0,0,0, 1.0)) # (0.0, 0.75, 1.0, 1)
mean_bm = import_vector_field(
    os.path.join(data_dir, render_name, f"mean_s2.csv")
)
mean_obj = add_vector_field(
    mean_bm, grey_arr_obj, scale=scale, name="means"
)

vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].input_type_x = "FLOAT"
vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].inputs[2].default_value = width
vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].input_type_z = "FLOAT"
vf_obj.modifiers["observations"].node_group.nodes["Attribute Combine XYZ"].inputs[6].default_value = width

# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].input_type_x = "FLOAT"
# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].inputs[2].default_value = width
# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].input_type_z = "FLOAT"
# mean_obj.modifiers["means"].node_group.nodes["Attribute Combine XYZ"].inputs[6].default_value = width

cc_bm = import_vector_field(os.path.join(data_dir, render_name, f"covariance_s2.csv"))
cc_obj = add_vector_field(cc_bm, ell_obj, scale = scale * ellipse_scale, name='cc')

bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
empty = bpy.context.selected_objects[0]
earth_obj.parent = empty
vf_obj.parent = empty
mean_obj.parent = empty
line_obj.parent = empty
cc_obj.parent = empty

set_object_collections(object=[earth_obj, vf_obj, mean_obj, cc_obj])

empty.rotation_euler = Euler((math.radians(160),math.radians(-25),math.radians(125)), "XYZ")
empty.scale = (-1,1,1)

bpy.context.scene.render.filepath = os.path.join(
    data_dir, render_name, "renders", 's2_sphere.png'
)
if bpy.app.background:
    bpy.ops.render.render(use_viewport=True, write_still=True)