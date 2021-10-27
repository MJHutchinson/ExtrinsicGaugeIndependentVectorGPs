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
texture_path = os.path.join(base_dir, "blender","textures")
col_dir = os.path.join(base_dir, "blender", "col")

os.makedirs(os.path.join(data_dir, 'blank_to_wrong', 'renders'), exist_ok=True)

with open(os.path.join(scripts_dir, "render.py")) as file:
    exec(file.read())

reset_scene()
set_renderer_settings(num_samples=16 if bpy.app.background else 128)
(cam_axis, cam_obj) = setup_camera(
    distance=15.5,
    angle=(0, 0, 0),
    lens=85,
    height=1500,
)
setup_lighting(
    shifts=(-10, -10, 10),
    sizes=(9, 18, 15),
    energies=(1500, 150, 1125),
    horizontal_angles=(-np.pi / 6, np.pi / 3, np.pi / 3),
    vertical_angles=(-np.pi / 3, -np.pi / 6, np.pi / 4),
)
set_resolution(1080, aspect=(16,9))

bd_obj = create_backdrop(location=(0, 0, -2), scale=(10, 5, 5))
arr_obj = create_vector_arrow(color=(1, 0, 0, 1))
# set_object_collections(backdrop=[bd_obj], instancing=[arr_obj])

bm = import_bmesh(os.path.join(data_dir, "unwrap_sphere", "frame_59.obj"))
import_color(bm, name='s_wrong', data_file = os.path.join(data_dir, "kernels", "s_wrong.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
import_color(bm, name='s_right', data_file = os.path.join(data_dir, "kernels", "s_right.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))

# VECTOR FIELD
vf_bm = import_vector_field(
    os.path.join(data_dir, "unwrap_sphere", f"frame_59.csv")
)
vf_obj = add_vector_field(
    vf_bm, arr_obj, scale=3, name="observations"
)

arr_obj = create_vector_arrow(color=(0.3,0.3,0.3, 1.0)) # (0.0, 0.75, 1.0, 1)
mean_bm = import_vector_field(
    os.path.join(data_dir, "unwrap_sphere", f"frame_59.csv")
)
mean_bm = import_vector_field(
    os.path.join(data_dir, "kernels", f"mean_wrong.csv"), name="_wrong"
)
mean_bm = import_vector_field(
    os.path.join(data_dir, "kernels", f"mean_right.csv"), bm=mean_bm, name="_right"
)
mean_obj = add_vector_field(
    mean_bm, arr_obj, scale=3, name="means"
)
vec_fraction_node = mix_geometry_attributes(mean_obj, ['arrow', 'normal_x', 'normal_z'], '_wrong', '_right')

bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
empty = bpy.context.selected_objects[0]
earth_obj.parent = empty
vf_obj.parent = empty
mean_obj.parent = empty
empty.rotation_euler = Euler((0,0,math.radians(90)), "XYZ")

mix_factor = earth_obj.data.materials[0].node_tree.nodes['Mix'].inputs[0]
mix_factor.keyframe_insert('default_value', frame=0)
vec_fraction_node.outputs['Value'].default_value = 0.0
vec_fraction_node.outputs['Value'].keyframe_insert('default_value', frame=0)

mix_factor.default_value = 1.0
mix_factor.keyframe_insert('default_value', frame=30)
vec_fraction_node.outputs['Value'].default_value = 1.0
vec_fraction_node.outputs['Value'].keyframe_insert('default_value', frame=30)

bpy.context.scene.frame_end = 30

bpy.context.scene.render.filepath = os.path.join(
    data_dir, "wrong_to_right", "renders", f"frame_"
)
bpy.ops.render.render(use_viewport=True, animation=True, write_still=True)
