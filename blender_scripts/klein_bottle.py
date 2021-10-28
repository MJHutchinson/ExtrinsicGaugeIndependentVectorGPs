import bpy
import bmesh
import math
from functools import partial
import numpy as np
import os
from mathutils import Vector
from mathutils import Euler

# directory = os.getcwd()
base_dir = os.path.expanduser(
    "~/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/"
)
scripts_dir = os.path.join(base_dir, "blender_scripts")
data_dir = os.path.join(base_dir, "blender")
texture_path = os.path.join(base_dir, "blender", "textures")
col_dir = os.path.join(base_dir, "blender", "col")

os.makedirs(os.path.join(data_dir, "klein_bottle", "renders"), exist_ok=True)

with open(os.path.join(scripts_dir, "render.py")) as file:
    exec(file.read())

reset_scene()
set_renderer_settings(num_samples=16384 if bpy.app.background else 128)
setup_layers()
setup_compositor(
    mask_center=(0.5, 0.3125),
    mask_size=(0.9, 0.325),
    shadow_color_correction_exponent=2.75,
)
(cam_axis, cam_obj) = setup_camera(
    distance=9.125,
    angle=(-np.pi / 16, 0, 0),
    lens=85,
    height=2560,
    crop=(1 / 5, 9 / 10, 1 / 7, 8 / 10),
)
setup_lighting(
    shifts=(-10, -10, 10),
    sizes=(9, 18, 15),
    energies=(1500, 150, 1125),
    horizontal_angles=(-np.pi / 6, np.pi / 3, np.pi / 3),
    vertical_angles=(-np.pi / 3, -np.pi / 6, np.pi / 4),
)
set_resolution(640)

bd_obj = create_backdrop(location=(0, 0, -0.75), scale=(10, 5, 5))
arr_obj = create_vector_arrow(color=(0, 0, 0, 1))

set_object_collections(backdrop=[bd_obj], instancing=[arr_obj])

bm = import_bmesh(os.path.join(data_dir, "klein_bottle", "klein_bottle.obj"))
import_color(bm, name="white", color=(0.7, 0.7, 0.7, 0.5))
klein_obj = add_mesh(bm, name="klein_bottle")
klein_mat = add_vertex_colors(klein_obj)

vf_bm = import_vector_field(os.path.join(data_dir, "klein_bottle", f"sample_vecs.csv"))
vf_obj = add_vector_field(vf_bm, arr_obj, scale=3, name="sample")

set_object_collections(object=[klein_obj, vf_obj])

bpy.ops.object.select_all(action="DESELECT")
klein_obj.select_set(True)
bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")

bpy.ops.object.empty_add(
    type="PLAIN_AXES", align="WORLD", location=klein_obj.location, scale=(1, 1, 1)
)
empty = bpy.context.selected_objects[0]

bpy.ops.object.select_all(action="DESELECT")
klein_obj.select_set(True)
empty.select_set(True)
bpy.ops.object.parent_set(type="OBJECT", keep_transform=False)

bpy.ops.object.select_all(action="DESELECT")
vf_obj.select_set(True)
empty.select_set(True)
bpy.ops.object.parent_set(type="OBJECT", keep_transform=False)

empty.rotation_euler = Euler(
    (math.radians(30), math.radians(130), math.radians(90)), "YZX"
)
empty.location = (0, 0, 0)
empty.scale = (0.5, 0.5, 0.5)

bpy.context.scene.render.filepath = os.path.join(
    data_dir, "klein_bottle", "renders", "klein_bottle.png"
)
if bpy.app.background:
    bpy.ops.render.render(use_viewport=True, write_still=True)
