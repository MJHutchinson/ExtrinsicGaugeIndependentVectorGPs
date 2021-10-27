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

os.makedirs(os.path.join(data_dir, 'torus', 'renders'), exist_ok=True)

with open(os.path.join(scripts_dir, "render.py")) as file:
    exec(file.read())

reset_scene()
set_renderer_settings(num_samples = 2048 if bpy.app.background else 128)
setup_layers()
setup_compositor(mask_center = (0.5,0.425), mask_size = (0.925,0.4), shadow_color_correction_exponent = 2.75)
cam_obj = setup_camera(offset = (0,0,-0.25), distance = 24.75, angle = (-5*np.pi/36, 0, 0), lens = 85, height = 430, crop = (1/12,1,1/12,5/6))
setup_lighting(shifts = (-15,-15,15), sizes = (15,24,9), energies = (3000,625,1000), 
               horizontal_angles = (-np.pi/4, np.pi/3, np.pi/4), vertical_angles = (-np.pi/3, -np.pi/4, np.pi/4))
set_resolution(640)

bd_obj = create_backdrop(location=(0, 0, -1), scale=(10, 5, 5))
arr_obj = create_vector_arrow(color=(0, 0, 0, 1))

set_object_collections(backdrop = [bd_obj], instancing = [arr_obj])

bm = import_bmesh(os.path.join(data_dir, "torus", "torus.obj"))
import_color(bm, name='white', color = (0.7,0.7,0.7,1))
obj = add_mesh(bm, name="torus")
klein_mat = add_vertex_colors(obj)

vf_bm = import_vector_field(
    os.path.join(data_dir, "torus", f"sample_vecs.csv")
)
vf_obj = add_vector_field(
    vf_bm, arr_obj, scale=1.5, name="sample"
)

set_object_collections(object = [obj, vf_obj])

bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=obj.location, scale=(1, 1, 1))
empty = bpy.context.selected_objects[0]
obj.parent = empty
vf_obj.parent = empty

empty.rotation_euler = Euler((math.radians(0),math.radians(0),math.radians(0)), "XYZ")
empty.location = (0,0,0)
empty.scale = (4,4,4)

bpy.context.scene.render.filepath = os.path.join(
    data_dir, "torus", "renders", 'torus.png'
)
if bpy.app.background:
    bpy.ops.render.render(use_viewport=True, write_still=True)
