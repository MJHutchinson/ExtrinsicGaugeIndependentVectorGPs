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

bm = import_bmesh(os.path.join(data_dir, "unwrap_sphere", "frame_0.obj"))
import_color(bm, color = (0.8,1,1,1))
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))

bpy.ops.import_scene.obj(
    filepath=os.path.join(data_dir, "satellite01.obj"), split_mode="OFF"
)
satellite_obj = bpy.context.selected_objects[0]
satellite_obj.rotation_euler = Euler((math.radians(90), 0, 0), "ZXY")
bpy.ops.transform.translate(
    value=(0, 0, 1.7),
    orient_type="GLOBAL",
    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    orient_matrix_type="GLOBAL",
    constraint_axis=(False, False, True),
    mirror=True,
    use_proportional_edit=False,
    proportional_edit_falloff="SMOOTH",
    proportional_size=1,
    use_proportional_connected=False,
    use_proportional_projected=False,
)
bpy.ops.transform.resize(
    value=(0.025, 0.025, 0.025),
    orient_type="GLOBAL",
    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    orient_matrix_type="GLOBAL",
    mirror=True,
    use_proportional_edit=False,
    proportional_edit_falloff="SMOOTH",
    proportional_size=1,
    use_proportional_connected=False,
    use_proportional_projected=False,
)

satellite_empty = bpy.context.selected_objects[0]
bpy.ops.object.empty_add(
    type="ARROWS", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
)
satellite_empty = bpy.context.selected_objects[0]
satellite_obj.parent = satellite_empty

bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
bpy.context.scene.cursor.rotation_euler = Vector((0.0, 0.0, 0.0))
bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")

track = np.genfromtxt(
    os.path.join(data_dir, "satellite_tracks", f"track.csv"), delimiter=","
)
track_angles = np.genfromtxt(
    os.path.join(data_dir, "satellite_tracks", f"track_angles.csv"), delimiter=","
)

track_objs = []
for i in range(len(track)):
    vf_bm = import_vector_field(
        os.path.join(data_dir, "satellite_tracks", f"vec_{i}.csv")
    )
    vf_obj = add_vector_field(vf_bm, arr_obj, scale=3, name=f"vec_{i}")
    vf_obj.scale = (0, 0, 0)
    track_objs.append(vf_obj)

bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
empty = bpy.context.selected_objects[0]
earth_obj.parent = empty
for obj in track_objs: 
    obj.parent = empty

empty.rotation_euler = Euler((0,0,math.radians(90)), "XYZ")

# set_object_collections(object=[earth_obj, satellite_obj, *track_objs])


# for i in range(len(track)):
max_frame = 0
frames_per_track = 1
for i in range(len(track) - 1):
    frame = (frames_per_track * i) + 1
    for j in range(frames_per_track):
        satellite_empty.rotation_euler = Euler(
            (
                0,
                ((frames_per_track - j) / frames_per_track) * track_angles[i, 0]
                + (j / frames_per_track) * track_angles[i + 1, 0],
                ((frames_per_track - j) / frames_per_track) * track_angles[i, 1]
                + (j / frames_per_track) * track_angles[i + 1, 1]
                + 1.5708,
            ),
            "YZX",
        )
        satellite_empty.keyframe_insert(data_path="rotation_euler", frame=frame)
    track_objs[i].scale = (1, 1, 1)
    for obj in track_objs:
        obj.keyframe_insert(data_path="scale", frame=frame)
    max_frame = frame


bpy.context.scene.render.filepath = os.path.join(
    data_dir, "satellite_tracks", "renders", "frame"
)
set_resolution(1080)
bpy.context.scene.frame_end = max_frame
bpy.context.scene.frame_current = bpy.context.scene.frame_start
bpy.ops.render.render(use_viewport=True, animation=True, write_still=True)
