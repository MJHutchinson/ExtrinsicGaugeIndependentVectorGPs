import bpy
import bmesh
import math
from functools import partial
import numpy as np
import os
from mathutils import Vector
from mathutils import Euler

# directory = os.getcwd()
scripts_dir = "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender_scripts"
data_dir = (
    "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender"
)
texture_path = (
    "/home/mhutchin/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/textures"
)

with open(os.path.join(scripts_dir, "render.py")) as file:
    exec(file.read())

reset_scene()
# set_renderer_settings(num_samples=2048 if bpy.app.background else 128)
set_renderer_settings(num_samples=16 if bpy.app.background else 128)
# setup_layers()
# setup_compositor(
#     mask_center=(0.5, 0.3125),
#     mask_size=(0.675, 0.325),
#     shadow_color_correction_exponent=2.75,
# )
(cam_axis, cam_obj) = setup_camera(
    distance=15.5,
    # angle=(-np.pi / 16, 0, 0),
    angle=(0, 0, 0),
    lens=85,
    # height=2560,
    height=1500,
    # crop=(1 / 5, 9 / 10, 0, 10 / 11),
)
setup_lighting(
    shifts=(-10, -10, 10),
    sizes=(9, 18, 15),
    energies=(1500, 150, 1125),
    horizontal_angles=(-np.pi / 6, np.pi / 3, np.pi / 3),
    vertical_angles=(-np.pi / 3, -np.pi / 6, np.pi / 4),
)
bd_obj = create_backdrop(location=(0, 0, -2), scale=(10, 5, 5))
arr_obj = create_vector_arrow(color=(1, 0, 0, 1))

bm = import_bmesh(os.path.join(data_dir, "unwrap_sphere", "frame_0.obj"))
earth_obj = add_mesh(bm, name="Earth")
bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["Earth"].select_set(True)
ov=bpy.context.copy()
ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]
bpy.ops.transform.rotate(
    ov,
    value=1.5708,
    orient_axis="Z",
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

earth_mat = add_base_color(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))

bpy.ops.import_scene.obj(
    filepath=os.path.join(data_dir, "satellite01.obj"), split_mode="OFF"
)
satellite_obj = bpy.context.selected_objects[0]
satellite_obj.rotation_euler = Euler((math.radians(90), 0, 0), "ZXY")
bpy.ops.transform.translate(
    value=(0, 0, 2),
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
    value=(0.05, 0.05, 0.05),
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
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[f"vec_{i}"].select_set(True)
    ov=bpy.context.copy()
    ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]
    bpy.ops.transform.rotate(
        ov,
        value=1.5708,
        orient_axis="Z",
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
    vf_obj.scale = (0, 0, 0)
    track_objs.append(vf_obj)


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
# for modifier in vf_obj.modifiers:
#     bpy.data.node_groups.remove(modifier.node_group, do_unlink=True)
# bpy.data.objects.remove(obj, do_unlink=True)
# bpy.data.objects.remove(vf_obj, do_unlink=True)
# cleanup(objects=[obj, vf_obj], materials=[], modifiers=vf_obj.modifiers)
