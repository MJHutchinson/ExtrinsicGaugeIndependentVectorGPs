import bpy
import bmesh
import math
from functools import partial
import numpy as np
import os
from mathutils import Vector
from mathutils import Euler

# directory = os.getcwd()
base_dir = "/home/mjhutchinson/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/"
scripts_dir = os.path.join(base_dir, "blender_scripts")
data_dir = os.path.join(base_dir, "blender")
texture_path = os.path.join(base_dir, "textures")
col_dir = os.path.join(base_dir, "col")

os.makedirs(os.path.join(data_dir, 'rewrap_sphere', 'renders'), exist_ok=True)

with open(os.path.join(scripts_dir, "render.py")) as file:
    exec(file.read())

reset_scene()
# set_renderer_settings(num_samples=2048 if bpy.app.background else 128)
set_renderer_settings(num_samples=128 if bpy.app.background else 128)
set_resolution(1080)
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
red_arr_obj = create_vector_arrow(color=(1, 0, 0, 1))
grey_arr_obj = create_vector_arrow(color=(0.3,0.3,0.3, 1.0))
# set_object_collections(backdrop=[bd_obj], instancing=[arr_obj])

frames = 60
for frame in range(frames):
    print(frame)
    frame_name = f"frame_{frames - frame - 1}"
    bm = import_bmesh(os.path.join(data_dir, "unwrap_sphere", f"{frame_name}.obj"))
    import_color(bm, data_file = os.path.join(data_dir, "kernels", "s_right.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
    obj = add_mesh(bm, name=frame_name)
    earth_mat = add_vertex_colors(obj)
    add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[frame_name].select_set(True)
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
    vf_bm = import_vector_field(
        os.path.join(data_dir, "rewrap_sphere", f"{frame_name}.csv")
    )
    mean_vf_obj = add_vector_field(
        vf_bm, grey_arr_obj, scale=3, name=frame_name + "_mean_field"
    )
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[frame_name + "_mean_field"].select_set(True)
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
    vf_bm = import_vector_field(
        os.path.join(data_dir, "unwrap_sphere", f"{frame_name}.csv")
    )
    print(os.path.join(data_dir, "unwrap_sphere", f"{frame_name}.csv"))
    track_vf_obj = add_vector_field(
        vf_bm, red_arr_obj, scale=3, name=frame_name + "_vector_field"
    )
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[frame_name + "_vector_field"].select_set(True)
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
    bpy.context.scene.render.filepath = os.path.join(
        data_dir, "rewrap_sphere", "renders", f"frame_{frames - frame - 1:04d}.png"
    )
    set_resolution(1080)
    bpy.ops.render.render(use_viewport=True, write_still=True)
    for modifier in mean_vf_obj.modifiers:
        bpy.data.node_groups.remove(modifier.node_group, do_unlink=True)
    for modifier in track_vf_obj.modifiers:
        bpy.data.node_groups.remove(modifier.node_group, do_unlink=True)
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.objects.remove(mean_vf_obj, do_unlink=True)
    bpy.data.objects.remove(track_vf_obj, do_unlink=True)
