import bpy
import bmesh
from functools import partial
import numpy as np
import os
from mathutils import Vector


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
setup_layers()
setup_compositor(
    mask_center=(0.5, 0.3125),
    mask_size=(0.675, 0.325),
    shadow_color_correction_exponent=2.75,
)
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
set_object_collections(backdrop=[bd_obj], instancing=[arr_obj])


for frame in [0, 29, 59]:
    print(frame)
    frame_name = f"frame_{frame}"
    bm = import_bmesh(os.path.join(data_dir, "unwrap_sphere", f"{frame_name}.obj"))
    vf_bm = import_vector_field(
        os.path.join(data_dir, "unwrap_sphere", f"{frame_name}.csv")
    )
    obj = add_mesh(bm, name=frame_name)
    earth_mat = add_base_color(obj)
    add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[frame_name].select_set(True)
    bpy.ops.transform.rotate(
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
    # bpy.ops.transform.resize(
    #     value=(0.580459, 0.580459, 0.580459),
    #     orient_type="GLOBAL",
    #     orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    #     orient_matrix_type="GLOBAL",
    #     mirror=True,
    #     use_proportional_edit=False,
    #     proportional_edit_falloff="SMOOTH",
    #     proportional_size=1,
    #     use_proportional_connected=False,
    #     use_proportional_projected=False,
    # )
    # obj.shape_key_add(from_mix=False)
    # bpy.data.objects[frame_name].data.shape_keys.key_blocks["Key"].name = frame_name
    vf_obj = add_vector_field(
        vf_bm, arr_obj, scale=3, name=frame_name + "_vector_field"
    )
    # vf_obj.shape_key_add(from_mix=False)
    # bpy.data.objects[frame_name + "_vector_field"].data.shape_keys.key_blocks[
    #     "Key"
    # ].name = frame_name
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[frame_name + "_vector_field"].select_set(True)
    bpy.ops.transform.rotate(
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
    # bpy.ops.transform.resize(
    #     value=(0.580459, 0.580459, 0.580459),
    #     orient_type="GLOBAL",
    #     orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    #     orient_matrix_type="GLOBAL",
    #     mirror=True,
    #     use_proportional_edit=False,
    #     proportional_edit_falloff="SMOOTH",
    #     proportional_size=1,
    #     use_proportional_connected=False,
    #     use_proportional_projected=False,
    # )
    # vf_obj.data.materials.append(red_mat)
    # bpy.ops.object.duplicates_make_real()
    set_object_collections(object=[obj, vf_obj])
    bpy.context.scene.render.filepath = os.path.join(
        data_dir, "unwrap_sphere", "renders", f"frame_{frame}.png"
    )
    set_resolution(480)
    # bpy.ops.render.render(use_viewport=True, write_still=True)
    # for modifier in vf_obj.modifiers:
    #     bpy.data.node_groups.remove(modifier.node_group, do_unlink=True)
    # bpy.data.objects.remove(obj, do_unlink=True)
    # bpy.data.objects.remove(vf_obj, do_unlink=True)
    # cleanup(objects=[obj, vf_obj], materials=[], modifiers=vf_obj.modifiers)
