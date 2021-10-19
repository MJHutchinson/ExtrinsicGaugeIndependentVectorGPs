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

os.makedirs(os.path.join(data_dir, 'blank_to_wrong', 'renders'), exist_ok=True)


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
arr_obj = create_vector_arrow(color=(1, 0, 0, 1))
# set_object_collections(backdrop=[bd_obj], instancing=[arr_obj])

bm = import_bmesh(os.path.join(data_dir, "unwrap_sphere", "frame_59.obj"))
import_color(bm, name='white', color = (1,1,1,1))
import_color(bm, name='s_wrong', data_file = os.path.join(data_dir, "kernels", "s_wrong.csv"), palette_file = os.path.join(col_dir, "viridis.csv"))
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))

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


# VECTOR FIELD
vf_bm = import_vector_field(
    os.path.join(data_dir, "unwrap_sphere", f"frame_59.csv")
)
vf_obj = add_vector_field(
    vf_bm, arr_obj, scale=3, name="observations"
)
bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["observations"].select_set(True)
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

arr_obj = create_vector_arrow(color=(0.3,0.3,0.3, 1.0)) # (0.0, 0.75, 1.0, 1)
mean_bm = import_vector_field(
    os.path.join(data_dir, "unwrap_sphere", f"frame_59.csv")
)
mean_bm = import_vector_field(
    os.path.join(data_dir, "kernels", f"mean_zero.csv"), name="_zero"
)
mean_bm = import_vector_field(
    os.path.join(data_dir, "kernels", f"mean_wrong.csv"), bm=mean_bm, name="_wrong"
)
mean_obj = add_vector_field(
    mean_bm, arr_obj, scale=3, name="means"
)
vec_fraction_node = mix_geometry_attributes(mean_obj, ['arrow', 'normal_x', 'normal_z'], '_zero', '_wrong')
bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["means"].select_set(True)
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
    data_dir, "blank_to_wrong", "renders", f"frame_"
)

bpy.ops.render.render(use_viewport=True, animation=True, write_still=True)

# # CHANGE COLORS 
# earth_obj.data.materials.pop(index=0)
# bm.to_mesh(earth_obj.data)
# # earth_obj = add_mesh(bm, name="Earth")
# earth_mat = add_vertex_colors(earth_obj)
# add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))

# earth_obj.keyframe_insert(data_path="materials", frame=30)
