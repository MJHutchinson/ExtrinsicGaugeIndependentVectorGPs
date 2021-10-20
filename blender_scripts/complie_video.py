import os
import sys

import bpy
from bpy import context
scene = context.scene

frame_path = str(sys.argv[-2])
reverse_frames = bool(int(sys.argv[-1]))

video_path = "/home/mjhutchinson/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender_outputs"

# frame_path = "/home/mjhutchinson/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/blender/blank_to_wrong/renders"
# reverse_frames = False

name = os.path.split(os.path.split(frame_path)[0])[1]

files = os.listdir(frame_path)
files.sort()
files = [f for f in files if f.endswith('.png')]

scene.sequence_editor_create()

seq = scene.sequence_editor.sequences.new_image(
        name="MyStrip",
        filepath=os.path.join(frame_path, files[0]),
        channel=1, frame_start=1)

for f in files[1:]:
    seq.elements.append(f)

seq.use_reverse_frames = reverse_frames

render = scene.render
scene.frame_start = 1
scene.frame_end = len(files)

render.resolution_x = 1920
render.resolution_y = 1080
# render.resolution_x = 2250
# render.resolution_y = 1500
render.fps = 30

render.image_settings.file_format = 'FFMPEG'
render.ffmpeg.format = 'MPEG4'
render.ffmpeg.codec = 'H264'

render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS'
render.ffmpeg.ffmpeg_preset = 'BEST'

render.use_file_extension = False
render.filepath = os.path.join(video_path, name + ".mp4")

bpy.ops.render.render(animation=True)