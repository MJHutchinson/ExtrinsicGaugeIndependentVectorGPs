base_dir=~/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs

function pull_endslate {
    cp $base_dir/blender/$1/renders/$( ls $base_dir/blender/$1/renders -1 | tail -n 1) $base_dir/blender_outputs/$1_endslate.png
}

cp $base_dir/blender/kernels/renders/* $base_dir/blender_outputs

pull_endslate satellite_tracks
pull_endslate unwrap_sphere
pull_endslate blank_to_wrong
pull_endslate sample_sphere
pull_endslate project_to_sphere
pull_endslate wrong_to_right
pull_endslate rewrap_sphere