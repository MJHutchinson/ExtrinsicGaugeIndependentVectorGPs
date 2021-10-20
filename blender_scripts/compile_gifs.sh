base_dir=~/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs

function compile_gif {
    gifski -o $base_dir/blender_outputs/$1.gif -Q 100 -r 30 -W 1920 -H 1080 --repeat -1 $base_dir/blender/$1/renders/frame*.png
}

compile_gif satellite_tracks
compile_gif unwrap_sphere
compile_gif blank_to_wrong
compile_gif sample_sphere
compile_gif project_to_sphere
compile_gif wrong_to_right
# compile_gif rewrap_sphere
gifski -o $base_dir/blender_outputs/rewrap_sphere.gif -Q 100 -r 30 -W 1920 -H 1080 --repeat -1 --nosort $(ls -d $base_dir/blender/rewrap_sphere/renders/* | sort --reverse)