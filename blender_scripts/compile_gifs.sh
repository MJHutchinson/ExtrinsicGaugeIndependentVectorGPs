base_dir=~/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs

function compile_gif {
    gifski -o $base_dir/blender_outputs/$1.gif -Q 100 -r 15 -W 1920 -H 1080 --repeat -$2 $base_dir/blender/$1/renders/frame*.png
}

# compile_gif satellite_tracks 1
# compile_gif unwrap_sphere 1
# compile_gif blank_to_wrong 1
# compile_gif sample_sphere 1
# compile_gif project_to_sphere 1
# compile_gif wrong_to_right 1
# compile_gif rewrap_sphere 1
compile_gif wind_plots 0
# gifski -o $base_dir/blender_outputs/rewrap_sphere.gif -Q 100 -r 30 -W 1920 -H 1080 --repeat -1 --nosort $(ls -d $base_dir/blender/rewrap_sphere/renders/* | sort --reverse)



# gifski -o $base_dir/blender_outputs/$frame_path.gif -Q 100 -r 30 -W 1920 -H 1080 --repeat -0 $base_dir/blender/$frame_path/renders/frame*.png
