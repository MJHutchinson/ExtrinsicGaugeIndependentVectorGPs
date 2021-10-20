base_dir=~/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs

for img in right_front right_back right_top right_bottom wrong_front wrong_back wrong_top wrong_bottom
do
    convert $base_dir/blender_outputs/$img.png -crop 960x540+480+270 $base_dir/blender_outputs/$img-crop.png
done
