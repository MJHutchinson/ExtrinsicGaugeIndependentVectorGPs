for img in $1/frame*; do
    # echo $img.bck
    # sed '/\//bck' $im
    mv $img $img.bck
    convert $img.bck -background white -alpha remove -alpha off $img
done