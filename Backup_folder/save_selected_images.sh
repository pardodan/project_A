#!/bin/bash
docker_id=`docker ps | tail -1 | cut -d " " -f1`
#for (( c=1; c<=$k; c++ ))
#do
file_name=test1.jpg
image_id=$1
#echo $image_id
dir_1=${image_id:0:1}
dir_2=${image_id:1:1}
dir_3=${image_id:2:1}
docker cp $docker_id:/home1/danpardo@staff.technion.ac.il/project/"$dir_1"/"$dir_2"/"$dir_3"/"$image_id".jpg /home/danpardo@staff.technion.ac.il/view_images/"$2".jpg

#done

