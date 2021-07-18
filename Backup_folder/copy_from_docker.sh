#!/bin/bash

# Check docker id:
docker ps
# copy_from docker
file_name="name.png"
docker cp 144991bc1320:/home1/danpardo@staff.technion.ac.il/project/$file_name $file_name

