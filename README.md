## Introduction
This is the implementation (version 2.0) for our paper "Road network based fast geolocalization", where the 
road networks in the aerial images are used to localize the images in the reference geographical 
coordinate system by registering the road map to the reference road map.

## Build
The project depends on the OpenCV, shapelib and the nanoflann. 
You can install the nanoflann following introduction in: https://github.com/jlblancoc/nanoflann.
And you should change the opencv path in the top cmakelist file to your own opencv path.
The shapelib is used to read .shp files, and convert them into images. We provide the lib in the directory ‘third_lib’.

After all the dependences have been install, you can build the project:

$ mkdir build

$ cd build

$ cmake ../

$ make ./

## Run
1. Detect all road intersections and compute their tangents with:

    $ ./make_reference_map [shp_file] [cross_file_name]
    
    shp_file --- the reference road vector map save in .shp format
    
    cross_file_name --- the file to save the road intersections and their tangents

2. Compute the Voronoi surface for reference road map

    $ ./shp_to_voronoi [shp_file] [info_file] [tile_img_path] [gsd] [tile_size]

    [shp_file] --- the reference road vector map save in .shp format

    [info_file] [tile_img_path] --- the file to store the Voronoi surface

    [gsd] --- the gsd of the Voronoi

    [tile_size] --- the maximum size of each Voronoi image


3. Perform geolocalization with:

    $ ./single_img_localization [map_file_name] [ref_vor_file] [shp_file] [query_img_dir] [result_file]

    [map_file_name] --- the file to save the road intersections, which can be get with step one above

    [ref_vor_file] --- The files where the Voronoi surface of the reference road map are stored
    
    [shp_file] --- The reference road map in the .shp file

    [query_img_dir] --- the directory where the query road map are 

    [result_file] --- the file to save the geolocalization result
