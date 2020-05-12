set(SOURCE_FILE_CROSS_POINT_DETECT ${SOURCE_FILE_CROSS_POINT_DETECT} ${PROJECT_SOURCE_DIR}/src/cross_point_detect/cross_line_extractor.cc)
set(SOURCE_FILE_CROSS_POINT_DETECT ${SOURCE_FILE_CROSS_POINT_DETECT} ${PROJECT_SOURCE_DIR}/src/cross_point_detect/cross_point_feature_creator.cc)
set(SOURCE_FILE_CROSS_POINT_DETECT ${SOURCE_FILE_CROSS_POINT_DETECT} ${PROJECT_SOURCE_DIR}/src/cross_point_detect/cross_points_detector.cc)

set(SOURCE_FILE_BASIC_STRUCTURE ${SOURCE_FILE_BASIC_STRUCTURE} ${PROJECT_SOURCE_DIR}/src/basic_structure/cross_point_tree.cc)
set(SOURCE_FILE_BASIC_STRUCTURE ${SOURCE_FILE_BASIC_STRUCTURE} ${PROJECT_SOURCE_DIR}/src/basic_structure/cross_point.cc)
set(SOURCE_FILE_BASIC_STRUCTURE ${SOURCE_FILE_BASIC_STRUCTURE} ${PROJECT_SOURCE_DIR}/src/basic_structure/cross_pt_tuple.cc)
set(SOURCE_FILE_BASIC_STRUCTURE ${SOURCE_FILE_BASIC_STRUCTURE} ${PROJECT_SOURCE_DIR}/src/basic_structure/cross_ratio.cc)
set(SOURCE_FILE_BASIC_STRUCTURE ${SOURCE_FILE_BASIC_STRUCTURE} ${PROJECT_SOURCE_DIR}/src/basic_structure/search_tree.cc)


set(SOURCE_FILE_LOCATOR ${SOURCE_FILE_LOCATOR} ${PROJECT_SOURCE_DIR}/src/locator/estimator.cc)     
set(SOURCE_FILE_LOCATOR ${SOURCE_FILE_LOCATOR} ${PROJECT_SOURCE_DIR}/src/locator/locator.cc)
set(SOURCE_FILE_LOCATOR ${SOURCE_FILE_LOCATOR} ${PROJECT_SOURCE_DIR}/src/locator/sampler.cc)
set(SOURCE_FILE_LOCATOR ${SOURCE_FILE_LOCATOR} ${PROJECT_SOURCE_DIR}/src/locator/similarity_evaluator.cc)
set(SOURCE_FILE_LOCATOR ${SOURCE_FILE_LOCATOR} ${PROJECT_SOURCE_DIR}/src/locator/random_grid.cc)
set(SOURCE_FILE_LOCATOR ${SOURCE_FILE_LOCATOR} ${PROJECT_SOURCE_DIR}/src/locator/consistency_checker.cc)

# shp
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/shp/draw_shp_elements.cc)
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/shp/get_shp_elements.cc)
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/shp/shp.cc)

# voronoi
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/voronoi/jc_voronoi.cc)
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/voronoi/voronoi_map.cc)
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/voronoi/voronoi_surface.cc)


# icp
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/icp/homography.cc)
set(SOURCE_FILE_SHP ${SOURCE_FILE_SHP} ${PROJECT_SOURCE_DIR}/src/icp/icp.cc)

set(SOURCE_FILE ${SOURCE_FILE} 
                ${SOURCE_FILE_LOCATOR} 
                ${SOURCE_FILE_BASIC_STRUCTURE} 
                ${SOURCE_FILE_CROSS_POINT_DETECT}
                ${SOURCE_FILE_SHP}
                ${SOURCE_FILE_SHP})





set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


# localization with single image
add_executable(single_img_localization ${SOURCE_FILE} ${PROJECT_SOURCE_DIR}/src/test/single_img_localization.cc)
TARGET_LINK_LIBRARIES(single_img_localization ${REQUIRED_LIBRARIES} )

# extract all road intersections for given shp file
add_executable(shp_to_map ${SOURCE_FILE} ${PROJECT_SOURCE_DIR}/src/test/shp_to_map.cc)
TARGET_LINK_LIBRARIES(shp_to_map ${REQUIRED_LIBRARIES} )

# generate voronoi surface for given shp file
add_executable(shp_to_voronoi ${SOURCE_FILE} ${PROJECT_SOURCE_DIR}/src/test/shp_to_voronoi.cc)
TARGET_LINK_LIBRARIES(shp_to_voronoi ${REQUIRED_LIBRARIES} )



