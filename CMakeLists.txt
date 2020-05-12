# 声明CMake的最低要求版本
cmake_minimum_required(VERSION 2.8)

PROJECT(rcll)
add_compile_options(-std=c++11)
SET(CMAKE_BUILD_TYPE "Release")

# 外部依赖项
# opencv
find_package(OpenCV 3.3.1 EXACT REQUIRED
            PATHS /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
include_directories(${OpenCV_INCLUDE_DIRS})
SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${OpenCV_LIBS})

# glog
find_package(glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})
SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} glog::glog)

# nanoflann
find_package(nanoflann)
include_directories(${NANOFLANN_INCLUDE_DIRS})
SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} nanoflann::nanoflann)

# deal with shapelib
# shplib
SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${PROJECT_SOURCE_DIR}/third_lib/lib/shp/libshp.so ${PROJECT_SOURCE_DIR}/third_lib/lib/shp/libshp.so.1)
include_directories(${PROJECT_SOURCE_DIR}/third_lib/include/shp/  )

# 添加头文件的查找目录
include_directories(${PROJECT_SOURCE_DIR}/include/basic_structure)
include_directories(${PROJECT_SOURCE_DIR}/include/cross_point_detect)
include_directories(${PROJECT_SOURCE_DIR}/include/locator)
include_directories(${PROJECT_SOURCE_DIR}/include/voronoi)
include_directories(${PROJECT_SOURCE_DIR}/include/shp)
include_directories(${PROJECT_SOURCE_DIR}/include/icp)

# 添加参与编译的子目录,会直接跳转到子目录，因此这一句应该放在所有需要继承的变量已经被设置后
ADD_SUBDIRECTORY(src)
