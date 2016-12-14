cmake_minimum_required(VERSION 2.8.3)
project(deep_rgbd_utils)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
# set(ROS_BUILD_TYPE RelWithDebInfo)
# set(ROS_BUILD_TYPE Debug)
set(ROS_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/venkatrn/research/caffe/cmake)
# set(CMAKE_COLOR_MAKEFILE ON)
set(ROS_COMPILE_FLAGS "-W -Wno-reorder")

# find_package(Caffe REQUIRED)
set(Caffe_DIR /home/venkatrn/research/caffe)
set(Caffe_INCLUDE_DIRS /home/venkatrn/research/caffe/include/ /home/venkatrn/research/caffe/build/include/)
set(Caffe_LIBRARIES /home/venkatrn/research/caffe/build/lib/libcaffe.so)

# find_package(PCL 1.7 REQUIRED COMPONENTS common io)
# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARIES})
# add_definitions(${PCL_DEFINITIONS})

# set(OpenCV_DIR /home/venkatrn/research/OpenCV/opencv/build)
# set(OpenCV_INCLUDE_DIRS /home/venkatrn/research/OpenCV/opencv/build)

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS 
  cv_bridge 
  kinect_sim 
  perception_utils 
  pcl_ros 
  roscpp 
  roslib)
find_package(Boost COMPONENTS serialization thread mpi REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA)

find_package(PkgConfig QUIET)

include(CMakeForceCompiler)
CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})
# set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
# set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(include ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS})
include_directories(MPI_INCLUDE_PATH)

catkin_package(
  CATKIN_DEPENDS 
    cv_bridge 
    kinect_sim 
    perception_utils 
    pcl_ros 
    roscpp 
    roslib 
  DEPENDS 
    # OpenCV 
    MPI 
    OpenMP 
    PCL 
    Boost
  INCLUDE_DIRS 
    include
  LIBRARIES 
    ${PROJECT_NAME}
  )

add_library(${PROJECT_NAME}
  src/feature_generator_net.cpp
  src/helpers.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES}
  ${PCL_LIBRARIES} ${Caffe_LIBRARIES} ${CUDA_LIBRARIES}
  libtiff.so libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so libglog.so)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES} ${Caffe_LIBRARIES} ${CUDA_LIBRARIES} libglog.so)

add_executable(vertex_net tools/pixel_vertex_mapper.cpp)
target_link_libraries(vertex_net ${PROJECT_NAME})
