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
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS  "-fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
# set(CMAKE_CXX_FLAGS "-std=c++11 -D GOOGLE_CUDA=1")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/venkatrn/research/caffe/cmake)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#   /home/venkatrn/third_party/tensorflow-cmake)
# set(CMAKE_COLOR_MAKEFILE ON)
set(ROS_COMPILE_FLAGS "-W -Wno-reorder -fPIC")

# find_package(Caffe REQUIRED)
# set(Caffe_DIR /home/venkatrn/research/caffe)
# set(Caffe_INCLUDE_DIRS /home/venkatrn/research/caffe/include/ /home/venkatrn/research/caffe/build/include/)
# set(Caffe_LIBRARIES /home/venkatrn/research/caffe/build/lib/libcaffe.so)

set(L2S_ROOT "/home/venkatrn/research/learningToSee")
include_directories(${L2S_ROOT}/include)
link_libraries(${L2S_ROOT}/build/liblearningToSee.a)

find_path(dart_ROOT NAMES lib/libdart.so src/tracker.h HINTS
  /home/venkatrn/research/dart)
set(dart_INCLUDE_DIRS ${dart_ROOT}/src)
set(dart_LIBRARY_DIRS ${dart_ROOT}/lib)
set(dart_LIBRARIES dart)
include_directories(${dart_ROOT}/src)
link_directories(${dart_ROOT}/lib)
link_libraries(${dart_ROOT}/lib/libdart.so)

set(YCB_ROOT "/home/venkatrn/research/ycb")
include_directories(${YCB_ROOT}/include)
link_libraries(${YCB_ROOT}/lib/libycb.so)

set(XFUSION_ROOT "/home/venkatrn/research/xfusion")
include_directories(${XFUSION_ROOT}/include)
link_libraries(${XFUSION_ROOT}/lib/libxFusion.so)

set(SUITESPARSE_INCLUDE_DIRS "/usr/include/suitesparse" CACHE PATH "suitesparse include directory")
set(SUITESPARSE_LIBRARIES "cholmod;metis")
include_directories(${SUITESPARSE_INCLUDE_DIRS})
link_libraries(${SUITESPARSE_LIBRARIES})

set(PROTOBUF_ROOT "/home/venkatrn/third_party/protobuf/build")
include_directories(${PROTOBUF_ROOT}/include)
link_libraries(${PROTOBUF_ROOT}/lib/libprotobuf.a)


# find_package(PCL 1.7 REQUIRED COMPONENTS common io)
# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARIES})
# add_definitions(${PCL_DEFINITIONS})

# set(OpenCV_DIR /home/venkatrn/research/OpenCV/opencv/build)
# set(OpenCV_INCLUDE_DIRS /home/venkatrn/research/OpenCV/opencv/build)

# Load catkin and all dependencies required for this package
# find_package(tensorflow_ros)
find_package(catkin REQUIRED COMPONENTS 
  cv_bridge 
  kinect_sim 
  perception_utils 
  pcl_ros 
  roscpp 
  roslib
  tensorflow_ros)
find_package(Boost COMPONENTS serialization thread mpi REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
# find_package(TensorFlow REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA)

find_package(OpenGL REQUIRED QUIET)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

find_package(PkgConfig QUIET)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
find_path(MLPACK_INCLUDE_DIRS
  NAMES core.hpp prereqs.hpp
  PATHS /usr/local/include/ /usr/local/include/mlpack/
)

# Find libmlpack.so (or equivalent) and store it in MLPACK_LIBRARY.
# If this example script were smarter, it would also find other
# dependencies of mlpack and store them in
# an MLPACK_LIBRARIES variable instead.
find_library(MLPACK_LIBRARIES
  NAMES mlpack
  PATHS /usr/lib64/ /usr/lib/
  /usr/local/lib64/ /usr/local/
)
include_directories(${MLPACK_INCLUDE_DIRS})

include_directories(include ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} 
  # ${TensorFlow_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIR} ${Pangolin_CONFIG_INCLUDE_DIR} 
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
include_directories(${Sophus_INCLUDE_DIR})
include_directories(MPI_INCLUDE_PATH)
include_directories( ${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})


include(CMakeForceCompiler)
CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})
# set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
# set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

catkin_package(
  CATKIN_DEPENDS 
    cv_bridge 
    kinect_sim 
    perception_utils 
    pcl_ros 
    roscpp 
    roslib
    tensorflow_ros
  DEPENDS 
    OpenCV 
    MPI 
    OpenMP 
    PCL 
    Boost
  INCLUDE_DIRS 
    include
    ${XFUSION_ROOT}/include
  LIBRARIES 
    ${PROJECT_NAME}
    ${dart_ROOT}/lib/libdart.so
    ${XFUSION_ROOT}/lib/libxFusion.so
  )

add_library(${PROJECT_NAME}
  # src/vertex_net.cpp
  src/pose_estimator.cpp
  src/pose_estimator_ext.cpp
  src/helpers.cpp)
#


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}  ${CUDA_LIBRARIES} ${Pangolin_LIBRARIES}
  ${ARMADILLO_LIBRARIES} ${MLPACK_LIBRARIES}
  ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} ${GLEW_LIBRARIES} ${dart_ROOT}/lib/libdart.so
  libtiff.so libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so libglog.so gflags)

add_executable(loader src/loader.cc)
target_link_libraries(loader ${catkin_LIBRARIES}
  )

# add_executable(net_driver src/net_driver.cpp)
# target_link_libraries(net_driver ${PROJECT_NAME})

add_executable(dataset_generator tools/dataset_generator.cpp)
target_link_libraries(dataset_generator ${PROJECT_NAME})

add_executable(run_lov_experiments tools/run_experiments.cpp)
target_link_libraries(run_lov_experiments ${PROJECT_NAME})

add_executable(run_vert_net tools/run_vert_net.cpp src/vertex_net.cpp)
target_link_libraries(run_vert_net ${PROJECT_NAME})

add_executable(test_main tools/test_main.cpp)
target_link_libraries(test_main ${PROJECT_NAME})

add_executable(evaluate_lov_results tools/evaluate_results.cpp)
target_link_libraries(evaluate_lov_results ${PROJECT_NAME})
