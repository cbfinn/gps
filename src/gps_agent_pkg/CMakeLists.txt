cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

#Include protobuf
include_directories($ENV{GPS_ROOT_DIR}/build/gps)

#Include boost
rosbuild_add_boost_directories()
#find_package(Boost 1.46.0 COMPONENTS)
#if(Boost_FOUND)
    #include_directories(${Boost_INCLUDE_DIRS})
#endif()

#Use C++0x
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(DDP_FILES src/robotplugin.cpp
              src/pr2plugin.cpp
              src/sample.cpp
              src/sensor.cpp
              src/neuralnetwork.cpp
              src/tfcontroller.cpp
              src/controller.cpp
              src/lingausscontroller.cpp
              src/camerasensor.cpp
              src/positioncontroller.cpp
              src/trialcontroller.cpp
              src/encodersensor.cpp
              src/encoderfilter.cpp
              src/util.cpp)

# Include Caffe
if (USE_CAFFE)
    # add definitions for the C++ code
    add_definitions(-DUSE_CAFFE)
    if (USE_CAFFE_GPU)
        add_definitions(-DCAFFE_GPU)
        find_package(CUDA 5.5 REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
    else (USE_CAFFE_GPU)
        add_definitions(-DCPU_ONLY)
    endif (USE_CAFFE_GPU)
    # add paths to CAFFE
    link_directories(${CAFFE_LIBRARY_PATH})  # Should be set to /path/to/caffe/build/lib
    include_directories(${CAFFE_INCLUDE_PATH})  # Should be set to /path/to/caffe/distribute/include
    # extra header paths
    include_directories(${CAFFE_EXTRA_PATH})
    # add neural network to DDP controller files
    set(DDP_FILES ${DDP_FILES} src/neuralnetworkcaffe.cpp src/caffenncontroller.cpp)
    # compile Caffe test
    #rosbuild_add_executable(caffe_test src/caffe_test.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_test caffe protobuf)

    # compile image processor node
    #rosbuild_add_executable(caffe_img_processor src/img_processor.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_img_processor caffe protobuf)
endif (USE_CAFFE)

rosbuild_add_library(gps_agent_lib ${DDP_FILES})
#uncomment for test executable
#rosbuild_add_executable(main src/main.cpp)

#rosbuild_add_executable(kinematic_baseline src/kinematic_baseline.cpp)

#rosbuild_add_executable(controller_switcher src/controller_switcher.cpp)

#rosbuild_add_executable(pointcloud_solver src/pointcloud_solver.cpp
#                                          src/keypoint_detector.cpp)

#rosbuild_add_executable(point_head src/point_head.cpp)
#rosbuild_add_executable(torso src/torso.cpp)

# Include Caffe in controller
if (USE_CAFFE)
    target_link_libraries(gps_agent_lib caffe protobuf)
endif (USE_CAFFE)

rosbuild_link_boost(gps_agent_lib thread)# smart_ptr)
