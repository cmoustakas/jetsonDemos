# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

# declare my-recognition project
project(stitch)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

pkg_search_module(GST REQUIRED gstreamer-1.0>=1.4
    gstreamer-sdp-1.0>=1.4
    gstreamer-video-1.0>=1.4
    gstreamer-app-1.0>=1.4)


# CUDA and Qt4 are required
find_package(CUDA)
find_package(Qt4)

# setup Qt4 for build
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# compile the my-recognition program
cuda_add_executable( stitch stitch.cu)

# link my-recognition to jetson-inference library
target_include_directories(stitch PRIVATE ${GST_INCLUDE_DIRS})
target_link_libraries( stitch jetson-inference )
target_link_libraries(stitch ${GST_LIBRARIES})
target_link_libraries(stitch ${OpenCV_LIBS})
target_compile_options(stitch PRIVATE -02 -g -pg)

