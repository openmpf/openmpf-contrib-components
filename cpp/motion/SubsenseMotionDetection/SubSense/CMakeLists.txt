project(SubSense)

find_package(OpenCV $ENV{OPENCV_VERSION} EXACT REQUIRED PATHS /opt/opencv-$ENV{OPENCV_VERSION}
    COMPONENTS opencv_features2d)

aux_source_directory(. SRC)
include_directories(.)

add_library(SubSense SHARED ${SRC})
target_link_libraries(SubSense ${OpenCV_LIBS})


