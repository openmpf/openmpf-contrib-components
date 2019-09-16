project(SubSense)

find_package(OpenCV 3.3.1 EXACT REQUIRED PATHS /opt/opencv-3.3.1
    COMPONENTS opencv_features2d)

aux_source_directory(. SRC)
include_directories(.)

add_library(SubSense SHARED ${SRC})
target_link_libraries(SubSense ${OpenCV_LIBS})


