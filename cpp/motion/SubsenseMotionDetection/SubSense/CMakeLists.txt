project(SubSense)

find_package(OpenCV 4.5.5 EXACT REQUIRED PATHS /opt/opencv-4.5.5
    COMPONENTS opencv_features2d)

aux_source_directory(. SRC)
include_directories(.)

add_library(SubSense SHARED ${SRC})
target_link_libraries(SubSense ${OpenCV_LIBS})


