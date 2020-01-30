#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2019 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# This file is part of the Media Processing Framework (MPF)                 #
# motion detection component software.                                      #
#                                                                           #
# The MPF motion detection component software is free software: you can     #
# redistribute it and/or modify it under the terms of the GNU General       #
# Public License as published by the Free Software Foundation, either       #
# version 3 of the License, or (at your option) any later version.          #
#                                                                           #
# The MPF motion detection component software is distributed in the hope    #
# that it will be useful, but WITHOUT ANY WARRANTY; without even the        #
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  #
# See the (GNU General Public License for more details.                     #
#                                                                           #
# You should have received a copy of the GNU General Public License         #
# along with the MPF motion detection component software. If not, see       #
# <http://www.gnu.org/licenses/>.                                           #
#############################################################################


cmake_minimum_required(VERSION 3.6)
project(mog-motion-detection)

set(CMAKE_CXX_STANDARD 11)

include(../../ComponentSetup.cmake)

find_package(OpenCV 3.4.7 EXACT REQUIRED PATHS /opt/opencv-3.4.7
    COMPONENTS opencv_objdetect opencv_ml opencv_video opencv_highgui)

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)

include(../STRUCK/CMakeLists.txt)

set(MOG_MOTION_SOURCE_FILES MotionDetection_MOG2.cpp MotionDetection_MOG2.h)

add_library(mpfMogMotionDetection SHARED ${MOG_MOTION_SOURCE_FILES})
target_link_libraries(mpfMogMotionDetection STRUCK ${OpenCV_LIBS} mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils)


configure_mpf_component(MogMotionDetection TARGETS mpfMogMotionDetection)


add_subdirectory(test)

add_executable(sample_mog_motion_detector sample_mog_motion_detector.cpp)
target_link_libraries(sample_mog_motion_detector mpfMogMotionDetection)

# Make plugin files available to sample binary
execute_process(
    COMMAND sh -c "ln --symbolic --force plugin/MogMotionDetection/* ."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
