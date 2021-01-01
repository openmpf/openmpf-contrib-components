#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# This file is part of the Media Processing Framework (MPF)                 #
# person detection component software.                                      #
#                                                                           #
# The MPF person detection component software is free software: you can     #
# redistribute it and/or modify it under the terms of the GNU General       #
# Public License as published by the Free Software Foundation, either       #
# version 3 of the License, or (at your option) any later version.          #
#                                                                           #
# The MPF person detection component software is distributed in the hope    #
# that it will be useful, but WITHOUT ANY WARRANTY; without even the        #
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  #
# See the (GNU General Public License for more details.                     #
#                                                                           #
# You should have received a copy of the GNU General Public License         #
# along with the MPF person detection component software. If not, see       #
# <http://www.gnu.org/licenses/>.                                           #
#############################################################################


cmake_minimum_required(VERSION 3.6)
project(ocv-person-detection)

set(CMAKE_CXX_STANDARD 11)

include(../ComponentSetup.cmake)

find_package(OpenCV 4.5.0 EXACT REQUIRED PATHS /opt/opencv-4.5.0
    COMPONENTS opencv_objdetect opencv_ml opencv_video opencv_highgui)
find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)


set(OCV_PERSON_DETECTION_SOURCE_FILES
    PersonDetection.cpp PersonDetection.h
)

set(TRACKING_SOURCE_FILES
    tracking/appTemplate.cpp tracking/appTemplate.h
    tracking/detector.cpp tracking/detector.h
    tracking/multiTrackAssociation.cpp tracking/multiTrackAssociation.h
    tracking/munkres.cpp tracking/munkres.h
    tracking/tracker.cpp tracking/tracker.h
)

add_library(mpfOcvPersonDetection SHARED ${OCV_PERSON_DETECTION_SOURCE_FILES} ${TRACKING_SOURCE_FILES})
target_link_libraries(mpfOcvPersonDetection mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils
    ${OpenCV_LIBS})


configure_mpf_component(OcvPersonDetection TARGETS mpfOcvPersonDetection)

add_subdirectory(test)

add_executable(sample_ocv_person_detector sample_ocv_person_detector.cpp)
target_link_libraries(sample_ocv_person_detector mpfOcvPersonDetection)
