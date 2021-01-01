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

project(STRUCK)

if(NOT TARGET STRUCK)
    find_package(Qt4 REQUIRED)
    find_package(OpenCV 4.5.0 EXACT REQUIRED PATHS /opt/opencv-4.5.0 COMPONENTS opencv_core opencv_imgproc)

    # Need to use absolute paths so that both Mog and Subsense can build STRUCK
    set(STRUCK_SRC
        ${CMAKE_CURRENT_LIST_DIR}/Config.h
        ${CMAKE_CURRENT_LIST_DIR}/Config.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Features.h
        ${CMAKE_CURRENT_LIST_DIR}/Features.cpp
        ${CMAKE_CURRENT_LIST_DIR}/HaarFeature.h
        ${CMAKE_CURRENT_LIST_DIR}/HaarFeature.cpp
        ${CMAKE_CURRENT_LIST_DIR}/HaarFeatures.h
        ${CMAKE_CURRENT_LIST_DIR}/HaarFeatures.cpp
        ${CMAKE_CURRENT_LIST_DIR}/HistogramFeatures.h
        ${CMAKE_CURRENT_LIST_DIR}/HistogramFeatures.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ImageRep.h
        ${CMAKE_CURRENT_LIST_DIR}/ImageRep.cpp
        ${CMAKE_CURRENT_LIST_DIR}/LaRank.h
        ${CMAKE_CURRENT_LIST_DIR}/LaRank.cpp
        ${CMAKE_CURRENT_LIST_DIR}/MultiFeatures.h
        ${CMAKE_CURRENT_LIST_DIR}/MultiFeatures.cpp
        ${CMAKE_CURRENT_LIST_DIR}/RawFeatures.h
        ${CMAKE_CURRENT_LIST_DIR}/RawFeatures.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Sampler.h
        ${CMAKE_CURRENT_LIST_DIR}/Sampler.cpp
        ${CMAKE_CURRENT_LIST_DIR}/struck.h
        ${CMAKE_CURRENT_LIST_DIR}/struck.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Tracker.h
        ${CMAKE_CURRENT_LIST_DIR}/Tracker.cpp)

    add_library(STRUCK SHARED ${STRUCK_SRC} ${CMAKE_CURRENT_LIST_DIR}/GraphUtils/GraphUtils.cpp)
    target_link_libraries(STRUCK ${OpenCV_LIBS} Qt4::QtCore)
    target_include_directories(STRUCK PUBLIC ${CMAKE_CURRENT_LIST_DIR}/GraphUtils ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/eigen)
endif()

