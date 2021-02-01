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
project(ocv-person-detection-tests)

set(CMAKE_CXX_STANDARD 11)

find_package(GTest)
if (${GTEST_FOUND})

    enable_testing()
    find_package(Qt4 REQUIRED)
    find_package(mpfComponentTestUtils REQUIRED)

    include_directories(..)

    add_executable(OcvPersonDetectionTest test_ocv_person_detection.cpp)
    target_link_libraries(OcvPersonDetectionTest mpfComponentTestUtils mpfOcvPersonDetection GTest::GTest GTest::Main Qt4::QtCore)

    add_test(NAME OcvPersonDetectionTest COMMAND OcvPersonDetectionTest)

    # Install config file
    file(COPY test_ocv_person_config.ini DESTINATION config)

    # Install test images and videos
    file(GLOB PERSON_TEST_IMAGE_FILES data/test_imgs/*)
    file(COPY ${PERSON_TEST_IMAGE_FILES} DESTINATION test/test_imgs)

    file(GLOB PERSON_TEST_VIDEO_FILES data/test_vids/*)
    file(COPY ${PERSON_TEST_VIDEO_FILES} DESTINATION test/test_vids)

    # Create directory to run tests
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/test_output/)

endif()