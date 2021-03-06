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


cmake_minimum_required(VERSION 3.6)
project(mog-motion-detection-tests)

set(CMAKE_CXX_STANDARD 11)

find_package(GTest)
if (${GTEST_FOUND})

    enable_testing()
    find_package(Qt4 REQUIRED)
    find_package(mpfComponentTestUtils REQUIRED)

    include_directories(..)

    add_executable(MogMotionDetectionTest test_mog_motion_detection.cpp)
    target_link_libraries(MogMotionDetectionTest
        mpfMogMotionDetection mpfComponentTestUtils
        GTest::GTest GTest::Main Qt4::QtCore)

    add_test(NAME MogMotionDetectionTest COMMAND MogMotionDetectionTest)

    # Install config file
    file(COPY test_mog_motion_config.ini DESTINATION config)

    # Install test videos
    file(GLOB MOG_TEST_VIDEO_FILES data/test_vids/*)
    file(COPY ${MOG_TEST_VIDEO_FILES} DESTINATION test/test_vids)

    # Create directory to run tests
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/test_output/)

endif()