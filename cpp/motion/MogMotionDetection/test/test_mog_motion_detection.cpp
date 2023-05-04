/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * This file is part of the Media Processing Framework (MPF)                  *
 * motion detection component software.                                       *
 *                                                                            *
 * The MPF motion detection component software is free software: you can      *
 * redistribute it and/or modify it under the terms of the GNU General        *
 * Public License as published by the Free Software Foundation, either        *
 * version 3 of the License, or (at your option) any later version.           *
 *                                                                            *
 * The MPF motion detection component software is distributed in the hope     *
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the         *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.   *
 * See the (GNU General Public License for more details.                      *
 *                                                                            *
 * You should have received a copy of the GNU General Public License          *
 * along with the MPF motion detection component software. If not, see        *
 * <http://www.gnu.org/licenses/>.                                            *
 ******************************************************************************/

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <log4cxx/basicconfigurator.h>

#include <MPFDetectionComponent.h>

#include "MotionDetection_MOG2.h"

using namespace MPF;
using namespace COMPONENT;


bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();


TEST(VideoGeneration, TestOnKnownVideo) {
    int start = 0;
    int stop = 199;
    std::string inVideoFile = "test/test_vids/ff-region-motion-face.avi";

    std::cout << "Start:\t" << start << std::endl;
    std::cout << "Stop:\t" << stop << std::endl;
    std::cout << "inVideo:\t" << inVideoFile << std::endl;

    std::cout << "\tCreating a MOG Motion Detector" << std::endl;
    MotionDetection_MOG2 motion_detection;
    motion_detection.SetRunDirectory("../plugin");
    ASSERT_TRUE(motion_detection.Init());

    std::cout << "\tRunning the tracker on the video: " << inVideoFile << std::endl;
    MPFVideoJob job("Testing", inVideoFile, start, stop, { }, { });
    std::vector<MPFVideoTrack> found_tracks = motion_detection.GetDetections(job);

    ASSERT_FALSE(found_tracks.empty());
    std::cout << "\tFound " << found_tracks.size() << " tracks" << std::endl;

    for (MPFVideoTrack &track:  found_tracks) {
        ASSERT_TRUE(track.start_frame >= 30); // motion starts on frame 31
    }

    std::cout << "\tClosing down detection." << std::endl;
    ASSERT_TRUE(motion_detection.Close());
}
