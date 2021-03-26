/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
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

#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <QDir>

#include <gtest/gtest.h>
#include <log4cxx/basicconfigurator.h>

#include <MPFDetectionComponent.h>
#include <Utils.h>
#include <DetectionComparison.h>
#include <ReadDetectionsFromFile.h>
#include <WriteDetectionsToFile.h>
#include <VideoGeneration.h>
#include <ImageGeneration.h>
#include <MPFSimpleConfigLoader.h>

#include "MotionDetection_MOG2.h"

using std::string;
using std::vector;
using std::map;
using std::pair;

using namespace std;
using namespace MPF;
using namespace COMPONENT;



static string GetCurrentWorkingDirectory() {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working dir: " << cwd << std::endl;
        return string(cwd);
    }
    else {
        std::cout << "getcwd() error";
        return "";
    }
}

bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();

TEST(Detection, Init) {

    std::string current_working_dir = GetCurrentWorkingDirectory();
	
    MotionDetection_MOG2 *motion_detection = new MotionDetection_MOG2();
    ASSERT_TRUE(NULL != motion_detection);

    std::string dir_input(current_working_dir + "/../plugin");
    motion_detection->SetRunDirectory(dir_input);
    std::string rundir = motion_detection->GetRunDirectory();
    ASSERT_EQ(dir_input, rundir);

    ASSERT_TRUE(motion_detection->Init());

    MPFComponentType comp_type = motion_detection->GetComponentType();
    ASSERT_EQ(MPF_DETECTION_COMPONENT, comp_type);

    ASSERT_TRUE(motion_detection->Close());
    delete motion_detection;
}

TEST(VideoGeneration, TestOnKnownVideo) {

    std::string current_working_dir = GetCurrentWorkingDirectory();

    QHash<QString, QString> parameters;
    QString current_path = QDir::currentPath();
    std::string config_path(current_path.toStdString() + "/config/test_mog_motion_config.ini");
    std::cout << "config path: " << config_path << std::endl;
    int rc = LoadConfig(config_path, parameters);
    ASSERT_EQ(0, rc);
    std::cout << "Test TestOnKnownVideo: config file loaded" << std::endl;

    cout << "Reading parameters for MOTION_MOG_DETECTION." << endl;

    int start = parameters["MOG_MOTION_START_FRAME"].toInt();
    int stop = parameters["MOG_MOTION_STOP_FRAME"].toInt();
    string inVideoFile = parameters["MOG_MOTION_VIDEO_FILE"].toStdString();

    cout << "Start:\t" << start << endl;
    cout << "Stop:\t" << stop << endl;
    cout << "inVideo:\t" << inVideoFile << endl;

    cout << "\tCreating a MOG Motion Detector" << endl;
    MotionDetection_MOG2 *motion_detection = new MotionDetection_MOG2();
    motion_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(NULL != motion_detection);
    ASSERT_TRUE(motion_detection->Init());

    cout << "\tRunning the tracker on the video: " << inVideoFile << endl;
    MPFVideoJob job("Testing", inVideoFile, start, stop, { }, { });
    std::vector<MPFVideoTrack> found_tracks = motion_detection->GetDetections(job);

    ASSERT_FALSE(found_tracks.empty());
    cout << "\tFound " << found_tracks.size() << " tracks" << endl;

    for (MPFVideoTrack &track:  found_tracks) {
        ASSERT_TRUE(track.start_frame >= 30); // motion starts on frame 31
    }

    cout << "\tClosing down detection." << endl;
    ASSERT_TRUE(motion_detection->Close());
    delete motion_detection;
}
