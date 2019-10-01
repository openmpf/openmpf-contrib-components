/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
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

#include "gtest/gtest.h"

#include <MPFDetectionComponent.h>
#include <Utils.h>
#include <DetectionComparison.h>
#include <ReadDetectionsFromFile.h>
#include <WriteDetectionsToFile.h>
#include <VideoGeneration.h>
#include <ImageGeneration.h>
#include <MPFSimpleConfigLoader.h>

#include "MotionDetection_Subsense.h"

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

TEST(Detection, Init) {

    std::string current_working_dir = GetCurrentWorkingDirectory();
	
    MotionDetection_Subsense *motion_detection = new MotionDetection_Subsense();
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
    std::string config_path(current_path.toStdString() + "/config/test_subsense_motion_config.ini");
    std::cout << "config path: " << config_path << std::endl;
    int rc = LoadConfig(config_path, parameters);
    ASSERT_EQ(0, rc);

    cout << "Reading parameters for MOTION_SUBSENSE_DETECTION." << endl;

    int start = parameters["SUBSENSE_MOTION_START_FRAME"].toInt();
    int stop = parameters["SUBSENSE_MOTION_STOP_FRAME"].toInt();
    string inVideoFile = parameters["SUBSENSE_MOTION_VIDEO_FILE"].toStdString();

    cout << "Start:\t" << start << endl;
    cout << "Stop:\t" << stop << endl;
    cout << "inVideo:\t" << inVideoFile << endl;

    cout << "\tCreating a SUBSENSE Motion Detection" << endl;
    MotionDetection_Subsense *motion_detection = new MotionDetection_Subsense();
    motion_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(NULL != motion_detection);
    ASSERT_TRUE(motion_detection->Init());

    cout << "\tRunning the tracker on the video: " << inVideoFile << endl;
    std::vector<MPFVideoTrack> found_tracks;
    MPFVideoJob job("Testing", inVideoFile, start, stop, { }, { });
    ASSERT_EQ(MPFDetectionError::MPF_DETECTION_SUCCESS, motion_detection->GetDetections(job, found_tracks));

    ASSERT_FALSE(found_tracks.empty());

    cout << "\tClosing down detection." << endl;
    ASSERT_TRUE(motion_detection->Close());
    delete motion_detection;
}
