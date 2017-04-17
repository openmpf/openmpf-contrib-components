/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2016 The MITRE Corporation. All Rights Reserved.                 *
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
    EXPECT_EQ(dir_input, rundir);

    EXPECT_TRUE(motion_detection->Init());

    MPFComponentType comp_type = motion_detection->GetComponentType();
    EXPECT_EQ(MPF_DETECTION_COMPONENT, comp_type);

    EXPECT_TRUE(motion_detection->Close());
    delete motion_detection;
}

TEST(VideoGeneration, TestOnKnownVideo) {

    std::string current_working_dir = GetCurrentWorkingDirectory();
    std::string test_output_dir = current_working_dir + "/test/test_output/";

    int start = 0;
    int stop = 0;
    int rate = 0;
    float comparison_score_threshold = 0.0;
    string inTrackFile;
    string inVideoFile;
    string outTrackFile;
    string outVideoFile;
    float threshold;

    QHash<QString, QString> parameters;
    QString current_path = QDir::currentPath();
    std::string config_path(current_path.toStdString() + "/config/test_subsense_motion_config.ini");
    std::cout << "config path: " << config_path << std::endl;
    int rc = LoadConfig(config_path, parameters);
    ASSERT_EQ(0, rc);
    std::cout << "Test TestOnKnownVideo: config file loaded" << std::endl;

    cout << "Reading parameters for MOTION_SUBSENSE_DETECTION." << endl;

    start = parameters["SUBSENSE_MOTION_START_FRAME"].toInt();
    stop = parameters["SUBSENSE_MOTION_STOP_FRAME"].toInt();
    rate = parameters["SUBSENSE_MOTION_FRAME_RATE"].toInt();
    inTrackFile = parameters["SUBSENSE_MOTION_KNOWN_TRACKS"].toStdString();
    inVideoFile = parameters["SUBSENSE_MOTION_VIDEO_FILE"].toStdString();
    outTrackFile = parameters["SUBSENSE_MOTION_FOUND_TRACKS"].toStdString();
    outVideoFile = parameters["SUBSENSE_MOTION_VIDEO_OUTPUT_FILE"].toStdString();
    comparison_score_threshold = parameters["SUBSENSE_MOTION_COMPARISON_SCORE_VIDEO"].toFloat();
    // 	Create a person tracker object.

    cout << "\tCreating a SUBSENSE Motion Detection" << endl;
    MotionDetection_Subsense *motion_detection = new MotionDetection_Subsense();
    motion_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(NULL != motion_detection);
    EXPECT_TRUE(motion_detection->Init());

    cout << "Start:\t" << start << endl;
    cout << "Stop:\t" << stop << endl;
    cout << "Rate:\t" << rate << endl;
    cout << "inTrack:\t" << inTrackFile << endl;
    cout << "outTrack:\t" << outTrackFile << endl;
    cout << "inVideo:\t" << inVideoFile << endl;
    cout << "outVideo:\t" << outVideoFile << endl;
    cout << "comparison threshold:\t" << comparison_score_threshold << endl;

    // 	Load the known tracks into memory.
    cout << "\tLoading the known tracks into memory: " << inTrackFile << endl;
    std::vector<MPFVideoTrack> known_tracks;
    EXPECT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    // 	Evaluate the known video file to generate the test tracks.
    cout << "\tRunning the tracker on the video: " << inVideoFile << endl;
    std::vector<MPFVideoTrack> found_tracks;
    MPFVideoJob job("Testing", inVideoFile, start, stop, { }, { });
    EXPECT_FALSE(motion_detection->GetDetections(job, found_tracks));

    EXPECT_FALSE(found_tracks.empty());

    // 	Compare the known and test track output.
    cout << "\tComparing the known and test tracks." << endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    cout << "Tracker comparison score: " << comparison_score << endl;
    EXPECT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    cout << "\tWriting detected video and test tracks to files." << endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    cout << "\tClosing down detection." << endl;
    EXPECT_TRUE(motion_detection->Close());
    delete motion_detection;
}
