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
#include "AssignDetectionConfidence.h"

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


MPFVideoTrack MakeTrack(int num_detections) {
    MPFVideoTrack track(0, num_detections-1, -1);
    for (int i = 0; i < num_detections; ++i) {
        track.frame_locations.insert({i, {i+1, i+1, i*5, i*10, -1}});
    }
    return track;
}

bool CheckTrack(MPFVideoTrack &track, vector<float> expected) {
    int index = 0;
    for (auto &entry : track.frame_locations) {
        if (abs(entry.second.confidence - expected[index]) > 0.00001) return false;
        index++;
    }
    return true;
}


TEST(TestAssignConfidence, BothWeightsZero) {

    MPFVideoTrack track = MakeTrack(1);
    float distance_factor = 0.0;
    float size_factor = 0.0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    // Both weight factors equal to zero must leave confidence unchanged.
    ASSERT_NEAR(track.frame_locations.begin()->second.confidence, -1.0, 0.000001);
}

TEST(TestAssignConfidence, BothWeightsOne) {

    MPFVideoTrack track = MakeTrack(5);
    float distance_factor = 1.0;
    float size_factor = 1.0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.45, 1.0, 0.85, 0.8}));
}

TEST(TestAssignConfidence, OneDetection) {

    MPFVideoTrack track;
    float distance_factor = 0.5;
    float size_factor = 0.5;
    MPFImageLocation loc(10, 10, 100, 100, -1);
    track.frame_locations.insert({1, loc});
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_NEAR(track.frame_locations.begin()->second.confidence, 1.0, 0.000001);
}

TEST(TestAssignConfidence, TwoDetections) {

    MPFVideoTrack track;
    float distance_factor = 0.25;
    float size_factor = 0.75;
    MPFImageLocation loc1(10, 10, 100, 100, -1);
    track.frame_locations.insert({1, loc1});
    MPFImageLocation loc2(15, 15, 150, 100, -1);
    track.frame_locations.insert({2, loc2});
    AssignDetectionConfidence(track, distance_factor, size_factor);
    auto start = track.frame_locations.begin();
    EXPECT_NEAR(start->second.confidence, 0.75, 0.000001);
    EXPECT_NEAR((start++)->second.confidence, 0.75, 0.000001);
}

TEST(TestAssignConfidence, NumDetectionEven) {

    MPFVideoTrack track = MakeTrack(8);
    float distance_factor = 0.4;
    float size_factor = 0.6;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.24263, 0.526077, 0.85034, 0.993197, 0.954648, 0.956916, 1.0}));
}

TEST(TestAssignConfidence, NumDetectionsOdd) {

    MPFVideoTrack track = MakeTrack(9);
    float distance_factor = 0.4;
    float size_factor = 0.6;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.182292, 0.395833, 0.640625, 0.916667, 0.890625, 0.895833, 0.932292, 1.0}));
}

TEST(TestAssignConfidence, DistanceWeightOnlyEvenNumDetections) {

    MPFVideoTrack track = MakeTrack(10);
    float distance_factor = 1.0;
    float size_factor = 0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.333333, 0.666667, 1.0, 1.0, 0.666667, 0.333333, 0}));
}

TEST(TestAssignConfidence, DistanceWeightOnlyOddNumDetections) {

    MPFVideoTrack track = MakeTrack(9);
    float distance_factor = 1.0;
    float size_factor = 0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0}));
}
TEST(TestAssignConfidence, SizeWeightOnly) {

    MPFVideoTrack track = MakeTrack(8);
    float distance_factor = 0;
    float size_factor = 1.0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.0204082, 0.0816327, 0.183673, 0.326531, 0.510204, 0.734694, 1.0}));
}
TEST(TestAssignConfidence, EqualWeights) {

    MPFVideoTrack track = MakeTrack(10);
    float distance_factor = 0.5;
    float size_factor = 0.5;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0, 0}));
}

