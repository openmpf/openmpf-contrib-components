/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
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
#include "MotionDetectionUtils.h"

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
    std::cout << "Test TestOnKnownVideo: config file loaded" << std::endl;

    cout << "Reading parameters for MOTION_SUBSENSE_DETECTION." << endl;

    int start = parameters["SUBSENSE_MOTION_START_FRAME"].toInt();
    int stop = parameters["SUBSENSE_MOTION_STOP_FRAME"].toInt();
    string inVideoFile = parameters["SUBSENSE_MOTION_VIDEO_FILE"].toStdString();

    cout << "Start:\t" << start << endl;
    cout << "Stop:\t" << stop << endl;
    cout << "inVideo:\t" << inVideoFile << endl;

    cout << "\tCreating a SUBSENSE Motion Detector" << endl;
    MotionDetection_Subsense *motion_detection = new MotionDetection_Subsense();
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


MPFVideoTrack MakeTrack(int num_detections) {
    MPFVideoTrack track(0, num_detections-1, -1);
    for (int i = 0; i < num_detections; ++i) {
        track.frame_locations.insert({i, {i+1, i+1, 5*(i+1), 10*(i+1), -1}});
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

TEST(TestAssignConfidence, NegativeWeight) {

    MPFVideoTrack track = MakeTrack(1);
    float distance_factor = -0.1;
    float size_factor = 0.0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    // One weight factor negative: treat it as zero, so no change in confidence
    ASSERT_NEAR(track.frame_locations.begin()->second.confidence, -1.0, 0.000001);
}

TEST(TestAssignConfidence, BothWeightsOne) {

    MPFVideoTrack track = MakeTrack(5);
    float distance_factor = 1.0;
    float size_factor = 1.0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0.0294118, 0.485294, 1.0, 0.838235, 0.735294}));
}

TEST(TestAssignConfidence, OneDetection) {

    MPFVideoTrack track = MakeTrack(1);
    float distance_factor = 0.5;
    float size_factor = 0.5;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    EXPECT_TRUE(CheckTrack(track, {1.0}));
}

TEST(TestAssignConfidence, TwoDetections) {

    MPFVideoTrack track = MakeTrack(2);
    float distance_factor = 0.25;
    float size_factor = 0.75;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    EXPECT_TRUE(CheckTrack(track,{0.4375, 1.0}));
}

TEST(TestAssignConfidence, NumDetectionEven) {

    MPFVideoTrack track = MakeTrack(8);
    float distance_factor = 0.4;
    float size_factor = 0.6;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0.0147783, 0.269294, 0.553366, 0.866995, 1.0, 0.952381, 0.934319, 0.945813}));
}

TEST(TestAssignConfidence, NumDetectionsOdd) {

    MPFVideoTrack track = MakeTrack(9);
    float distance_factor = 0.4;
    float size_factor = 0.6;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0.0123457, 0.216049, 0.444444, 0.697531, 0.975309, 0.944444, 0.938272, 0.95679, 1.0}));
}

TEST(TestAssignConfidence, DistanceWeightOnlyEvenNumDetections) {

    MPFVideoTrack track = MakeTrack(10);
    float distance_factor = 1.0;
    float size_factor = 0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25, 0}));
}

TEST(TestAssignConfidence, DistanceWeightOnlyOddNumDetections) {

    MPFVideoTrack track = MakeTrack(9);
    float distance_factor = 1.0;
    float size_factor = 0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0}));
}

TEST(TestAssignConfidence, DistanceWeightOneSizeWeightNegative) {

    // This test should give the same results as the one above.
    MPFVideoTrack track = MakeTrack(9);
    float distance_factor = 1.0;
    float size_factor = -0.5;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0}));
}

TEST(TestAssignConfidence, SizeWeightOnly) {

    MPFVideoTrack track = MakeTrack(8);
    float distance_factor = 0;
    float size_factor = 1.0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0.015625, 0.0625, 0.140625, 0.25, 0.390625, 0.5625, 0.765625, 1.0}));
}

TEST(TestAssignConfidence, EqualWeights) {

    MPFVideoTrack track = MakeTrack(7);
    float distance_factor = 0.5;
    float size_factor = 0.5;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_TRUE(CheckTrack(track, {0.0153846, 0.312821, 0.641026, 1.0, 0.88718, 0.805128, 0.753846}));
}

TEST(TestAssignConfidence, EqualBoundingBoxesAtEqualDistance) {

    MPFVideoTrack track = MakeTrack(5);
    track.frame_locations.at(1) = track.frame_locations.at(3);
    float distance_factor = 0.5;
    float size_factor = 0.5;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_NEAR(track.frame_locations.at(1).confidence, track.frame_locations.at(3).confidence, 0.000001);
}

TEST(TestAssignConfidence, EqualBoundingBoxesAtDifferentDistance) {

    MPFVideoTrack track = MakeTrack(7);
    track.frame_locations.at(1) = track.frame_locations.at(4);
    float distance_factor = 0;
    float size_factor = 1.0;
    AssignDetectionConfidence(track, distance_factor, size_factor);
    ASSERT_NEAR(track.frame_locations.at(1).confidence, track.frame_locations.at(4).confidence, 0.000001);
}
