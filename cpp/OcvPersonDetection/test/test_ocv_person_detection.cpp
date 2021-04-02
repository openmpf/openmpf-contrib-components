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
 * person detection component software.                                       *
 *                                                                            *
 * The MPF person detection component software is free software: you can      *
 * redistribute it and/or modify it under the terms of the GNU General        *
 * Public License as published by the Free Software Foundation, either        *
 * version 3 of the License, or (at your option) any later version.           *
 *                                                                            *
 * The MPF person detection component software is distributed in the hope     *
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the         *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.   *
 * See the (GNU General Public License for more details.                      *
 *                                                                            *
 * You should have received a copy of the GNU General Public License          *
 * along with the MPF person detection component software. If not, see        *
 * <http://www.gnu.org/licenses/>.                                            *
 ******************************************************************************/

#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <log4cxx/basicconfigurator.h>

#include <QDir>

#include <Utils.h>
#include <DetectionComparison.h>
#include <ReadDetectionsFromFile.h>
#include <WriteDetectionsToFile.h>
#include <VideoGeneration.h>
#include <ImageGeneration.h>
#include <MPFSimpleConfigLoader.h>

#include "PersonDetection.h"

using std::string;
using std::vector;
using std::map;
using std::pair;

using namespace MPF;
using namespace COMPONENT;


// global variable to hold the file name parameters
QHash<QString, QString> parameters;
bool parameters_loaded = false;

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

    string current_working_dir = GetCurrentWorkingDirectory();

    PersonDetection *person_detection = new PersonDetection();
    ASSERT_TRUE(NULL != person_detection);

    string dir_input(current_working_dir + "/../plugin");
    person_detection->SetRunDirectory(dir_input);
    string rundir = person_detection->GetRunDirectory();
    EXPECT_EQ(dir_input, rundir);

    ASSERT_TRUE(person_detection->Init());

    MPFComponentType comp_type = person_detection->GetComponentType();
    EXPECT_EQ(MPF_DETECTION_COMPONENT, comp_type);

    EXPECT_TRUE(person_detection->Close());
    delete person_detection;
}

TEST(VideoGeneration, TestOnKnownVideo) {

    string current_working_dir = GetCurrentWorkingDirectory();
    string test_output_dir = current_working_dir + "/test/test_output/";

    int start = 0;
    int stop = 0;
    int rate = 0;
    float comparison_score_threshold = 0.0;
    string inTrackFile;
    string inVideoFile;
    string outTrackFile;
    string outVideoFile;
    float threshold;

    if (!parameters_loaded) {
      QString current_path = QDir::currentPath();
      string config_path(current_path.toStdString() + "/config/test_ocv_person_config.ini");
      std::cout << "config path: " << config_path << std::endl;
      int rc = LoadConfig(config_path, parameters);
      ASSERT_EQ(0, rc);
      std::cout << "Test TestOnKnownVideo: config file loaded" << std::endl;
      parameters_loaded = true;
    }

    std::cout << "Reading parameters for video test." << std::endl;

    start = parameters["OCV_PERSON_START_FRAME"].toInt();
    stop = parameters["OCV_PERSON_STOP_FRAME"].toInt();
    rate = parameters["OCV_PERSON_FRAME_RATE"].toInt();
    inTrackFile = parameters["OCV_PERSON_KNOWN_TRACKS"].toStdString();
    inVideoFile = parameters["OCV_PERSON_VIDEO_FILE"].toStdString();
    outTrackFile = parameters["OCV_PERSON_FOUND_TRACKS"].toStdString();
    outVideoFile = parameters["OCV_PERSON_VIDEO_OUTPUT_FILE"].toStdString();
    comparison_score_threshold = parameters["OCV_PERSON_COMPARISON_SCORE_VIDEO"].toFloat();

    // 	Create an OCV person detection object.
    std::cout << "\tCreating OCV Person Detection" << std::endl;
    PersonDetection *person_detection = new PersonDetection();
    ASSERT_TRUE(NULL != person_detection);
    person_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(person_detection->Init());


    std::cout << "Start:\t" << start << std::endl;
    std::cout << "Stop:\t" << stop << std::endl;
    std::cout << "Rate:\t" << rate << std::endl;
    std::cout << "inTrack:\t" << inTrackFile << std::endl;
    std::cout << "outTrack:\t" << outTrackFile << std::endl;
    std::cout << "inVideo:\t" << inVideoFile << std::endl;
    std::cout << "outVideo:\t" << outVideoFile << std::endl;
    std::cout << "comparison threshold:\t" << comparison_score_threshold << std::endl;

    // 	Load the known tracks into memory.
    std::cout << "\tLoading the known tracks into memory: " << inTrackFile << std::endl;
    vector<MPFVideoTrack> known_tracks;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    // 	Evaluate the known video file to generate the test tracks.
    std::cout << "\tRunning the tracker on the video: " << inVideoFile << std::endl;
    MPFVideoJob job("Testing", inVideoFile, start, stop, { }, { });
    vector<MPFVideoTrack> found_tracks = person_detection->GetDetections(job);
    EXPECT_FALSE(found_tracks.empty());

    // 	Compare the known and test track output.
    std::cout << "\tComparing the known and test tracks." << std::endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    std::cout << "Tracker comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    std::cout << "\tWriting detected video and test tracks to files." << std::endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    // don't forget
    std::cout << "\tClosing down detection." << std::endl;
    EXPECT_TRUE(person_detection->Close());
    delete person_detection;
}

TEST(ImageGeneration, TestOnKnownImage) {

    string current_working_dir = GetCurrentWorkingDirectory();
    string test_output_dir = current_working_dir + "/test/test_output/";

    string known_image_file;
    string known_detections_file;
    string output_image_file;
    string output_detections_file;
    float comparison_score_threshold = 0.0;

    if (!parameters_loaded) {
      QString current_path = QDir::currentPath();
      string config_path(current_path.toStdString() + "/config/test_ocv_person_config.ini");
      std::cout << "config path: " << config_path << std::endl;
      int rc = LoadConfig(config_path, parameters);
      ASSERT_EQ(0, rc);
      std::cout << "Test TestOnKnownImage: config file loaded" << std::endl;
      parameters_loaded = true;
    }

    std::cout << "Reading parameters for image test." << std::endl;

    known_image_file = parameters["OCV_PERSON_IMAGE_FILE"].toStdString();
    known_detections_file = parameters["OCV_PERSON_KNOWN_DETECTIONS"].toStdString();
    output_image_file = parameters["OCV_PERSON_IMAGE_OUTPUT_FILE"].toStdString();
    output_detections_file = parameters["OCV_PERSON_FOUND_DETECTIONS"].toStdString();

    comparison_score_threshold = parameters["OCV_PERSON_COMPARISON_SCORE_IMAGE"].toFloat();
    // 	Create an OCV person detection object.
    PersonDetection *person_detection = new PersonDetection();
    ASSERT_TRUE(NULL != person_detection);

    person_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(person_detection->Init());

    std::cout << "Input Known Detections:\t" << known_detections_file << std::endl;
    std::cout << "Output Found Detections:\t" << output_detections_file << std::endl;
    std::cout << "Input Image:\t" << known_image_file << std::endl;
    std::cout << "Output Image:\t" << output_image_file << std::endl;
    std::cout << "comparison threshold:\t" << comparison_score_threshold << std::endl;

    // 	Load the known detections into memory.
    vector<MPFImageLocation> known_detections;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadImageLocations(known_detections_file, known_detections));

    MPFImageJob job("Testing", known_image_file, { }, { });
    vector<MPFImageLocation> found_detections = person_detection->GetDetections(job);
    EXPECT_FALSE(found_detections.empty());

    float comparison_score = DetectionComparison::CompareDetectionOutput(found_detections, known_detections);

    std::cout << "Detection comparison score: " << comparison_score << std::endl;

    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(known_image_file,
                                                found_detections,
                                                test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                              found_detections);

    EXPECT_TRUE(person_detection->Close());
    delete person_detection;
}
