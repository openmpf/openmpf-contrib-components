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
 * Copyright 2019 The MITRE Corporation                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <QDir>
#include "gtest/gtest.h"

#include <opencv2/opencv.hpp>

#include <Utils.h>

#include <DetectionComparison.h>
#include <ReadDetectionsFromFile.h>
#include <WriteDetectionsToFile.h>
#include <VideoGeneration.h>
#include <ImageGeneration.h>
#include "MPFSimpleConfigLoader.h"

#include "StreamingMotionDetection_Subsense.h"

using std::string;
using std::vector;
using std::map;
using std::pair;

using namespace MPF;
using namespace COMPONENT;

class StreamingDetectionTest : public ::testing::Test {
  protected:
    StreamingDetectionTest() = default;

    static std::shared_ptr<QHash<QString, QString> > parameters_;
    static string plugin_dir_;
    static string test_output_dir_;

    static void SetUpTestCase() {

        string base_path(QDir::currentPath().toStdString());

        plugin_dir_ = base_path + "/../plugin";
        std::cout << "plugin_dir: " << plugin_dir_ << std::endl;

        // Define the test output directory
        test_output_dir_ = base_path + "/test/test_output/";
        std::cout << "test_output_dir: " << test_output_dir_ << std::endl;

        // Read the parameters file
        string config_path(base_path + "/config/test_subsense_motion_config.ini");
        std::cout << "config path: " << config_path << std::endl;

        parameters_.reset(new QHash<QString, QString>);

        int rc = LoadConfig(config_path, *parameters_);
        ASSERT_EQ(0, rc);
        std::cout << "StreamingDetectionTest::SetupTestCase config file loaded" << std::endl;

    }

    static void TearDownTestCase() {
    }
};

std::shared_ptr<QHash<QString, QString> > StreamingDetectionTest::parameters_ = NULL;
string StreamingDetectionTest::plugin_dir_ = "";
string StreamingDetectionTest::test_output_dir_ = "";


TEST_F(StreamingDetectionTest, TestConstructor) {

    Properties job_props, media_props;

    MPFStreamingVideoJob job("TestConstructor", plugin_dir_, job_props, media_props);
    SubsenseStreamingDetection *streaming_motion_detection;
    try {
        streaming_motion_detection = new SubsenseStreamingDetection(job);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from constructor: " << e.what();
    }
    
    ASSERT_TRUE(NULL != streaming_motion_detection);

    string detection_type = streaming_motion_detection->GetDetectionType();
    EXPECT_EQ("MOTION", detection_type);

    delete streaming_motion_detection;
}

TEST_F(StreamingDetectionTest, TestBeginSegment) {

    Properties job_props, media_props;
    MPFStreamingVideoJob job("TestBeginSegment", plugin_dir_, job_props, media_props);

    // 	Create a Subsense streaming motion detection object.
    SubsenseStreamingDetection *streaming_motion_detection;
    try {
        streaming_motion_detection = new SubsenseStreamingDetection(job);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from constructor: " << e.what();
    }

    ASSERT_TRUE(NULL != streaming_motion_detection);
    VideoSegmentInfo seg_info(0, 0, 100, 500, 500);

    try {
        streaming_motion_detection->BeginSegment(seg_info);
    }
    catch (std::exception &e) {
        delete streaming_motion_detection;
        FAIL() << "Exception thrown from BeginSegment: " << e.what();
    }

    delete streaming_motion_detection;
}

TEST_F(StreamingDetectionTest, TestProcessFrame) {

    Properties job_props, media_props;
    MPFStreamingVideoJob job("TestProcessFrame", plugin_dir_, job_props, media_props);

    SubsenseStreamingDetection *streaming_motion_detection;
    try {
        streaming_motion_detection = new SubsenseStreamingDetection(job);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from constructor: " << e.what();
    }

    ASSERT_TRUE(NULL != streaming_motion_detection);

    // The motion detector requires at least two frames to detect
    // motion. The first frame will be used to initialize the
    // motion detector, and the next frame will be compared with it to
    // detect motion. So this test needs to read two frames and call
    // ProcessFrame twice.

    // In order to ensure that motion is detected when it should, we
    // exaggerate the motion by running the detection on a frame that
    // is some number of frames after the initialization frame. This
    // works with the specified input test video, in which the motion
    // is continuous after an initial period of time.
    string inVideoFile = parameters_->value("SUBSENSE_STREAMING_MOTION_VIDEO_FILE").toStdString();
    int init_frame_index = parameters_->value("SUBSENSE_STREAMING_MOTION_INIT_FRAME").toInt();
    int process_frame_index = parameters_->value("SUBSENSE_STREAMING_MOTION_PROCESS_FRAME").toInt();

    cv::VideoCapture cap(inVideoFile);
    ASSERT_TRUE(cap.isOpened());

    // read the initialization frame
    cv::Mat frame;
    cap.set(cv::CAP_PROP_POS_FRAMES, init_frame_index);

    ASSERT_TRUE(cap.read(frame));

    ASSERT_FALSE(frame.empty());

    int segment_num = 0;
    int start_frame = init_frame_index;
    int end_frame = process_frame_index;
    VideoSegmentInfo seg_info(segment_num, start_frame, end_frame,
                              frame.cols, frame.rows);

    try {
        streaming_motion_detection->BeginSegment(seg_info);
    }
    catch (std::exception &e) {
        delete streaming_motion_detection;
        FAIL() << "Exception thrown from BeginSegment: " << e.what();
    }

    // First call to process frame: this will initialize the motion
    // detector and will not produce an activity alert.
    bool activity_alert = false;
    try {
        activity_alert = streaming_motion_detection->ProcessFrame(frame, 0);
    }
    catch (std::exception &e) {
        delete streaming_motion_detection;
        FAIL() << "Exception thrown from ProcessFrame: " << e.what();
    }

    ASSERT_FALSE(activity_alert);

    // Next call to process frame: read ahead a number of frames to
    // exaggerate the motion, to ensure that we get an activity
    // alert.
    cap.set(cv::CAP_PROP_POS_FRAMES, process_frame_index);
    ASSERT_TRUE(cap.read(frame));

    ASSERT_FALSE(frame.empty());

    try {
        activity_alert = streaming_motion_detection->ProcessFrame(frame, 1);
    }
    catch (std::exception &e) {
        delete streaming_motion_detection;
        FAIL() << "Exception thrown from ProcessFrame: " << e.what();
    }

    ASSERT_TRUE(activity_alert);

    delete streaming_motion_detection;
}


TEST_F(StreamingDetectionTest, TestEndSegment) {

    int start = 0;
    int stop = 0;
    float threshold = 0.0;
    string inTrackFile;
    string inVideoFile;
    string outTrackFile;
    string outVideoFile;

    start = parameters_->value("SUBSENSE_STREAMING_MOTION_START_FRAME").toInt();
    stop = parameters_->value("SUBSENSE_STREAMING_MOTION_STOP_FRAME").toInt();
    inTrackFile = parameters_->value("SUBSENSE_STREAMING_MOTION_KNOWN_TRACKS").toStdString();
    inVideoFile = parameters_->value("SUBSENSE_STREAMING_MOTION_VIDEO_FILE").toStdString();
    outTrackFile = parameters_->value("SUBSENSE_STREAMING_TEST_ENDSEGMENT_FOUND_TRACKS").toStdString();
    outVideoFile = parameters_->value("SUBSENSE_STREAMING_TEST_ENDSEGMENT_VIDEO_OUTPUT_FILE").toStdString();
    threshold = parameters_->value("SUBSENSE_STREAMING_MOTION_COMPARISON_SCORE_VIDEO").toFloat();

    std::cout << "Start:\t" << start << std::endl;
    std::cout << "Stop:\t" << stop << std::endl;
    std::cout << "inTrack:\t" << inTrackFile << std::endl;
    std::cout << "outTrack:\t" << outTrackFile << std::endl;
    std::cout << "inVideo:\t" << inVideoFile << std::endl;
    std::cout << "outVideo:\t" << outVideoFile << std::endl;
    std::cout << "comparison threshold:\t" << threshold << std::endl;

    Properties job_props, media_props;
    MPFStreamingVideoJob job("TestEndSegment", plugin_dir_, job_props, media_props);

    SubsenseStreamingDetection *streaming_motion_detection;
    try {
        streaming_motion_detection = new SubsenseStreamingDetection(job);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from constructor: " << e.what();
    }

    ASSERT_TRUE(NULL != streaming_motion_detection);

    // 	Load the known tracks into memory.
    std::cout << "\tLoading the known tracks into memory: " << inTrackFile << std::endl;
    vector<MPFVideoTrack> known_tracks;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    cv::VideoCapture cap(inVideoFile);
    ASSERT_TRUE(cap.isOpened());
    cap.set(cv::CAP_PROP_POS_FRAMES, start);

    int frame_index = start;
    cv::Mat frame;
    ASSERT_TRUE(cap.read(frame));

    ASSERT_FALSE(frame.empty());

    int segment_num = 0;
    VideoSegmentInfo seg_info(segment_num, start, stop, frame.cols, frame.rows);

    try {
        streaming_motion_detection->BeginSegment(seg_info);
    }
    catch (std::exception &e) {
        delete streaming_motion_detection;
        FAIL() << "Exception thrown from BeginSegment: " << e.what();
    }

    bool first_track_reported = false;
    while (frame_index < stop) {
        bool activity_alert;
        EXPECT_TRUE(cap.read(frame));
        if(frame.empty()) {
            continue;
        }
        else { 

            try {
                activity_alert = streaming_motion_detection->ProcessFrame(frame, frame_index);
            }
            catch (std::exception &e) {
                delete streaming_motion_detection;
                FAIL() << "Exception thrown from ProcessFrame: " << e.what();
            }
            // Check that if the first track has been reported, then
            // we no longer get activity alerts
            if (activity_alert) {
                ASSERT_FALSE(first_track_reported) << "More than one activity alert received";
                first_track_reported = true;
            }
            frame_index++;
        }
    }

    cap.release();
    vector<MPFVideoTrack> found_tracks;
    try {
        found_tracks = streaming_motion_detection->EndSegment();
    }
    catch (std::exception &e) {
        delete streaming_motion_detection;
        FAIL() << "Exception thrown from EndSegment: " << e.what();
    }

    // 	Compare the known and test track output.
    std::cout << "\tComparing the known and test tracks." << std::endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    std::cout << "Tracker comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > threshold);

    // create output video to view performance
    std::cout << "\tWriting detected video and test tracks to files." << std::endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir_ + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir_ + "/" + outTrackFile), found_tracks);

    delete streaming_motion_detection;
}



TEST_F(StreamingDetectionTest, TestMultipleSegments) {

    int segment_length = 0;
    int num_segments = 0;
    float threshold = 0.0;
    string inTrackFile;
    string inVideoFile;
    string outTrackFile;
    string outVideoFile;

    segment_length = parameters_->value("SUBSENSE_STREAMING_MOTION_SEGMENT_LENGTH").toInt();
    num_segments = parameters_->value("SUBSENSE_STREAMING_MOTION_NUM_SEGMENTS").toInt();
    inTrackFile = parameters_->value("SUBSENSE_STREAMING_MOTION_KNOWN_TRACKS").toStdString();
    inVideoFile = parameters_->value("SUBSENSE_STREAMING_MOTION_VIDEO_FILE").toStdString();
    outTrackFile = parameters_->value("SUBSENSE_STREAMING_TEST_MULTISEGMENT_FOUND_TRACKS").toStdString();
    outVideoFile = parameters_->value("SUBSENSE_STREAMING_TEST_MULTISEGMENT_VIDEO_OUTPUT_FILE").toStdString();
    threshold = parameters_->value("SUBSENSE_STREAMING_MOTION_COMPARISON_SCORE_VIDEO").toFloat();

    std::cout << "Segment length:\t" << segment_length << std::endl;
    std::cout << "Num segments:\t" << num_segments << std::endl;
    std::cout << "inTrack:\t" << inTrackFile << std::endl;
    std::cout << "outTrack:\t" << outTrackFile << std::endl;
    std::cout << "inVideo:\t" << inVideoFile << std::endl;
    std::cout << "outVideo:\t" << outVideoFile << std::endl;
    std::cout << "comparison threshold:\t" << threshold << std::endl;

    Properties job_props, media_props;
    MPFStreamingVideoJob job("TestMultipleSegments", plugin_dir_, job_props, media_props);

    SubsenseStreamingDetection *streaming_motion_detection;
    try {
        streaming_motion_detection = new SubsenseStreamingDetection(job);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from constructor: " << e.what();
    }

    ASSERT_TRUE(NULL != streaming_motion_detection);

    // 	Load the known tracks into memory.
    std::cout << "\tLoading the known tracks into memory: " << inTrackFile << std::endl;
    vector<MPFVideoTrack> known_tracks;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    cv::VideoCapture cap(inVideoFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    // get first frame
    while (cap.read(frame)) {
        if (!frame.empty()) {
            break;
        }
    }
    int segment_frame_count = 0;
    int frame_index = 0;
    vector<MPFVideoTrack> found_tracks;

    for (int seg_index = 0; seg_index < num_segments; seg_index++) {
        int start = frame_index;
        int stop = frame_index + (segment_length-1);
        VideoSegmentInfo seg_info(seg_index, start, stop,
                                  frame.cols, frame.rows);

        try {
            streaming_motion_detection->BeginSegment(seg_info);
        }
        catch (std::exception &e) {
            delete streaming_motion_detection;
            FAIL() << "Exception thrown from BeginSegment: " << e.what();
        }
        bool first_track_reported = false;

        do {
            if (!frame.empty()) {
                bool activity_alert = false;
                try {
                    activity_alert = streaming_motion_detection->ProcessFrame(frame, frame_index);
                }
                catch (std::exception &e) {
                    delete streaming_motion_detection;
                    FAIL() << "Exception thrown from ProcessFrame: " << e.what();
                }

                // Check that if the first track has been reported, then
                // we no longer get activity alerts
                if (activity_alert) {
                    ASSERT_FALSE(first_track_reported) << "More than one activity alert received for segment #" << seg_index;
                    first_track_reported = true;
                }
                segment_frame_count++;
                frame_index++;
            }
        } while (cap.read(frame) && (segment_frame_count < segment_length));

        vector<MPFVideoTrack> tracks;
        try {
            tracks = streaming_motion_detection->EndSegment();
        }
        catch (std::exception &e) {
            delete streaming_motion_detection;
            FAIL() << "Exception thrown from EndSegment: " << e.what();
        }

        // Add the tracks for this segment to the vector of found tracks
        for (auto track : tracks) {
            found_tracks.push_back(std::move(track));
        }

        segment_frame_count = 0;
    }
    cap.release();

    // 	Compare the known and test track output.
    std::cout << "\tComparing the known and test tracks." << std::endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    std::cout << "Tracker comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > threshold);

    // create output video to view performance
    std::cout << "\tWriting detected video and test tracks to files." << std::endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir_ + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir_ + "/" + outTrackFile), found_tracks);

    delete streaming_motion_detection;
}



TEST_F(StreamingDetectionTest, TestMotionTracking) {

    int segment_length = 0;
    int num_segments = 0;
    float threshold = 0.0;
    string inTrackFile;
    string inVideoFile;
    string outTrackFile;
    string outVideoFile;

    segment_length = parameters_->value("SUBSENSE_STREAMING_MOTION_SEGMENT_LENGTH").toInt();
    num_segments = parameters_->value("SUBSENSE_STREAMING_MOTION_NUM_SEGMENTS").toInt();
    inTrackFile = parameters_->value("SUBSENSE_STREAMING_TEST_MOTIONTRACKING_KNOWN_TRACKS").toStdString();
    inVideoFile = parameters_->value("SUBSENSE_STREAMING_MOTION_VIDEO_FILE").toStdString();
    outTrackFile = parameters_->value("SUBSENSE_STREAMING_TEST_MOTIONTRACKING_FOUND_TRACKS").toStdString();
    outVideoFile = parameters_->value("SUBSENSE_STREAMING_TEST_MOTIONTRACKING_VIDEO_OUTPUT_FILE").toStdString();
    threshold = parameters_->value("SUBSENSE_STREAMING_MOTION_COMPARISON_SCORE_VIDEO").toFloat();

    std::cout << "Segment length:\t" << segment_length << std::endl;
    std::cout << "Num segments:\t" << num_segments << std::endl;
    std::cout << "inTrack:\t" << inTrackFile << std::endl;
    std::cout << "outTrack:\t" << outTrackFile << std::endl;
    std::cout << "inVideo:\t" << inVideoFile << std::endl;
    std::cout << "outVideo:\t" << outVideoFile << std::endl;
    std::cout << "comparison threshold:\t" << threshold << std::endl;

    Properties job_props, media_props;

    // Turn on motion tracking in the job properties
    job_props["USE_MOTION_TRACKING"] = "1";
    MPFStreamingVideoJob job("TestMotionTracking", plugin_dir_, job_props, media_props);

    SubsenseStreamingDetection *streaming_motion_detection;
    try {
        streaming_motion_detection = new SubsenseStreamingDetection(job);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from constructor: " << e.what();
    }

    ASSERT_TRUE(NULL != streaming_motion_detection);

    // 	Load the known tracks into memory.
    std::cout << "\tLoading the known tracks into memory: " << inTrackFile << std::endl;
    vector<MPFVideoTrack> known_tracks;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    cv::VideoCapture cap(inVideoFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    // get first frame
    while (cap.read(frame)) {
        if (!frame.empty()) {
            break;
        }
    }
    int segment_frame_count = 0;
    int frame_index = 0;
    vector<MPFVideoTrack> found_tracks;

    for (int seg_index = 0; seg_index < num_segments; seg_index++) {
        int start = frame_index;
        int stop = frame_index + (segment_length-1);
        VideoSegmentInfo seg_info(seg_index, start, stop,
                                  frame.cols, frame.rows);

        try {
            streaming_motion_detection->BeginSegment(seg_info);
        }
        catch (std::exception &e) {
            delete streaming_motion_detection;
            FAIL() << "Exception thrown from BeginSegment: " << e.what();
        }
        bool first_track_reported = false;

        do {
            if (!frame.empty()) {
                bool activity_alert = false;
                try {
                    activity_alert = streaming_motion_detection->ProcessFrame(frame, frame_index);
                }
                catch (std::exception &e) {
                    delete streaming_motion_detection;
                    FAIL() << "Exception thrown from ProcessFrame: " << e.what();
                }

                // Check that we only get one activity alert per segment
                if (activity_alert) {
                    ASSERT_FALSE(first_track_reported) << "More than one activity alert received for segment #" << seg_index;
                    first_track_reported = true;
                }

                segment_frame_count++;
                frame_index++;
            }
        } while (cap.read(frame) && (segment_frame_count < segment_length));

        vector<MPFVideoTrack> tracks;
        try {
            tracks = streaming_motion_detection->EndSegment();
        }
        catch (std::exception &e) {
            delete streaming_motion_detection;
            FAIL() << "Exception thrown from EndSegment: " << e.what();
        }

        // Add the tracks for this segment to the vector of found tracks
        for (auto track : tracks) {
            found_tracks.push_back(std::move(track));
        }

        segment_frame_count = 0;
    }
    cap.release();

    // 	Compare the known and test track output.
    std::cout << "\tComparing the known and test tracks." << std::endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    std::cout << "Tracker comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > threshold);

    // create output video to view performance
    std::cout << "\tWriting detected video and test tracks to files." << std::endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir_ + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir_ + "/" + outTrackFile), found_tracks);

    delete streaming_motion_detection;
}
