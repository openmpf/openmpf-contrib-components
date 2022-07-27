/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2022 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2022 The MITRE Corporation                                       *
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

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>
#include <log4cxx/basicconfigurator.h>

#include "StreamingMotionDetection_Subsense.h"

using namespace MPF;
using namespace COMPONENT;

bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();

constexpr auto plugin_dir = "../plugin";


TEST(StreamingDetectionTest, TestProcessFrame) {

    Properties job_props, media_props;
    MPFStreamingVideoJob job("TestProcessFrame", plugin_dir, job_props, media_props);

    SubsenseStreamingDetection streaming_motion_detection(job);

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
    std::string inVideoFile = "test/test_vids/ff-region-motion-face.avi";
    int init_frame_index = 50;
    int process_frame_index = 80;

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
        streaming_motion_detection.BeginSegment(seg_info);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from BeginSegment: " << e.what();
    }

    // First call to process frame: this will initialize the motion
    // detector and will not produce an activity alert.
    bool activity_alert;
    try {
        activity_alert = streaming_motion_detection.ProcessFrame(frame, 0);
    }
    catch (std::exception &e) {
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
        activity_alert = streaming_motion_detection.ProcessFrame(frame, 1);
    }
    catch (std::exception &e) {
        FAIL() << "Exception thrown from ProcessFrame: " << e.what();
    }

    ASSERT_TRUE(activity_alert);
}


static void runStreamingTest(const MPFStreamingVideoJob &job) {

    int segment_length = 50;
    int num_segments = 2;
    std::string inVideoFile = "test/test_vids/ff-region-motion-face.avi";

    std::cout << "Segment length:\t" << segment_length << std::endl;
    std::cout << "Num segments:\t" << num_segments << std::endl;
    std::cout << "inVideo:\t" << inVideoFile << std::endl;

    SubsenseStreamingDetection streaming_motion_detection(job);

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
    std::vector<MPFVideoTrack> found_tracks;
    int num_activity_alerts = 0;

    for (int seg_index = 0; seg_index < num_segments; seg_index++) {
        int start = frame_index;
        int stop = frame_index + (segment_length-1);
        VideoSegmentInfo seg_info(seg_index, start, stop,
                                  frame.cols, frame.rows);

        try {
            streaming_motion_detection.BeginSegment(seg_info);
        }
        catch (std::exception &e) {
            FAIL() << "Exception thrown from BeginSegment: " << e.what();
        }
        bool first_track_reported = false;

        do {
            if (!frame.empty()) {
                bool activity_alert;
                try {
                    activity_alert = streaming_motion_detection.ProcessFrame(frame, frame_index);
                }
                catch (std::exception &e) {
                    FAIL() << "Exception thrown from ProcessFrame: " << e.what();
                }

                // Check that we only get one activity alert per segment
                if (activity_alert) {
                    ASSERT_FALSE(first_track_reported) << "More than one activity alert received for segment #" << seg_index;
                    first_track_reported = true;
                    num_activity_alerts++;
                }

                segment_frame_count++;
                frame_index++;
            }
        } while (cap.read(frame) && (segment_frame_count < segment_length));

        std::vector<MPFVideoTrack> tracks;
        try {
            tracks = streaming_motion_detection.EndSegment();
        }
        catch (std::exception &e) {
            FAIL() << "Exception thrown from EndSegment: " << e.what();
        }

        // Add the tracks for this segment to the vector of found tracks
        for (auto track : tracks) {
            found_tracks.push_back(std::move(track));
        }

        segment_frame_count = 0;
    }
    cap.release();

    ASSERT_TRUE(num_activity_alerts > 0) << "Expected at least one activity alert";
    ASSERT_FALSE(found_tracks.empty()) << "Expected at least one track";
    std::cout << "Found " << found_tracks.size() << " tracks" << std::endl;

    for (MPFVideoTrack &track:  found_tracks) {
        ASSERT_TRUE(track.start_frame >= 30); // motion starts on frame 31
    }
}


TEST(StreamingDetectionTest, TestMultipleSegments) {
    MPFStreamingVideoJob job("TestMultipleSegments", plugin_dir, {}, {});
    ASSERT_NO_FATAL_FAILURE(runStreamingTest(job));
}


TEST(StreamingDetectionTest, TestMotionTracking) {
    MPFStreamingVideoJob job(
            "TestMotionTracking", plugin_dir,
            { {"USE_MOTION_TRACKING", "TRUE"} }, {});
    ASSERT_NO_FATAL_FAILURE(runStreamingTest(job));
}
