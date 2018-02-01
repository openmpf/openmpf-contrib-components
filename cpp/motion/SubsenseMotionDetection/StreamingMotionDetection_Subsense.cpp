/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

#include <stdexcept>

#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>

#include <MPFSimpleConfigLoader.h>
#include "StreamingMotionDetection_Subsense.h"
#include "MotionDetectionUtils.h"

using namespace MPF;
using namespace COMPONENT;

using std::string;
using std::map;
using std::pair;
using std::vector;

SubsenseStreamingDetection::SubsenseStreamingDetection(const MPFStreamingVideoJob &job)
        : MPFStreamingDetectionComponent(job) {

    job_name_ = job.job_name;
    msg_prefix_ =  "[" + job_name_ + "] ";

    string plugin_path = job.run_directory + "/SubsenseMotionDetection";
    string logger_file = plugin_path + "/config/Log4cxxConfig.xml";
    log4cxx::xml::DOMConfigurator::configure(logger_file);
    motion_logger_ = log4cxx::Logger::getLogger("SubsenseStreamingMotionDetection");

    string config_file = plugin_path + "/config/mpfSubsenseMotionDetection.ini";

    int rc = LoadConfig(config_file, parameters_);
    if (rc) {
        string tmp = "failed to load config file: " + config_file;
        LOG4CXX_ERROR(motion_logger_, msg_prefix_ << tmp);
        throw std::runtime_error(tmp);
    }

    verbose_ = parameters_["VERBOSE"].toUInt();

    if (verbose_ > 0)
        motion_logger_->setLevel(log4cxx::Level::getDebug());

    GetPropertySettings(job.job_properties, parameters_);

    activity_detected_ = false;

    // Create background subtractor
    LOG4CXX_TRACE(motion_logger_, msg_prefix_ << "Creating background subtractor");

    BackgroundSubtractorSuBSENSE bg_(parameters_["F_REL_LBSP_THRESHOLD"].toFloat(),
                                     static_cast<size_t>(parameters_["N_MIN_DESC_DIST_THRESHOLD"].toInt()),
                                     static_cast<size_t>(parameters_["N_MIN_COLOR_DIST_THRESHOLD"].toInt()),
                                     static_cast<size_t>(parameters_["N_BG_SAMPLES"].toInt()),
                                     static_cast<size_t>(parameters_["N_REQUIRED_BG_SAMPLES"].toInt()),
                                     static_cast<size_t>(parameters_["N_SAMPLES_FOR_MOVING_AVGS"].toInt()));
    bg_initialized_ = false;
    previous_segment_number_ = -1;

}

void SubsenseStreamingDetection::BeginSegment(const VideoSegmentInfo &segment_info) {

    frame_width_ = segment_info.frame_width;
    frame_height_ = segment_info.frame_height;
    current_segment_number_ = segment_info.segment_number;
    segment_frame_index_ = 0;
    // If this segment is not in sequence with the previously
    // processed segment (or its the first segment), then we need to
    // initialize the background subtractor when we get the first
    // frame of the segment.
    if (current_segment_number_ == previous_segment_number_ + 1) {
        bg_initialized_ = false;
    }
    previous_segment_number_ = current_segment_number_;

}


bool SubsenseStreamingDetection::ProcessFrame(const cv::Mat &frame,
                                              int frame_number) {

    bool activity_found = false;
    int downsample_count = 0;
    cv::mat orig_frame;

    frame.copyTo(orig_frame);
    // Downsample frame
    while (frame.rows > parameters["MAXIMUM_FRAME_HEIGHT"].toInt() || frame.cols > parameters["MAXIMUM_FRAME_WIDTH"].toInt()) {
        cv::pyrDown(frame, frame);
        downsample_count++;
    }

    // If the background subtractor has not yet been initialized, we
    // need to use this frame to do that. If it has already been
    // initialized, but this is the first frame in the segment, and
    // the segment number is equal to the previous segment number + 1,
    // then we don't need to initialize and we can simply continue to
    // use the foreground Mat computed during the processing of the
    // last frame of the previous  segment. If the segments are not
    // sequential, all we can do here is re-initialize the background
    // subtractor, and wait for the next frame. This decision
    // processing is done in BeginSegment().
    if ((!bg_initialized_) {
        cv::Mat roi;
        bg_initialize(frame, roi);
        bg_initialized_ = true;
        // No motion detected because this frame had to be used to
        // initialize the background subtractor.
        return false;
    }

    // Run the background subtractor.
    bg_apply(frame, fore_);

}
vector<MPFVideoTrack> EndSegment() {}
