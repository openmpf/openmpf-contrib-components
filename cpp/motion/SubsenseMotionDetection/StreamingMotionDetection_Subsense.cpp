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
 * Copyright 2021 The MITRE Corporation                                       *
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

#include <MPFSimpleConfigLoader.h>
#include "StreamingMotionDetection_Subsense.h"
#include "SubsenseUtils.h"

using namespace MPF;
using namespace COMPONENT;

using std::string;
using std::map;
using std::pair;
using std::vector;

SubsenseStreamingDetection::SubsenseStreamingDetection(const MPFStreamingVideoJob &job)
        : SubsenseStreamingDetection(job, SubsenseConfig(job.job_properties)) {
}


SubsenseStreamingDetection::SubsenseStreamingDetection(
            const MPFStreamingVideoJob &job, const SubsenseConfig &config)
        : MPFStreamingDetectionComponent(job)
        , config_(config)
        , job_name_(job.job_name)
        , msg_prefix_("[" + job_name_ + "] ")
        , motion_logger_(log4cxx::Logger::getLogger("SubsenseStreamingMotionDetection"))
        , bg_(config_.f_rel_lbsp_threshold,
              config_.n_min_desc_dist_threshold,
              config_.n_min_color_dist_threshold,
              config_.n_bg_samples,
              config_.n_required_bg_samples,
              config_.n_samples_for_moving_avgs) {
    if (config_.verbose > 0) {
        motion_logger_->setLevel(log4cxx::Level::getDebug());
    }
}



void SubsenseStreamingDetection::BeginSegment(const VideoSegmentInfo &segment_info) {

    frame_width_ = segment_info.frame_width;
    frame_height_ = segment_info.frame_height;
    current_segment_number_ = segment_info.segment_number;
    segment_frame_index_ = 0;
    tracks_.clear();

    // If this is the first segment to be processed, then
    // "bg_initialized_" will be set to false.
    //
    // If the background subtractor has already been initialized, then
    // we will need to re-initialize it if the current segment is not
    // in sequence with the previously processed segment.
    if ((bg_initialized_) && (current_segment_number_ != previous_segment_number_ + 1)) {
        bg_initialized_ = false;
        downsample_count_ = 0;
    }
    previous_segment_number_ = current_segment_number_;
    LOG4CXX_INFO(motion_logger_, msg_prefix_ << "Begin segment #" << current_segment_number_);
    LOG4CXX_DEBUG(motion_logger_, msg_prefix_ << "previous segment number = " << previous_segment_number_);
}


bool SubsenseStreamingDetection::ProcessFrame(const cv::Mat &orig_frame,
                                              int frame_number) {

    bool frame_activity_found = false;
    cv::Mat frame, fore;

    // Initialization: Use this frame to initialize but don't do detection.
    if (!bg_initialized_) {
        // Downsample frame and initialize the downsample count.
        // Since the input frame is read only, the first call to
        // pyrDown() needs to be out of place.
        if (orig_frame.rows > config_.maximum_frame_height ||
            orig_frame.cols > config_.maximum_frame_width) {
            cv::pyrDown(orig_frame, frame);
            downsample_count_++;
        }
        else {
            orig_frame.copyTo(frame);
        }

        while (frame.rows > config_.maximum_frame_height ||
               frame.cols > config_.maximum_frame_width) {
            cv::pyrDown(frame, frame);
            downsample_count_++;
        }

        cv::Mat roi;
        bg_.initialize(frame, roi);
        bg_initialized_ = true;
        segment_frame_index_++;
        LOG4CXX_INFO(motion_logger_, msg_prefix_ << "Background subtractor initialized.");
        // No motion detected because this frame had to be used to
        // initialize the background subtractor.
        return false;
    }

    // Steady state

    if (downsample_count_ > 0) {
        cv::pyrDown(orig_frame, frame);
        for (int i = 0; i < (downsample_count_ - 1); ++i) {
            cv::pyrDown(frame, frame);
        }
    }
    else {
        orig_frame.copyTo(frame);
    }

    // Run the background subtractor.
    LOG4CXX_TRACE(motion_logger_, msg_prefix_ << __FUNCTION__ << ": " << __LINE__ << ": Apply");
    bg_.apply(frame, fore);

    if (config_.use_preprocessor) {
        LOG4CXX_TRACE(motion_logger_, msg_prefix_ << __FUNCTION__ << ": " << __LINE__);
        SetPreprocessorTrack(fore, segment_frame_index_,
                             frame_width_, frame_height_,
                             preprocessor_track_, tracks_);
        if (!segment_activity_reported_) {
            // See if we started a track
            if (preprocessor_track_.start_frame != -1) {
                frame_activity_found = true;
                segment_activity_reported_ = true;
            }
        }
    }
    else {
        LOG4CXX_TRACE(motion_logger_, msg_prefix_ << __FUNCTION__ << ": " << __LINE__);
        vector<cv::Rect> resized_rects = GetResizedRects(job_name_,
                                                         motion_logger_,
                                                         config_,
                                                         fore,
                                                         frame_width_,
                                                         frame_height_,
                                                         downsample_count_);

        if (config_.use_motion_tracking) {
            LOG4CXX_TRACE(motion_logger_, msg_prefix_ << __FUNCTION__ << ": " << __LINE__);
            ProcessMotionTracks(config_, resized_rects, orig_frame,
                                segment_frame_index_, tracker_id_,
                                tracker_map_, track_map_, tracks_);
            if (!segment_activity_reported_) {
                if (!tracks_.empty()) {
                    frame_activity_found = true;
                    segment_activity_reported_ = true;
                }
            }
        }
        else {
            LOG4CXX_TRACE(motion_logger_, msg_prefix_ << __FUNCTION__ << ": " << __LINE__);

            if (!segment_activity_reported_) {
                if (!resized_rects.empty()) {
                    frame_activity_found = true;
                    segment_activity_reported_ = true;
                }
            }

            for (const cv::Rect &rect : resized_rects) {
                MPFVideoTrack track;
                track.start_frame = segment_frame_index_;
                track.stop_frame = track.start_frame;
                track.frame_locations.insert(
                    std::pair<int, MPFImageLocation>(segment_frame_index_,
                                                     MPFImageLocation(rect.x, rect.y,
                                                                      rect.width,
                                                                      rect.height)));

                tracks_.push_back(std::move(track));
            }
        }
    }
    frame_number_map_[segment_frame_index_++] = frame_number;

    return frame_activity_found;
}
vector<MPFVideoTrack> SubsenseStreamingDetection::EndSegment() {

    segment_activity_reported_ = false;

    // Finish the last track if we ended iteration through the video with an open track.
    if (config_.use_preprocessor && preprocessor_track_.start_frame != -1) {
        tracks_.push_back(std::move(preprocessor_track_));
    }

    preprocessor_track_ = MPFVideoTrack();

    // Complete open tracks
    for (const auto &pair : tracker_map_) {
        tracks_.push_back(track_map_.at(pair.first));
    }

    // The frame numbers in the track are segment frame indices; i.e.,
    // they can range from 0 to the segment length. They need to be
    // converted to the true frame number. The track frame index is
    // used to look up the true frame number in the
    // frame_number_map_.

    for (auto &track : tracks_) {
        track.start_frame = frame_number_map_[track.start_frame];
        track.stop_frame = frame_number_map_[track.stop_frame];
        std::map<int, MPFImageLocation> new_locations;
        for (auto &loc : track.frame_locations) {
            int frame_index = frame_number_map_[loc.first];
            new_locations.emplace(frame_index, std::move(loc.second));
        }
        track.frame_locations = std::move(new_locations);
    }

    // Assign a confidence value to each detection
    float distance_factor = config_.distance_confidence_weight_factor;
    float size_factor = config_.size_confidence_weight_factor;
    for(MPFVideoTrack &track : tracks_) {
        AssignDetectionConfidence(track, distance_factor, size_factor);
    }

    track_map_.clear();
    tracker_map_.clear();
    tracker_id_ = 0;
    LOG4CXX_INFO(motion_logger_, msg_prefix_ << "End segment #" << current_segment_number_);
    LOG4CXX_INFO(motion_logger_, msg_prefix_ << tracks_.size() << " tracks reported.");
    vector<MPFVideoTrack> tmp;
    tmp.swap(tracks_);
    return tmp;
}


EXPORT_MPF_STREAMING_COMPONENT(SubsenseStreamingDetection)
