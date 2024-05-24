/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2024 The MITRE Corporation. All Rights Reserved.                 *
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

#include "MotionDetection_Subsense.h"

#include <map>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <Utils.h>
#include <MPFImageReader.h>
#include <MPFDetectionException.h>

#include "SubSense/BackgroundSubtractorSuBSENSE.h"
#include "struck.h"
#include "SubsenseUtils.h"


using namespace MPF;
using namespace COMPONENT;

void displayTracks(const std::string &origPath, int frameCount,
                   const std::vector<MPFVideoTrack> &tracks);

bool MotionDetection_Subsense::Init() {
    motion_logger_ = log4cxx::Logger::getLogger("SubsenseMotionDetection");
    return true;
}

bool MotionDetection_Subsense::Close() {
    return true;
}


std::vector<MPFVideoTrack> MotionDetection_Subsense::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_DEBUG(motion_logger_, "Starting motion detection");
        SubsenseConfig config(job.job_properties);

        MPFVideoCapture video_capture(job, true, true);

        std::vector<MPFVideoTrack> tracks = GetDetectionsFromVideoCapture(
                job, video_capture, config);
        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }

        if (config.verbose > 1) {
            displayTracks(job.data_uri, video_capture.GetFrameCount(), tracks);
        }

        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, motion_logger_);
    }
}

std::vector<MPFVideoTrack> MotionDetection_Subsense::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job, MPFVideoCapture &video_capture, const SubsenseConfig &config) {

    int downsample_count = 0;
    cv::Mat orig_frame, frame, fore;
    std::map<int, STRUCK> tracker_map;
    std::map<int, MPFVideoTrack> track_map;
    MPFVideoTrack preprocessor_track;
    int tracker_id = 0;

    // Set track used for "preprocessing mode" to default
    preprocessor_track.start_frame = -1;


    // Create background subtractor
    LOG4CXX_TRACE(motion_logger_, "Creating background subtractor");
    BackgroundSubtractorSuBSENSE bg(
            config.f_rel_lbsp_threshold,
            config.n_min_desc_dist_threshold,
            config.n_min_color_dist_threshold,
            config.n_bg_samples,
            config.n_required_bg_samples,
            config.n_samples_for_moving_avgs);


    int frame_index;
    const std::vector<cv::Mat> &init_frames = video_capture.GetInitializationFramesIfAvailable(1);
    // Attempt to use the frame before the start of the segment to initialize the foreground.
    // If one is not available, use frame 0 and start processing at frame 1.
    if (init_frames.empty()) {
        frame_index = 1;
        video_capture.Read(frame);
    }
    else {
        frame_index = 0;
        frame = init_frames.at(0);
    }


    if (frame.empty()) {
        throw MPFDetectionException(MPF_BAD_FRAME_SIZE);
    }

    frame.copyTo(orig_frame);
    // Downsample frame
    while (frame.rows > config.maximum_frame_height || frame.cols > config.maximum_frame_width) {
        cv::pyrDown(frame, frame);
        downsample_count++;
    }

    cv::Mat Roi;
    bg.initialize(frame, Roi);


    LOG4CXX_TRACE(motion_logger_, " Starting video processing");

    std::vector<MPFVideoTrack> tracks;
    while (video_capture.Read(frame)) {

        if (frame.empty()) {
            LOG4CXX_DEBUG(motion_logger_, "Empty frame encountered at frame " << frame_index);
            break;
        }
        LOG4CXX_DEBUG(motion_logger_, "frame index = " << frame_index);

        // Downsample frame
        for (int x = 0; x < downsample_count; ++x) {
            cv::pyrDown(frame, frame);
        }

        bg.apply(frame, fore);

        if (config.use_preprocessor) {
            SetPreprocessorTrack(fore, frame_index,
                                 orig_frame.cols, orig_frame.rows,
                                 preprocessor_track, tracks);
        }
        else {
            std::vector<cv::Rect> resized_rects = GetResizedRects(motion_logger_,
                                                                  config,
                                                                  fore,
                                                                  orig_frame.cols,
                                                                  orig_frame.rows,
                                                                  downsample_count);

            if (config.use_motion_tracking) {
                ProcessMotionTracks(config, resized_rects, orig_frame,
                                    frame_index, tracker_id,
                                    tracker_map, track_map, tracks);
            }
            else {
                for(const cv::Rect &rect : resized_rects) {
                    MPFVideoTrack track;
                    track.start_frame = frame_index;
                    track.stop_frame = track.start_frame;
                    track.frame_locations.emplace(
                            frame_index,
                            MPFImageLocation(rect.x, rect.y, rect.width, rect.height));

                    tracks.push_back(track);
                }
            }
        }
        frame_index++;
    }

    // Finish the last track if we ended iteration through the video with an open track.
    if (config.use_preprocessor && preprocessor_track.start_frame != -1) {
        tracks.push_back(preprocessor_track);
    }

    // Complete open tracks
    for (const auto& pair : tracker_map) {
        tracks.push_back(std::move(track_map.at(pair.first)));
    }

    // Close video capture
    video_capture.Release();

    // Assign a confidence value to each detection
    float distance_factor = config.distance_confidence_weight_factor;
    float size_factor = config.size_confidence_weight_factor;
    for(MPFVideoTrack &track : tracks) {
        AssignDetectionConfidence(track, distance_factor, size_factor);
    }

    if (config.verbose > 0) {
        //now print tracks if available
        if (!tracks.empty()) {
            for (unsigned int i=0; i<tracks.size(); i++) {
                LOG4CXX_DEBUG(motion_logger_, "Track index: " << i);
                LOG4CXX_DEBUG(motion_logger_, "Track start frame: " << tracks[i].start_frame);
                LOG4CXX_DEBUG(motion_logger_, "Track end frame: " << tracks[i].stop_frame);

                for (std::map<int, MPFImageLocation>::const_iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it) {
                    LOG4CXX_DEBUG(motion_logger_, "Frame num: " << it->first);
                    LOG4CXX_DEBUG(motion_logger_, "Bounding rect: (" << it->second.x_left_upper << ", " <<
                                                     it->second.y_left_upper << ", " << it->second.width << ", " << it->second.height <<
                                                     ")");
                    LOG4CXX_DEBUG(motion_logger_, "Confidence: " << it->second.confidence);
                }
            }
        }
        else {
            LOG4CXX_DEBUG(motion_logger_, "No tracks found");
        }
    }

    LOG4CXX_INFO(motion_logger_, "Processing complete. Found " << tracks.size() << " tracks.");

    return tracks;
}

std::vector<MPFImageLocation> MotionDetection_Subsense::GetDetections(const MPFImageJob &job) {
    try {
        SubsenseConfig config(job.job_properties);

        std::vector<MPFImageLocation> locations;
        // if this component is used as a preprocessor then it will return that it detects motion in every image
        // (although no actual motion detection is performed)
        if (config.use_preprocessor) {
            MPFImageLocation detection;
            MPFImageReader image_reader(job);
            cv::Mat cv_image = image_reader.GetImage();

            detection.x_left_upper = 0;
            detection.y_left_upper = 0;
            detection.width = cv_image.cols;
            detection.height = cv_image.rows;
            locations.push_back(detection);

            for (auto &location : locations) {
                image_reader.ReverseTransform(location);
            }
        }

        LOG4CXX_INFO(motion_logger_,
                     "Processing complete. Found " << locations.size() << " detections.");
        return locations;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, motion_logger_);
    }
}



// NOTE: This only draws a bounding box around the first detection in each track
void displayTracks(const std::string &origPath, int frameCount,
                   const std::vector<MPFVideoTrack> &tracks) {
    cv::VideoCapture capture(origPath);

    cv::Mat frame;
    auto it = tracks.begin();

    while (capture.get(cv::CAP_PROP_POS_FRAMES) < frameCount) {
        capture >> frame;

        while (it != tracks.end() && it->start_frame == capture.get(cv::CAP_PROP_POS_FRAMES)-1) {
            cv::Rect rect;

            rect.x = it->frame_locations.begin()->second.x_left_upper;
            rect.y = it->frame_locations.begin()->second.y_left_upper;
            rect.width = it->frame_locations.begin()->second.width;
            rect.height = it->frame_locations.begin()->second.height;
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
            ++it;
        }
        cv::imshow("Any key to continue; 'q'' to quit", frame);
        if (cv::waitKey() == 'q') {
            break;
        }
    }
}


MPF_COMPONENT_CREATOR(MotionDetection_Subsense);
MPF_COMPONENT_DELETER();
