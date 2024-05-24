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

#include "MotionDetection_MOG2.h"

#include <map>
#include <tuple>

#include <opencv2/opencv.hpp>

#include <MPFImageReader.h>
#include <detectionComponentUtils.h>
#include <Utils.h>

#include "struck.h"


using namespace MPF;
using namespace COMPONENT;

using DetectionComponentUtils::GetProperty;

struct MogConfig {
    int verbose;
    int history_length;
    int var_threshold;
    bool background_shadow_detection;
    int maximum_frame_height;
    int maximum_frame_width;
    bool use_preprocessor;
    int erode_anchor_x;
    int erode_anchor_y;
    int erode_iterations;
    int dilate_anchor_x;
    int dilate_anchor_y;
    int dilate_iterations;
    int median_blur_k_size;
    int group_rectangles_group_threshold;
    double group_rectangles_eps;
    int min_rect_width;
    int min_rect_height;
    bool use_motion_tracking;
    double tracking_max_object_percentage;
    double tracking_threshold;
    double tracking_min_overlap_percentage;

    explicit MogConfig(const std::map<std::string, std::string> &props)
            : verbose(GetProperty(props, "VERBOSE", 0))
            , history_length(GetProperty(props, "HISTORY_LENGTH", 500))
            , var_threshold(GetProperty(props, "VAR_THRESHOLD", 16))
            , background_shadow_detection(GetProperty(props, "BACKGROUND_SHADOW_DETECTION", true))
            , maximum_frame_height(GetProperty(props, "MAXIMUM_FRAME_HEIGHT", 128))
            , maximum_frame_width(GetProperty(props, "MAXIMUM_FRAME_WIDTH", 128))
            , use_preprocessor(GetProperty(props, "USE_PREPROCESSOR", false))
            , erode_anchor_x(GetProperty(props, "ERODE_ANCHOR_X", -1))
            , erode_anchor_y(GetProperty(props, "ERODE_ANCHOR_Y", -1))
            , erode_iterations(GetProperty(props, "ERODE_ITERATIONS", 1))
            , dilate_anchor_x(GetProperty(props, "DILATE_ANCHOR_X", -1))
            , dilate_anchor_y(GetProperty(props, "DILATE_ANCHOR_Y", -1))
            , dilate_iterations(GetProperty(props, "DILATE_ITERATIONS", 4))
            , median_blur_k_size(GetProperty(props, "MEDIAN_BLUR_K_SIZE", 3))
            , group_rectangles_group_threshold(GetProperty(props, "GROUP_RECTANGLES_GROUP_THRESHOLD", 1))
            , group_rectangles_eps(GetProperty(props, "GROUP_RECTANGLES_EPS", 0.4))
            , min_rect_width(GetProperty(props, "MIN_RECT_WIDTH", 16))
            , min_rect_height(GetProperty(props, "MIN_RECT_HEIGHT", 16))
            , use_motion_tracking(GetProperty(props, "USE_MOTION_TRACKING", false))
            , tracking_max_object_percentage(GetProperty(props, "TRACKING_MAX_OBJECT_PERCENTAGE", 0.9))
            , tracking_threshold(GetProperty(props, "TRACKING_THRESHOLD", -1))
            , tracking_min_overlap_percentage(GetProperty(props, "TRACKING_MIN_OVERLAP_PERCENTAGE", 0))
    {
    }

};

struct RectLessThan {
    bool operator() (const cv::Rect &r1, const cv::Rect &r2) const {
        return std::tie(r1.x, r1.y, r1.width, r1.height)
               < std::tie(r2.x, r2.y, r2.width, r2.height);
    }
};

void displayTracks(const std::string& origPath, int frameCount,
                   const std::vector<MPFVideoTrack> &tracks);


bool MotionDetection_MOG2::Init() {
    motion_logger_ = log4cxx::Logger::getLogger("MogMotionDetection");
    return true;
}

bool MotionDetection_MOG2::Close() {
    return true;
}


std::vector<MPFVideoTrack> MotionDetection_MOG2::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_DEBUG(motion_logger_, "Starting motion detection");

        MogConfig config(job.job_properties);

        MPFVideoCapture video_capture(job, true, true);

        std::vector<MPFVideoTrack> tracks = GetDetectionsFromVideoCapture(
                job, config, video_capture);
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


std::vector<MPFVideoTrack> MotionDetection_MOG2::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job, const MogConfig &config, MPFVideoCapture &video_capture) {

    MPFVideoTrack track;
    int downsample_count = 0;
    cv::Mat orig_frame, frame, fore;
    std::vector<std::vector<cv::Point>> contours;
    std::map<int, STRUCK> tracker_map;
    std::map<int, MPFVideoTrack> track_map;
    int tracker_id = 0;

    // Set track for default
    track.start_frame = -1;


    // Create background subtractor
    LOG4CXX_TRACE(motion_logger_, "Creating background subtractor");
    int history_length = config.history_length;
    int var_threshold = config.var_threshold;
    bool detectShadows = config.background_shadow_detection;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg = cv::createBackgroundSubtractorMOG2(
            history_length, static_cast<double>(var_threshold), detectShadows);

    // Create video capture and set start frame

    int frame_count = video_capture.GetFrameCount();
    LOG4CXX_DEBUG(motion_logger_, "frame count = " << frame_count);
    LOG4CXX_DEBUG(motion_logger_, "begin frame = " << job.start_frame);
    LOG4CXX_DEBUG(motion_logger_, "end frame = " << job.stop_frame);


    int frame_index;
    const std::vector<cv::Mat> &init_frames = video_capture.GetInitializationFramesIfAvailable(1);
    // Attempt to use the frame before the start of the segment to initialize the foreground.
    // If one is not available, use frame 0 and start processing at frame 1.
    if (init_frames.empty()) {
        frame_index = 1;
        video_capture.Read(orig_frame);
    }
    else {
        frame_index = 0;
        orig_frame = init_frames.at(0);
    }

    if (orig_frame.empty()) {
        throw MPFDetectionException(MPF_BAD_FRAME_SIZE);
    }

    // Calculate downsample rate
    orig_frame.copyTo(frame);
    while (frame.rows > config.maximum_frame_height || frame.cols > config.maximum_frame_width) {
        cv::pyrDown(frame, frame);
        downsample_count++;
    }

    // Initialize the foreground
    bg->apply(frame, fore);


    LOG4CXX_TRACE(motion_logger_, "Starting video processing");

    std::vector<MPFVideoTrack> tracks;
    while (video_capture.Read(frame)) {
        std::vector<cv::Rect> rects;
        LOG4CXX_DEBUG(motion_logger_, "frame index = " << frame_index);

        if (frame.empty()) {
            LOG4CXX_DEBUG(motion_logger_, "Empty frame encountered at frame " << frame_index);
            break;
        }

        // Downsample frame
        for (int x = 0; x < downsample_count; ++x) {
            cv::pyrDown(frame, frame);
        }

        bg->apply(frame, fore);

        if (config.use_preprocessor) {
            if (cv::countNonZero(fore) > 0) {
                if (track.start_frame == -1) {
                    track.start_frame = frame_index;
                    track.stop_frame = track.start_frame;
                    track.frame_locations.emplace(
                            frame_index,
                            MPFImageLocation(0, 0,
                                             static_cast<int>(orig_frame.cols),
                                             static_cast<int>(orig_frame.rows)));
                } else {
                    track.stop_frame = frame_index;
                    track.frame_locations.emplace(
                            frame_index,
                            MPFImageLocation(0, 0, orig_frame.cols, orig_frame.rows));
                }
            } else {
                // Here, we have stopped detecting motion, so we
                // complete the track and save it.
                if (track.start_frame != -1) {
                    tracks.push_back(track);

                    // Clear track variable
                    track.start_frame = -1;
                    track.stop_frame = -1;
                    track.frame_locations.clear();
                }
            }
        } else {

            // Make motion larger objects
            LOG4CXX_TRACE(motion_logger_, "Eroding and blurring background");
            cv::erode(fore, fore, cv::Mat(),
                      cv::Point(config.erode_anchor_x, config.erode_anchor_y),
                      config.erode_iterations);
            cv::dilate(fore, fore, cv::Mat(),
                       cv::Point(config.dilate_anchor_x, config.dilate_anchor_y),
                       config.dilate_iterations);
            cv::medianBlur(fore, fore, config.median_blur_k_size);

            // Find the contours and then make bounding rects
            LOG4CXX_TRACE(motion_logger_, "Finding contours and combine overlaps");
            cv::findContours(fore, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


            for (const std::vector<cv::Point> &contour : contours) {
                // Need to add the rect twice so that it doesnt get removed
                rects.push_back(cv::boundingRect(contour));
                rects.push_back(cv::boundingRect(contour));
            }

            // Combines overlapping rects together
            LOG4CXX_TRACE(motion_logger_, "Converting contours to rects");

            cv::groupRectangles(
                    rects, config.group_rectangles_group_threshold, config.group_rectangles_eps);

            LOG4CXX_TRACE(motion_logger_, "Writing rects to VideoTrack");
            std::vector<cv::Rect> resized_rects;
            for (const cv::Rect &rect : rects) {
                if ((rect.width * pow(2, downsample_count)) >= config.min_rect_width &&
                    (rect.height * pow(2, downsample_count)) >= config.min_rect_height) {

                    if (config.use_motion_tracking) {
                        resized_rects.push_back(Upscale(rect, orig_frame, downsample_count));
                    } else {
                        MPFVideoTrack track;
                        track.start_frame = frame_index;
                        track.stop_frame = track.start_frame;
                        cv::Rect resized = Upscale(rect, orig_frame, downsample_count);
                        track.frame_locations.emplace(
                                frame_index,
                                MPFImageLocation(resized.x, resized.y,
                                                 resized.width, resized.height));

                        tracks.push_back(track);
                    }
                }
            }

            if (config.use_motion_tracking) {
                std::map<cv::Rect, std::vector<int>, RectLessThan> tracked_rects;
                size_t total_tracked_rects = 0;

                // Append to tracks
                for (auto it = tracker_map.begin(); it != tracker_map.end();) {
                    cv::Rect rect = it->second.nextFrame(orig_frame, resized_rects);
                    // A rect with starting point 0,0 and w/h of 1,1 will mark the end of the track
                    if (rect == cv::Rect(0, 0, 1, 1)) {
                        tracks.push_back(track_map.at(it->first));
                        track_map.erase(track_map.find(it->first));
                        it = tracker_map.erase(it);
                    } else {
                        auto &track_to_update = track_map.at(it->first);
                        track_to_update.stop_frame = frame_index;
                        LOG4CXX_DEBUG(motion_logger_, __LINE__ << " stop_frame = " << track_to_update.stop_frame);
                        track_to_update.frame_locations.emplace(
                                frame_index,
                                MPFImageLocation(rect.x, rect.y, rect.width, rect.height));
                        tracked_rects[rect].push_back(it->first);
                        total_tracked_rects++;
                        ++it;
                    }
                }

                // Create new tracks
                if (total_tracked_rects != resized_rects.size()) {
                    for (const cv::Rect &rect : resized_rects) {
                        if (tracked_rects.count(rect) == 0 || tracked_rects.at(rect).empty()) {
                            if ((orig_frame.rows * orig_frame.cols * config.tracking_max_object_percentage) < rect.area()) {
                                continue;
                            }
                            auto& new_struck = tracker_map.emplace(++tracker_id, STRUCK()).first->second;
                            new_struck.initialize(
                                    orig_frame, rect,
                                    config.tracking_threshold,
                                    config.tracking_min_overlap_percentage);

                            MPFVideoTrack temp(frame_index, frame_index);
                            temp.frame_locations.emplace(
                                    frame_index,
                                    MPFImageLocation(rect.x, rect.y, rect.width, rect.height));

                            track_map.emplace(tracker_id, std::move(temp));
                            tracked_rects[rect].push_back(tracker_id);
                        }
                    }
                }

                // Remove merged tracks
                for (const auto &pair : tracked_rects) {
                    const auto& rect = pair.first;
                    const auto& ids = pair.second;
                    for (int id_idx = 1; id_idx < ids.size(); ++id_idx) {
                        int id = ids.at(id_idx);
                        tracks.push_back(std::move(track_map.at(id)));

                        tracker_map.erase(id);
                        track_map.erase(id);
                    }
                }
            }
        }
        frame_index++;
    }


    // Finish the last track if we ended iteration through the video with an open track.
    if (config.use_preprocessor && track.start_frame != -1) {
        tracks.push_back(track);
    }

    // Complete open tracks
    for (auto& pair : tracker_map) {
        tracks.push_back(track_map.at(pair.first));
    }


    // Close video capture
    video_capture.Release();

    if (config.verbose > 0) {
        // Now print tracks if available
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
        } else {
            LOG4CXX_DEBUG(motion_logger_, "No tracks found");
        }
    }

    LOG4CXX_INFO(motion_logger_, "Processing complete. Found " << tracks.size() << " tracks.");

    return tracks;
}


std::vector<MPFImageLocation> MotionDetection_MOG2::GetDetections(const MPFImageJob &job) {
    try {
        std::vector<MPFImageLocation> locations;
        // if this component is used as a preprocessor then it will return that it detects motion in every image
        // (although no actual motion detection is performed)
        if (MogConfig(job.job_properties).use_preprocessor) {
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
void displayTracks(const std::string& origPath, int frameCount,
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


cv::Rect MotionDetection_MOG2::Upscale(const cv::Rect &rect, const cv::Mat &orig_frame, int downsample_count) {
    cv::Rect resized(rect.x * pow(2, downsample_count), rect.y * pow(2, downsample_count),
                     rect.width * pow(2, downsample_count), rect.height * pow(2, downsample_count));

    // NOTE: When pyrDown() is used to downsample an image with a width and/or height that is not a
    // power of 2, the resulting width and/or height value will be rounded up to the nearest integer.
    // Thus, when we calculate the resized rectangle it may extend beyond the bounds of the original
    // image, so we have to correct for the overflow.

    int overflow_x = resized.x + resized.width - orig_frame.cols;
    if (overflow_x > 0) {
        resized.width -= overflow_x;
    }

    int overflow_y = resized.y + resized.height - orig_frame.rows;
    if (overflow_y > 0) {
        resized.height -= overflow_y;
    }

    return resized;
}


MPF_COMPONENT_CREATOR(MotionDetection_MOG2);
MPF_COMPONENT_DELETER();
