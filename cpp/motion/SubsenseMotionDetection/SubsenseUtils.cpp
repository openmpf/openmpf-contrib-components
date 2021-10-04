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

#include "SubsenseUtils.h"

#include <tuple>

#include <detectionComponentUtils.h>

using namespace MPF;
using namespace COMPONENT;

using DetectionComponentUtils::GetProperty;

SubsenseConfig::SubsenseConfig(const std::map<std::string, std::string> &props)
        : verbose(GetProperty(props, "VERBOSE", 0))
        , f_rel_lbsp_threshold(GetProperty(props, "F_REL_LBSP_THRESHOLD", 0.333f))
        , n_min_desc_dist_threshold(GetProperty(props, "N_MIN_DESC_DIST_THRESHOLD", 3))
        , n_min_color_dist_threshold(GetProperty(props, "N_MIN_COLOR_DIST_THRESHOLD", 30))
        , n_bg_samples(GetProperty(props, "N_BG_SAMPLES", 50))
        , n_required_bg_samples(GetProperty(props, "N_REQUIRED_BG_SAMPLES", 2))
        , n_samples_for_moving_avgs(GetProperty(props, "N_SAMPLES_FOR_MOVING_AVGS", 25))
        , maximum_frame_width(GetProperty(props, "MAXIMUM_FRAME_WIDTH", 128))
        , maximum_frame_height(GetProperty(props, "MAXIMUM_FRAME_HEIGHT", 128))
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
        , use_preprocessor(GetProperty(props, "USE_PREPROCESSOR", false))
        , use_motion_tracking(GetProperty(props, "USE_MOTION_TRACKING", false))
        , distance_confidence_weight_factor(GetProperty(props, "DISTANCE_CONFIDENCE_WEIGHT_FACTOR", 0.0f))
        , size_confidence_weight_factor(GetProperty(props, "SIZE_CONFIDENCE_WEIGHT_FACTOR", 0.0f))
        , tracking_max_object_percentage(GetProperty(props, "TRACKING_MAX_OBJECT_PERCENTAGE", 0.9f))
        , tracking_threshold(GetProperty(props, "TRACKING_THRESHOLD", -1))
        , tracking_min_overlap_percentage(GetProperty(props, "TRACKING_MIN_OVERLAP_PERCENTAGE", 0.0))
{
}


void SetPreprocessorTrack(const cv::Mat &fore, int frame_index,
                          int frame_cols, int frame_rows,
                          MPFVideoTrack &track,
                          std::vector<MPFVideoTrack> &tracks) {

    if (cv::countNonZero(fore) > 0) {
        if (track.start_frame == -1) {
            track.start_frame = frame_index;
            track.stop_frame = track.start_frame;
            track.frame_locations.emplace(
                    frame_index,
                    MPFImageLocation(0, 0, frame_cols, frame_rows));
        } else {
            track.stop_frame = frame_index;
            track.frame_locations.emplace(
                    frame_index,
                MPFImageLocation(0, 0, frame_cols, frame_rows));
        }
    } else {
        if (track.start_frame != -1) {
            tracks.push_back(track);

            // Clear track variable
            track.start_frame = -1;
            track.stop_frame = -1;
            track.frame_locations.clear();
        }
    }

}

std::vector<cv::Rect> GetResizedRects(const std::string &job_name,
                                      const log4cxx::LoggerPtr &logger,
                                      const SubsenseConfig &config,
                                      cv::Mat &fore,
                                      int frame_cols,
                                      int frame_rows,
                                      int downsample_count) {

    std::vector<cv::Rect> rects;
    // Make motion larger objects
    LOG4CXX_TRACE(logger, "[" << job_name << "] Eroding and blurring background");
    cv::erode(fore, fore, cv::Mat(),
              cv::Point(config.erode_anchor_x, config.erode_anchor_y),
              config.erode_iterations);
    cv::dilate(fore, fore, cv::Mat(),
               cv::Point(config.dilate_anchor_x, config.dilate_anchor_y),
               config.dilate_iterations);
    cv::medianBlur(fore, fore, config.median_blur_k_size);


    std::vector<std::vector<cv::Point> > contours;
    // Find the contours and then make bounding rects
    LOG4CXX_TRACE(logger, "[" << job_name << "] Finding contours and combine overlaps");
    cv::findContours(fore, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


    for (const std::vector<cv::Point> &contour : contours) {
        // Need to add the rect twice so that it doesnt get removed
        rects.push_back(cv::boundingRect(contour));
        rects.push_back(cv::boundingRect(contour));
    }

    // Combines overlapping rects together
    LOG4CXX_TRACE(logger, "[" << job_name << "] Converting contours to rects");

    cv::groupRectangles(rects, config.group_rectangles_group_threshold,
                        config.group_rectangles_eps);

    LOG4CXX_TRACE(logger, "[" << job_name << "] Resizing rects");
    std::vector<cv::Rect> resized_rects;
    for (const cv::Rect &rect : rects) {
        if ((rect.width * pow(2, downsample_count)) >= config.min_rect_width &&
            rect.height * pow(2, downsample_count) >= config.min_rect_height) {
            resized_rects.push_back(Upscale(rect, frame_cols, frame_rows, downsample_count));
        }
    }
    return resized_rects;
}

struct RectLessThan {
    bool operator() (const cv::Rect &r1, const cv::Rect &r2) const {
        return std::tie(r1.x, r1.y, r1.width, r1.height)
               < std::tie(r2.x, r2.y, r2.width, r2.height);
    }
};


void ProcessMotionTracks(const SubsenseConfig &config,
                         const std::vector<cv::Rect> &resized_rects,
                         const cv::Mat &orig_frame,
                         int frame_index, int &tracker_id,
                         std::map<int, STRUCK> &tracker_map,
                         std::map<int, MPFVideoTrack> &track_map,
                         std::vector<MPFVideoTrack> &tracks) {

    std::map<cv::Rect, std::vector<int>, RectLessThan> tracked_rects;
    size_t total_tracked_rects = 0;

    // Append to tracks
    for (auto tracker_it = tracker_map.begin(); tracker_it != tracker_map.end();) {
        cv::Rect rect = tracker_it->second.nextFrame(orig_frame, resized_rects);
        // A rect with starting point 0,0 and w/h of 1,1 will mark the end of the track
        if (rect == cv::Rect(0, 0, 1, 1)) {
            tracks.push_back(std::move(track_map.at(tracker_it->first)));
            track_map.erase(tracker_it->first);
            tracker_it = tracker_map.erase(tracker_it);
        }
        else {
            auto &track = track_map.at(tracker_it->first);
            track.stop_frame = frame_index;
            track.frame_locations.emplace(
                    frame_index, MPFImageLocation(rect.x, rect.y, rect.width, rect.height));

            tracked_rects[rect].push_back(tracker_it->first);
            total_tracked_rects++;
            ++tracker_it;
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
                        frame_index, MPFImageLocation(rect.x, rect.y, rect.width, rect.height));

                track_map.emplace(tracker_id, std::move(temp));
                tracked_rects[rect].push_back(tracker_id);
            }
        }
    }

    // Remove merged tracks
    for (const auto& pair : tracked_rects) {
        const cv::Rect &rect = pair.first;
        const std::vector<int> &ids = pair.second;
        for (int id_idx = 1; id_idx < ids.size(); id_idx++) {
            int id = ids.at(id_idx);
            tracks.push_back(std::move(track_map.at(id)));

            tracker_map.erase(id);
            track_map.erase(id);
        }
    }
}


cv::Rect Upscale(const cv::Rect &rect,
                 int frame_cols, int frame_rows,
                 int downsample_count) {

    cv::Rect resized(rect.x * pow(2, downsample_count),
                     rect.y * pow(2, downsample_count),
                     rect.width * pow(2, downsample_count),
                     rect.height * pow(2, downsample_count));

    // NOTE: When pyrDown() is used to downsample an image with a
    // width and/or height that is not a power of 2, the resulting
    // width and/or height value will be rounded up to the nearest
    // integer. Thus, when we calculate the resized rectangle it may
    // extend beyond the bounds of the original image, so we have to
    // correct for the overflow.

    int overflow_x = resized.x + resized.width - frame_cols;
    if (overflow_x > 0) {
        resized.width -= overflow_x;
    }

    int overflow_y = resized.y + resized.height - frame_rows;
    if (overflow_y > 0) {
        resized.height -= overflow_y;
    }

    return resized;
}


void AssignDetectionConfidence(MPFVideoTrack &track, float distance_factor,
                               float size_factor) {

    // Make it so that both weights are positive
    distance_factor = (distance_factor < 0.0) ? 0.0 : distance_factor;
    size_factor = (size_factor < 0.0) ? 0.0 : size_factor;
    if ((distance_factor > 0.0) || (size_factor > 0.0)) {

        // Distance confidence: Detections closer to the "center" of the
        // track have higher confidence. Determine which detection is at the center
        // of the track, and assign it the highest confidence (1.0). The rest of the
        // detections are assigned a confidence based on their distance
        // from the "center", with the first and last detections in the
        // track having confidence equal to 0. If there are an even number of
        // detections in the track, then the two detections nearest the
        // track center will both be given confidence equal to 1.0.

        // Size confidence: Detections that have greater area in their
        // bounding boxes have higher confidence. Each detection has a
        // size confidence equal to the ratio of the size of its bounding
        // box to the maximum size bounding box in the track.

        int number_of_detections = track.frame_locations.size();
        float max_size = 0.0;
        for (auto &entry: track.frame_locations) {
            float entry_size = entry.second.width * entry.second.height;
            max_size = (entry_size > max_size) ? entry_size : max_size;
        }

        if (number_of_detections <= 2) {
            for (auto &entry : track.frame_locations) {
                MPFImageLocation &loc = entry.second;
                loc.confidence = distance_factor + size_factor*(static_cast<float>(loc.height*loc.width)/max_size);
            }
        }
        else {
            float center = number_of_detections/2.0;
            int center_index = static_cast<int>(center);
            float confidence_incr = (center_index < center) ? 1.0/center_index : 1.0/(center_index - 1);

            auto first = track.frame_locations.begin();
            auto last = track.frame_locations.rbegin();
            for (int i = 0; i < center_index; ++i) {
                MPFImageLocation &loc1 = (first++)->second;
                MPFImageLocation &loc2 = (last++)->second;
                loc1.confidence = distance_factor*(i*confidence_incr) +
                                  size_factor*((loc1.height*loc1.width)/max_size);
                loc2.confidence = distance_factor*(i*confidence_incr) +
                                  size_factor*((loc2.height*loc2.width)/max_size);
            }
            if (center_index < center) {
                MPFImageLocation &loc1 = (first)->second;
                loc1.confidence = distance_factor +
                                  size_factor*((loc1.height*loc1.width)/max_size);
            }
        }
        // Finally, normalize all confidences to between 0 and 1.
        float max_confidence = 0.0;
        for (auto &entry : track.frame_locations) {
            max_confidence = (entry.second.confidence > max_confidence) ? entry.second.confidence : max_confidence;
        }
        for (auto &entry : track.frame_locations) {
            MPFImageLocation &loc = entry.second;
            loc.confidence = loc.confidence/max_confidence;
        }
    }
}
