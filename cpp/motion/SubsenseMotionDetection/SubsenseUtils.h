/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
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

#ifndef OPENMPF_CONTRIB_COMPONENTS_SUBSENSE_UTILS_H
#define OPENMPF_CONTRIB_COMPONENTS_SUBSENSE_UTILS_H

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <log4cxx/logger.h>

#include <MPFDetectionObjects.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include <struck.h>
#pragma GCC diagnostic pop


struct SubsenseConfig {
    int verbose;
    float f_rel_lbsp_threshold;
    size_t n_min_desc_dist_threshold;
    size_t n_min_color_dist_threshold;
    size_t n_bg_samples;
    size_t n_required_bg_samples;
    size_t n_samples_for_moving_avgs;
    int maximum_frame_width;
    int maximum_frame_height;
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
    bool use_preprocessor;
    bool use_motion_tracking;
    float distance_confidence_weight_factor;
    float size_confidence_weight_factor;
    double tracking_max_object_percentage;
    double tracking_threshold;
    double tracking_min_overlap_percentage;


    explicit SubsenseConfig(const std::map<std::string, std::string> &props);
};


void SetPreprocessorTrack(const cv::Mat &fore, int frame_index,
                          int frame_cols, int frame_rows,
                          MPF::COMPONENT::MPFVideoTrack &track,
                          std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);


std::vector<cv::Rect> GetResizedRects(const log4cxx::LoggerPtr &logger,
                                      const SubsenseConfig &config,
                                      cv::Mat &fore,
                                      int frame_cols,
                                      int frame_rows,
                                      int downsample_count);


void ProcessMotionTracks(const SubsenseConfig &config,
                         const std::vector<cv::Rect> &resized_rects,
                         const cv::Mat &orig_frame,
                         int frame_index, int &tracker_id,
                         std::map<int, STRUCK> &tracker_map,
                         std::map<int, MPF::COMPONENT::MPFVideoTrack> &track_map,
                         std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);

cv::Rect Upscale(const cv::Rect &rect,
                 int frame_cols, int frame_rows,
                 int downsample_count);
    
void AssignDetectionConfidence(MPF::COMPONENT::MPFVideoTrack &track,
                               float distance_factor,
                               float size_factor);


#endif   // OPENMPF_CONTRIB_COMPONENTS_SUBSENSE_UTILS_H
