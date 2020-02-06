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

#include <vector>

#include "MotionDetectionUtils.h"

using namespace MPF;
using namespace COMPONENT;

void GetPropertySettings(const std::map<std::string, std::string> &algorithm_properties,
                         QHash<QString, QString> &parameters) {
    std::string property;
    std::string str_value;

    for (std::map<std::string,std::string>::const_iterator imap = algorithm_properties.begin(); imap != algorithm_properties.end(); imap++) {
        property = imap->first;
        str_value = imap->second;
        parameters.insert(QString::fromStdString(property), QString::fromStdString(str_value));
    }
}


void SetPreprocessorTrack(const cv::Mat fore, int frame_index,
                          int frame_cols, int frame_rows,
                          MPFVideoTrack &track,
                          std::vector<MPFVideoTrack> &tracks) {

    if (cv::countNonZero(fore) > 0) {
        if (track.start_frame == -1) {
            track.start_frame = frame_index;
            track.stop_frame = track.start_frame;
            track.frame_locations.insert(
                std::pair<int, MPFImageLocation>(frame_index,
                                                 MPFImageLocation(0, 0,
                                                                  frame_cols,
                                                                  frame_rows)));
        } else {
            track.stop_frame = frame_index;
            track.frame_locations.insert(
                std::pair<int, MPFImageLocation>(frame_index,
                                                 MPFImageLocation(0, 0,
                                                                  frame_cols,
                                                                  frame_rows)));
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
                                      const QHash<QString, QString> &parameters,
                                      cv::Mat &fore,
                                      int frame_cols,
                                      int frame_rows,
                                      int downsample_count) {

    std::vector<cv::Rect> rects;
    // Make motion larger objects
    LOG4CXX_TRACE(logger, "[" << job_name << "] Eroding and blurring background");
    cv::erode(fore, fore, cv::Mat(), cv::Point(parameters["ERODE_ANCHOR_X"].toInt(), parameters["ERODE_ANCHOR_Y"].toInt()),
              parameters["ERODE_ITERATIONS"].toInt());
    cv::dilate(fore, fore, cv::Mat(), cv::Point(parameters["DILATE_ANCHOR_X"].toInt(), parameters["DILATE_ANCHOR_Y"].toInt()),
               parameters["DILATE_ITERATIONS"].toInt());
    cv::medianBlur(fore, fore, parameters["MEDIAN_BLUR_K_SIZE"].toInt());


    std::vector<std::vector<cv::Point> > contours;
    // Find the contours and then make bounding rects
    LOG4CXX_TRACE(logger, "[" << job_name << "] Finding contours and combine overlaps");
    cv::findContours(fore, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


    foreach (const std::vector<cv::Point> &contour, contours) {
        // Need to add the rect twice so that it doesnt get removed
        rects.push_back(cv::boundingRect(contour));
        rects.push_back(cv::boundingRect(contour));
    }

    // Combines overlapping rects together
    LOG4CXX_TRACE(logger, "[" << job_name << "] Converting contours to rects");

    cv::groupRectangles(rects, parameters["GROUP_RECTANGLES_GROUP_THRESHOLD"].toInt(), parameters["GROUP_RECTANGLES_EPS"].toDouble());

    LOG4CXX_TRACE(logger, "[" << job_name << "] Resizing rects");
    std::vector<cv::Rect> resized_rects;
    foreach (const cv::Rect &rect, rects) {
        if ((rect.width * pow(2, downsample_count)) >= parameters["MIN_RECT_WIDTH"].toInt() &&
            rect.height * pow(2, downsample_count) >= parameters["MIN_RECT_HEIGHT"].toInt()) {
            resized_rects.push_back(Upscale(rect, frame_cols, frame_rows, downsample_count));
        }
    }
    return resized_rects;
}


void ProcessMotionTracks(const QHash<QString, QString> &parameters,
                         const std::vector<cv::Rect> &resized_rects,
                         const cv::Mat &orig_frame,
                         int frame_index, int &tracker_id,
                         QMap<int, STRUCK> &tracker_map,
                         QMap<int, MPFVideoTrack> &track_map,
                         std::vector<MPFVideoTrack> &tracks) {

    QMultiMap<cv::Rect, int> tracked_rects;

    // Append to tracks
    for(QMap<int, STRUCK>::iterator it= tracker_map.begin(); it != tracker_map.end(); it++) {

        cv::Rect track = it.value().nextFrame(orig_frame, resized_rects);

        // A rect with starting point 0,0 and w/h of 1,1 will mark the end of the track
        if (track == cv::Rect(0, 0, 1, 1)) {
            tracks.push_back(track_map.value(it.key()));
            track_map.erase(track_map.find(it.key()));
            it = tracker_map.erase(it);
            it--;
        } else {
            track_map.find(it.key()).value().stop_frame = frame_index;
            track_map.find(it.key()).value().frame_locations.insert(
                std::pair<int, MPFImageLocation>(frame_index,
                                                 MPFImageLocation(track.x, track.y, track.width, track.height)));
            tracked_rects.insert(track, it.key());
        }
    }

    // Create new tracks
    if (tracked_rects.size() != resized_rects.size()) {
        foreach (cv::Rect rect, resized_rects) {
            if (!tracked_rects.keys().contains(rect)) {
                if ((orig_frame.rows * orig_frame.cols * parameters["TRACKING_MAX_OBJECT_PERCENTAGE"].toDouble()) < rect.area()) {
                    continue;
                }
                tracker_map.insert(++tracker_id, STRUCK());
                tracker_map.find(tracker_id).value().initialize(orig_frame, rect, parameters["TRACKING_THRESHOLD"].toDouble(), parameters["TRACKING_MIN_OVERLAP_PERCENTAGE"].toDouble());
                MPFVideoTrack temp(frame_index, frame_index);
                temp.frame_locations.insert(
                    std::pair<int, MPFImageLocation>(frame_index,
                                                     MPFImageLocation(rect.x, rect.y, rect.width, rect.height)));

                track_map.insert(tracker_id, temp);
                tracked_rects.insert(rect, tracker_id);
            }
        }
    }

    // Remove merged tracks
    foreach (const cv::Rect &rect, tracked_rects.keys()) {
        for (int x = 1; x < tracked_rects.values(rect).size(); x++) {
            int id = tracked_rects.values(rect)[x];
            tracks.push_back(track_map.value(id));

            tracker_map.erase(tracker_map.find(id));
            track_map.erase(track_map.find(id));
            tracked_rects.erase(tracked_rects.find(rect, id));
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


bool operator <(const cv::Rect &r1, const cv::Rect &r2) {
    return r1.x != r2.x || r1.y != r2.y || r1.area() < r2.area();
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
        float max_size = 0;
        float max_confidence = 0.0;
        for (auto entry: track.frame_locations) {
            float entry_size = static_cast<float>(entry.second.width * entry.second.height);
            if (entry_size > max_size) {
                max_size = entry_size;
            }
        }

        if (number_of_detections <= 2) {
            for (auto &entry : track.frame_locations) {
                MPFImageLocation &loc = entry.second;
                loc.confidence = distance_factor + size_factor*(static_cast<float>(loc.height*loc.width)/max_size);
                max_confidence = (loc.confidence > max_confidence) ? loc.confidence : max_confidence;
            }
        }
        else {
            float center = static_cast<float>(number_of_detections)/2.0;
            int center_index = static_cast<int>(center);
            float confidence_incr = (center_index < center) ? 1.0/center_index : 1.0/(center_index - 1);

            auto first = track.frame_locations.begin();
            auto last = track.frame_locations.rbegin();
            for (int i = 0; i < center_index; ++i) {
                MPFImageLocation &loc1 = (first++)->second;
                MPFImageLocation &loc2 = (last++)->second;
                loc1.confidence = distance_factor*(i*confidence_incr) +
                                  size_factor*(static_cast<float>(loc1.height*loc1.width)/max_size);
                loc2.confidence = distance_factor*(i*confidence_incr) +
                                  size_factor*(static_cast<float>(loc2.height*loc2.width)/max_size);
                max_confidence = (loc1.confidence > max_confidence) ? loc1.confidence : max_confidence;
                max_confidence = (loc2.confidence > max_confidence) ? loc2.confidence : max_confidence;
            }
            MPFImageLocation &loc1 = (first)->second;
            loc1.confidence = distance_factor +
               size_factor*(static_cast<float>(loc1.height*loc1.width)/max_size);
            max_confidence = (loc1.confidence > max_confidence) ? loc1.confidence : max_confidence;
            if (center_index < center) {
                MPFImageLocation &loc2 = (last)->second;
                loc2.confidence = distance_factor +
                                  size_factor*(static_cast<float>(loc2.height*loc2.width)/max_size);
                max_confidence = (loc2.confidence > max_confidence) ? loc2.confidence : max_confidence;
            }
        }

        // Finally, normalize all confidences to between 0 and 1.
        for (auto &entry : track.frame_locations) {
            MPFImageLocation &loc = entry.second;
            loc.confidence = loc.confidence/max_confidence;
        }
    }
}

