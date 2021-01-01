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

#include <QDebug>
#include <QtCore>
#include <QMap>

#include <opencv2/opencv.hpp>
#include <log4cxx/xml/domconfigurator.h>

#include <MPFImageReader.h>
#include <MPFSimpleConfigLoader.h>
#include <Utils.h>

#include "struck.h"

#include "MotionDetection_MOG2.h"

using namespace MPF;
using namespace COMPONENT;

using std::pair;


void displayTracks(QString origPath, int frameCount, std::vector<MPFVideoTrack> tracks);

MotionDetection_MOG2::MotionDetection_MOG2() {
}

MotionDetection_MOG2::~MotionDetection_MOG2() {
}

std::string MotionDetection_MOG2::GetDetectionType() {
    return "MOTION";
}

bool MotionDetection_MOG2::Init() {
    std::string plugin_path = GetRunDirectory() + "/MogMotionDetection";
    std::string logger_file = plugin_path + "/config/Log4cxxConfig.xml";
    log4cxx::xml::DOMConfigurator::configure(logger_file);
    motion_logger = log4cxx::Logger::getLogger("MogMotionDetection");

    std::string config_file = plugin_path + "/config/mpfMogMotionDetection.ini";

    if (LoadConfig(config_file, parameters) == -1) {
        LOG4CXX_ERROR(motion_logger, "failed to load config file " << config_file);
        return false;
    }

    if (parameters["VERBOSE"].toInt() > 0)
        motion_logger->setLevel(log4cxx::Level::getDebug());


    return true;
}

bool MotionDetection_MOG2::Close() {
    return true;
}


std::vector<MPFVideoTrack> MotionDetection_MOG2::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Starting motion detection");

        LoadConfig(GetRunDirectory() + "/MogMotionDetection/config/mpfMogMotionDetection.ini", parameters);
        GetPropertySettings(job.job_properties);


        MPFVideoCapture video_capture(job, true, true);

        std::vector<MPFVideoTrack> tracks = GetDetectionsFromVideoCapture(job, video_capture);
        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }

        if (parameters["VERBOSE"].toInt() > 1) {
            displayTracks(QString::fromStdString(job.data_uri), video_capture.GetFrameCount(), tracks);
        }

        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, motion_logger);
    }
}


std::vector<MPFVideoTrack> MotionDetection_MOG2::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job, MPFVideoCapture &video_capture) {

    MPFVideoTrack track;
    int stop, downsample_count = 0;
    cv::Mat orig_frame, frame, fore;
    std::vector<std::vector<cv::Point> > contours;
    QMap<int, STRUCK> tracker_map;
    QMap<int, MPFVideoTrack> track_map;
    int tracker_id = 0;

    // Set track for default
    track.start_frame = -1;


    // Create background subtractor
    LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Creating background subtractor");
    int history_length = parameters["HISTORY_LENGTH"].toInt();
    int var_threshold = parameters["VAR_THRESHOLD"].toInt();
    int detectShadows = parameters["BACKGROUND_SHADOW_DETECTION"].toInt();
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg;
    bg = cv::createBackgroundSubtractorMOG2(history_length,
                                            static_cast<double>(var_threshold),
                                            detectShadows);

    // Create video capture and set start frame

    int frame_count = video_capture.GetFrameCount();
    LOG4CXX_DEBUG(motion_logger, "frame count = " << frame_count);
    LOG4CXX_DEBUG(motion_logger, "begin frame = " << job.start_frame);
    LOG4CXX_DEBUG(motion_logger, "end frame = " << job.stop_frame);


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
    while (frame.rows > parameters["MAXIMUM_FRAME_HEIGHT"].toInt() || frame.cols > parameters["MAXIMUM_FRAME_WIDTH"].toInt()) {
        cv::pyrDown(frame, frame);
        downsample_count++;
    }

    // Initialize the foreground
    bg->apply(frame, fore);


    LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Starting video processing");

    std::vector<MPFVideoTrack> tracks;
    while (video_capture.Read(frame)) {
        std::vector<cv::Rect> rects;
        LOG4CXX_DEBUG(motion_logger, "frame index = " << frame_index);

        if (frame.empty()) {
            LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Empty frame encountered at frame " << frame_index);
            break;
        }

        // Downsample frame
        for (int x = 0; x < downsample_count; ++x) {
            cv::pyrDown(frame, frame);
        }

        bg->apply(frame, fore);

        if (parameters["USE_PREPROCESSOR"].toInt() == 1) {
            if (cv::countNonZero(fore) > 0) {
                if (track.start_frame == -1) {
                    track.start_frame = frame_index;
                    track.stop_frame = track.start_frame;
                    track.frame_locations.insert(
                            std::pair<int, MPFImageLocation>(frame_index,
                                                             MPFImageLocation(0, 0,
                                                                              static_cast<int>(orig_frame.cols),
                                                                              static_cast<int>(orig_frame.rows))));
                } else {
                    track.stop_frame = frame_index;
                    track.frame_locations.insert(
                            std::pair<int, MPFImageLocation>(frame_index,
                                                             MPFImageLocation(0, 0,
                                                                              orig_frame.cols,
                                                                              orig_frame.rows)));
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
            LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Eroding and blurring background");
            cv::erode(fore, fore, cv::Mat(), cv::Point(parameters["ERODE_ANCHOR_X"].toInt(), parameters["ERODE_ANCHOR_Y"].toInt()),
                      parameters["ERODE_ITERATIONS"].toInt());
            cv::dilate(fore, fore, cv::Mat(), cv::Point(parameters["DILATE_ANCHOR_X"].toInt(), parameters["DILATE_ANCHOR_Y"].toInt()),
                       parameters["DILATE_ITERATIONS"].toInt());
            cv::medianBlur(fore, fore, parameters["MEDIAN_BLUR_K_SIZE"].toInt());

            // Find the contours and then make bounding rects
            LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Finding contours and combine overlaps");
            cv::findContours(fore, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


                    foreach (std::vector<cv::Point> contour, contours) {
                    // Need to add the rect twice so that it doesnt get removed
                    rects.push_back(cv::boundingRect(contour));
                    rects.push_back(cv::boundingRect(contour));
                }

            // Combines overlapping rects together
            LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Converting contours to rects");

            cv::groupRectangles(rects, parameters["GROUP_RECTANGLES_GROUP_THRESHOLD"].toInt(), parameters["GROUP_RECTANGLES_EPS"].toDouble());

            LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Writing rects to VideoTrack");
            std::vector<cv::Rect> resized_rects;
                    foreach (cv::Rect rect, rects) {
                    if ((rect.width * pow(2, downsample_count)) >= parameters["MIN_RECT_WIDTH"].toInt() &&
                        (rect.height * pow(2, downsample_count)) >= parameters["MIN_RECT_HEIGHT"].toInt()) {

                        if (parameters["USE_MOTION_TRACKING"].toInt() == 1) {
                            resized_rects.push_back(Upscale(rect, orig_frame, downsample_count));
                        } else {
                            MPFVideoTrack track;
                            track.start_frame = frame_index;
                            track.stop_frame = track.start_frame;
                            cv::Rect resized = Upscale(rect, orig_frame, downsample_count);
                            track.frame_locations.insert(
                                    std::pair<int, MPFImageLocation>(frame_index,
                                                                     MPFImageLocation(resized.x, resized.y, resized.width, resized.height)));

                            tracks.push_back(track);
                        }
                    }
                }

            if (parameters["USE_MOTION_TRACKING"].toInt() == 1) {

                QMultiMap<cv::Rect, int> tracked_rects;

                // Append to tracks
                for(QMap<int, STRUCK>::iterator it = tracker_map.begin(); it != tracker_map.end(); it++) {

                    cv::Rect track = it.value().nextFrame(orig_frame, resized_rects);

                    // A rect with starting point 0,0 and w/h of 1,1 will mark the end of the track
                    if (track == cv::Rect(0, 0, 1, 1)) {
                        tracks.push_back(track_map.value(it.key()));
                        track_map.erase(track_map.find(it.key()));
                        it = tracker_map.erase(it);
                        it--;
                    } else {
                        track_map.find(it.key()).value().stop_frame = frame_index;
                        LOG4CXX_DEBUG(motion_logger, __LINE__ << " stop_frame = " << track_map.find(it.key()).value().stop_frame);
                        track_map.find(it.key()).value().frame_locations.insert(
                                std::pair<int, MPFImageLocation>(frame_index, MPFImageLocation(track.x, track.y,
                                                                                               track.width, track.height)));
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
                foreach (cv::Rect rect, tracked_rects.keys()) {
                    for (int x = 1; x < tracked_rects.values(rect).size(); x++) {
                        int id = tracked_rects.values(rect)[x];
                        tracks.push_back(track_map.value(id));

                        tracker_map.erase(tracker_map.find(id));
                        track_map.erase(track_map.find(id));
                        tracked_rects.erase(tracked_rects.find(rect, id));
                    }
                }
            }
        }
        frame_index++;
    }


    // Finish the last track if we ended iteration through the video with an open track.
    if (parameters["USE_PREPROCESSOR"].toInt() == 1 && track.start_frame != -1) {
        tracks.push_back(track);
    }

    // Complete open tracks
    for(QMap<int, STRUCK>::iterator it= tracker_map.begin(); it != tracker_map.end(); it++) {
        tracks.push_back(track_map.value(it.key()));
        track_map.erase(track_map.find(it.key()));
        it = tracker_map.erase(it);
        it--;
    }


    // Close video capture
    video_capture.Release();

    if (parameters["VERBOSE"].toInt() > 0) {
        // Now print tracks if available
        if (!tracks.empty()) {
            for (unsigned int i=0; i<tracks.size(); i++) {
                LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Track index: " << i);
                LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Track start frame: " << tracks[i].start_frame);
                LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Track end frame: " << tracks[i].stop_frame);

                for (std::map<int, MPFImageLocation>::const_iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it) {
                    LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Frame num: " << it->first);
                    LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Bounding rect: (" << it->second.x_left_upper << ", " <<
                                                     it->second.y_left_upper << ", " << it->second.width << ", " << it->second.height <<
                                                     ")");
                    LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Confidence: " << it->second.confidence);
                }
            }
        } else {
            LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] No tracks found");
        }
    }

    LOG4CXX_INFO(motion_logger, "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(tracks.size()) << " tracks.");

    return tracks;
}


std::vector<MPFImageLocation> MotionDetection_MOG2::GetDetections(const MPFImageJob &job) {
    try {
        LoadConfig(GetRunDirectory() + "/MogMotionDetection/config/mpfMogMotionDetection.ini", parameters);
        GetPropertySettings(job.job_properties);

        std::vector<MPFImageLocation> locations;
        // if this component is used as a preprocessor then it will return that it detects motion in every image
        // (although no actual motion detection is performed)
        if (parameters["USE_PREPROCESSOR"].toInt() == 1) {
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


        LOG4CXX_INFO(motion_logger,
                     "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(locations.size())
                         << " detections.");
        return locations;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, motion_logger);
    }
}

// NOTE: This only draws a bounding box around the first detection in each track
void displayTracks(QString origPath, int frameCount, std::vector<MPFVideoTrack> tracks) {
    cv::VideoCapture capture(qPrintable(origPath));

    cv::Mat frame;
    std::vector<MPFVideoTrack>::iterator it = tracks.begin();

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

void MotionDetection_MOG2::GetPropertySettings(const std::map<std::string, std::string> &algorithm_properties) {
    std::string property;
    std::string str_value;

    for (std::map<std::string,std::string>::const_iterator imap = algorithm_properties.begin(); imap != algorithm_properties.end(); imap++) {
        property = imap->first;
        str_value = imap->second;
        parameters.insert(QString::fromStdString(property), QString::fromStdString(str_value));
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

bool operator <(const cv::Rect &r1, const cv::Rect &r2) {
    return r1.x != r2.x || r1.y != r2.y || r1.area() < r2.area();
}

MPF_COMPONENT_CREATOR(MotionDetection_MOG2);
MPF_COMPONENT_DELETER();
