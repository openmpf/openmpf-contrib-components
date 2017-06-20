/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2016 The MITRE Corporation. All Rights Reserved.                 *
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

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <log4cxx/xml/domconfigurator.h>

#include <Utils.h>
#include <MPFImageReader.h>
#include <MPFSimpleConfigLoader.h>

#include "MotionDetection_Subsense.h"
#include "SubSense/BackgroundSubtractorSuBSENSE.h"
#include "struck.h"

using std::pair;

using namespace MPF;
using namespace COMPONENT;


void displayTracks(QString origPath, std::vector<MPFVideoTrack> tracks);

MotionDetection_Subsense::MotionDetection_Subsense() {
}

MotionDetection_Subsense::~MotionDetection_Subsense() {
}

std::string MotionDetection_Subsense::GetDetectionType() {
    return "MOTION";
}

bool MotionDetection_Subsense::Init() {
    std::string plugin_path = GetRunDirectory() + "/SubsenseMotionDetection";
    std::string logger_file = plugin_path + "/config/Log4cxxConfig.xml";
    log4cxx::xml::DOMConfigurator::configure(logger_file);
    motion_logger = log4cxx::Logger::getLogger("SubsenseMotionDetection");

    std::string config_file = plugin_path + "/config/mpfSubsenseMotionDetection.ini";

    if (LoadConfig(config_file, parameters) == -1) {
        LOG4CXX_ERROR(motion_logger, "failed to load config file " << config_file);
        return false;
    }

    if (parameters["VERBOSE"].toInt() > 0)
        motion_logger->setLevel(log4cxx::Level::getDebug());

    return true;
}

bool MotionDetection_Subsense::Close() {
    return true;
}


MPFDetectionError MotionDetection_Subsense::GetDetections(const MPFVideoJob &job, std::vector<MPFVideoTrack> &tracks) {
    try {
        LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Starting motion detection");

        LoadConfig(GetRunDirectory() + "/SubsenseMotionDetection/config/mpfSubsenseMotionDetection.ini", parameters);
        GetPropertySettings(job.job_properties);

        int detection_interval = parameters["FRAME_INTERVAL"].toInt();

        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(motion_logger, "[" << job.job_name << "] Input video file path is empty");
            return (MPF_INVALID_DATAFILE_URI);
        }

        int frame_skip = (detection_interval > 0) ? detection_interval : 1;

        MPFVideoCapture video_capture(job);
        if (!video_capture.IsOpened()) {
            return MPF_COULD_NOT_OPEN_DATAFILE;
        }

        MPFDetectionError detections_result = GetDetectionsFromVideoCapture(job, frame_skip, video_capture, tracks);
        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }

        return detections_result;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, motion_logger);
    }
}

MPFDetectionError MotionDetection_Subsense::GetDetectionsFromVideoCapture(const MPFVideoJob &job,
                                                                          const int frame_skip,
                                                                          MPFVideoCapture &video_capture,
                                                                          std::vector<MPFVideoTrack> &tracks) {

    int stop, downsample_count = 0;
    cv::Mat orig_frame, frame, fore;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::KeyPoint> points;
    QMap<int, STRUCK> tracker_map;
    QMap<int, MPFVideoTrack> track_map;
    MPFVideoTrack track;
    int tracker_id = 0;

    // Set track for default
    track.start_frame = -1;


    // Create background subtractor
    LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Creating background subtractor");
    BackgroundSubtractorSuBSENSE bg(parameters["F_REL_LBSP_THRESHOLD"].toFloat(), static_cast<size_t>(parameters["N_MIN_DESC_DIST_THRESHOLD"].toInt()),
                                    static_cast<size_t>(parameters["N_MIN_COLOR_DIST_THRESHOLD"].toInt()), static_cast<size_t>(parameters["N_BG_SAMPLES"].toInt()),
                                    static_cast<size_t>(parameters["N_REQUIRED_BG_SAMPLES"].toInt()), static_cast<size_t>(parameters["N_SAMPLES_FOR_MOVING_AVGS"].toInt()));


    int frame_count = video_capture.GetFrameCount();

    // If the segment we are working on does not start at frame 0 of
    // the video, then we use the previous frame to initialize the
    // foreground. Otherwise, we use frame 0 to initialize, and then
    // start processing at frame 1.

    /*** NOTE
         This means that we will never see a detection in frame
         0. Furthermore, the frame interval will be added to 1
         (instead of 0) at the start of the first segment of the
         video, but in subsequent segments it will be added to
         job.start_frame. For example, if the segment size is 20, and
         the frame interval is 5, you will get a sequence of
         detections that looks something like this:
         1,6,11,16, 20,25,30,35,...
    ***/

    int init_frame_index = 0;
    if (job.start_frame > 0) {
        init_frame_index = job.start_frame - 1;
    }
    video_capture.SetFramePosition(init_frame_index);
    video_capture >> frame;
    frame.copyTo(orig_frame);

    if (frame.empty()) {
        return MPF_BAD_FRAME_SIZE;
    }

    // Downsample frame
    while (frame.rows > parameters["MAXIMUM_FRAME_HEIGHT"].toInt() || frame.cols > parameters["MAXIMUM_FRAME_WIDTH"].toInt()) {
        cv::pyrDown(frame, frame);
        downsample_count++;
    }

    cv::Mat Roi;
    bg.initialize(frame, Roi);

    int frame_index = init_frame_index + 1;
    video_capture.SetFramePosition(frame_index);

    LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Starting video processing");

    while (frame_index < qMin(video_capture.GetFrameCount(), job.stop_frame + 1)) {
        std::vector<cv::Rect> rects;
        LOG4CXX_DEBUG(motion_logger, "frame index = " << frame_index);
        video_capture >> frame;
        if (frame.empty() || frame.rows == 0 || frame.cols == 0) {
            LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Empty frame encountered at frame " << frame_index);
            break;
        }

        // Downsample frame
        for (int x = 0; x < downsample_count; ++x) {
            cv::pyrDown(frame, frame);
        }

        bg.apply(frame, fore);

        if (parameters["USE_PREPROCESSOR"].toInt() == 1) {
            if (cv::countNonZero(fore) > 0) {
                if (track.start_frame == -1) {
                    track.start_frame = frame_index;
                    track.stop_frame = track.start_frame;
                    track.frame_locations.insert(
                            std::pair<int, MPFImageLocation>(frame_index,
                                                             MPFImageLocation(0, 0, orig_frame.cols, orig_frame.rows)));
                } else {
                    track.stop_frame = frame_index;
                    track.frame_locations.insert(
                            std::pair<int, MPFImageLocation>(frame_index,
                                                             MPFImageLocation(0, 0, orig_frame.cols, orig_frame.rows)));
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
            cv::findContours(fore, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


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
                        rect.height * pow(2, downsample_count) >= parameters["MIN_RECT_HEIGHT"].toInt()) {

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
        // Change the frame index based on the detection interval
        frame_index = frame_index + frame_skip;

        // Setting CV_CAP_POS_FRAME may result in not capturing the
        // last frame of the video. This has been observed on some,
        // but not all, VM architectures. The issue may be related to
        // http://code.opencv.org/issues/1419 and/or
        // http://code.opencv.org/issues/1335. Thus, only set the
        // frame position if necessary, otherwise use >> to capture
        // the next frame.
        if (frame_skip > 1) {
            video_capture.SetFramePosition(frame_index);
        }
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
        //now print tracks if available
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

    if (parameters["VERBOSE"].toInt() > 1) {
        displayTracks(QString::fromStdString(job.data_uri), tracks);
    }

    LOG4CXX_INFO(motion_logger, "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(tracks.size()) << " detections.");

    return MPF_DETECTION_SUCCESS;
}

MPFDetectionError MotionDetection_Subsense::GetDetections(const MPFImageJob &job, std::vector<MPFImageLocation> &locations) {
    try {
        LoadConfig(GetRunDirectory() + "/SubsenseMotionDetection/config/mpfSubsenseMotionDetection.ini", parameters);
        GetPropertySettings(job.job_properties);

        // if this component is used as a preprocessor then it will return that it detects motion in every image
        // (although no actual motion detection is performed)
        if (parameters["USE_PREPROCESSOR"].toInt() == 1) {
            MPFImageLocation detection;
            MPFImageReader image_reader(job);
            cv::Mat cv_image = image_reader.GetImage();

            // need to make sure it is a valid image
            if (cv_image.empty()) {
                LOG4CXX_ERROR(motion_logger, "[" << job.job_name << "] failed to read image file");
                return MPF_IMAGE_READ_ERROR;
            }

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
        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, motion_logger);
    }
}

void displayTracks(QString origPath, std::vector<MPFVideoTrack> tracks) {
    cv::VideoCapture capture(qPrintable(origPath));

    cv::Mat frame;
    std::vector<MPFVideoTrack>::iterator it = tracks.begin();

    while (capture.get(CV_CAP_PROP_POS_FRAMES) < capture.get(CV_CAP_PROP_FRAME_COUNT)) {
        capture >> frame;

        while (it != tracks.end() && it->start_frame == capture.get(CV_CAP_PROP_POS_FRAMES)-1) {
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

void MotionDetection_Subsense::GetPropertySettings(const std::map<std::string, std::string> &algorithm_properties) {
    std::string property;
    std::string str_value;

    for (std::map<std::string,std::string>::const_iterator imap = algorithm_properties.begin(); imap != algorithm_properties.end(); imap++) {
        property = imap->first;
        str_value = imap->second;
        parameters.insert(QString::fromStdString(property), QString::fromStdString(str_value));
    }
}

cv::Rect MotionDetection_Subsense::Upscale(const cv::Rect rect, const cv::Mat orig_frame, int downsample_count) {
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

MPF_COMPONENT_CREATOR(MotionDetection_Subsense);
MPF_COMPONENT_DELETER();
