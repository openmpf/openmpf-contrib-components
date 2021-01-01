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

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <log4cxx/xml/domconfigurator.h>

#include <Utils.h>
#include <MPFImageReader.h>
#include <MPFSimpleConfigLoader.h>
#include <MPFDetectionException.h>

#include "MotionDetection_Subsense.h"
#include "SubSense/BackgroundSubtractorSuBSENSE.h"
#include "struck.h"
#include "MotionDetectionUtils.h"

using std::pair;

using namespace MPF;
using namespace COMPONENT;

void displayTracks(QString origPath, int frameCount, std::vector<MPFVideoTrack> tracks);

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


std::vector<MPFVideoTrack> MotionDetection_Subsense::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Starting motion detection");

        LoadConfig(GetRunDirectory() + "/SubsenseMotionDetection/config/mpfSubsenseMotionDetection.ini", parameters);
        GetPropertySettings(job.job_properties, parameters);

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

std::vector<MPFVideoTrack> MotionDetection_Subsense::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job, MPFVideoCapture &video_capture) {

    int downsample_count = 0;
    cv::Mat orig_frame, frame, fore;
    QMap<int, STRUCK> tracker_map;
    QMap<int, MPFVideoTrack> track_map;
    MPFVideoTrack preprocessor_track;
    int tracker_id = 0;

    // Set track used for "preprocessing mode" to default
    preprocessor_track.start_frame = -1;


    // Create background subtractor
    LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Creating background subtractor");
    BackgroundSubtractorSuBSENSE bg(parameters["F_REL_LBSP_THRESHOLD"].toFloat(), static_cast<size_t>(parameters["N_MIN_DESC_DIST_THRESHOLD"].toInt()),
                                    static_cast<size_t>(parameters["N_MIN_COLOR_DIST_THRESHOLD"].toInt()), static_cast<size_t>(parameters["N_BG_SAMPLES"].toInt()),
                                    static_cast<size_t>(parameters["N_REQUIRED_BG_SAMPLES"].toInt()), static_cast<size_t>(parameters["N_SAMPLES_FOR_MOVING_AVGS"].toInt()));


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
    while (frame.rows > parameters["MAXIMUM_FRAME_HEIGHT"].toInt() || frame.cols > parameters["MAXIMUM_FRAME_WIDTH"].toInt()) {
        cv::pyrDown(frame, frame);
        downsample_count++;
    }

    cv::Mat Roi;
    bg.initialize(frame, Roi);


    LOG4CXX_TRACE(motion_logger, "[" << job.job_name << "] Starting video processing");

    std::vector<MPFVideoTrack> tracks;
    while (video_capture.Read(frame)) {

        if (frame.empty()) {
            LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] Empty frame encountered at frame " << frame_index);
            break;
        }
        LOG4CXX_DEBUG(motion_logger, "frame index = " << frame_index);

        // Downsample frame
        for (int x = 0; x < downsample_count; ++x) {
            cv::pyrDown(frame, frame);
        }

        bg.apply(frame, fore);

        if (parameters["USE_PREPROCESSOR"].toInt() == 1) {
            SetPreprocessorTrack(fore, frame_index,
                                 orig_frame.cols, orig_frame.rows,
                                 preprocessor_track, tracks);
        }
        else {
            std::vector<cv::Rect> resized_rects = GetResizedRects(job.job_name,
                                                                  motion_logger,
                                                                  parameters,
                                                                  fore,
                                                                  orig_frame.cols,
                                                                  orig_frame.rows,
                                                                  downsample_count);

            if (parameters["USE_MOTION_TRACKING"].toInt() == 1) {
                ProcessMotionTracks(parameters, resized_rects, orig_frame,
                                    frame_index, tracker_id,
                                    tracker_map, track_map, tracks);
            }
            else {
                foreach(const cv::Rect &rect, resized_rects) {
                    MPFVideoTrack track;
                    track.start_frame = frame_index;
                    track.stop_frame = track.start_frame;
                    track.frame_locations.insert(
                        std::pair<int, MPFImageLocation>(frame_index,
                                                         MPFImageLocation(rect.x, rect.y,
                                                                          rect.width,
                                                                          rect.height)));

                    tracks.push_back(track);
                }
            }
        }
        frame_index++;
    }

    // Finish the last track if we ended iteration through the video with an open track.
    if (parameters["USE_PREPROCESSOR"].toInt() == 1 && preprocessor_track.start_frame != -1) {
        tracks.push_back(preprocessor_track);
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

    // Assign a confidence value to each detection
    float distance_factor = parameters["DISTANCE_CONFIDENCE_WEIGHT_FACTOR"].toFloat();
    float size_factor = parameters["SIZE_CONFIDENCE_WEIGHT_FACTOR"].toFloat();
    for(MPFVideoTrack &track : tracks) {
        AssignDetectionConfidence(track, distance_factor, size_factor);
    }

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
        }
        else {
            LOG4CXX_DEBUG(motion_logger, "[" << job.job_name << "] No tracks found");
        }
    }

    LOG4CXX_INFO(motion_logger, "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(tracks.size()) << " tracks.");

    return tracks;
}

std::vector<MPFImageLocation> MotionDetection_Subsense::GetDetections(const MPFImageJob &job) {
    try {
        LoadConfig(GetRunDirectory() + "/SubsenseMotionDetection/config/mpfSubsenseMotionDetection.ini", parameters);
        GetPropertySettings(job.job_properties, parameters);

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


MPF_COMPONENT_CREATOR(MotionDetection_Subsense);
MPF_COMPONENT_DELETER();
