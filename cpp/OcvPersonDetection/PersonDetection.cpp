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
 * person detection component software.                                       *
 *                                                                            *
 * The MPF person detection component software is free software: you can      *
 * redistribute it and/or modify it under the terms of the GNU General        *
 * Public License as published by the Free Software Foundation, either        *
 * version 3 of the License, or (at your option) any later version.           *
 *                                                                            *
 * The MPF person detection component software is distributed in the hope     *
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the         *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.   *
 * See the (GNU General Public License for more details.                      *
 *                                                                            *
 * You should have received a copy of the GNU General Public License          *
 * along with the MPF person detection component software. If not, see        *
 * <http://www.gnu.org/licenses/>.                                            *
 ******************************************************************************/

#include <iostream>

#include <MPFImageReader.h>
#include <MPFSimpleConfigLoader.h>
#include <detectionComponentUtils.h>
#include <Utils.h>

#include "tracking/detector.h"
#include "tracking/multiTrackAssociation.h"
#include "tracking/tracker.h"

#include "PersonDetection.h"

using std::string;
using std::vector;

using cv::Mat;
using cv::Rect;
using cv::Scalar;
using cv::Size;
using cv::VideoWriter;

using log4cxx::Logger;
using log4cxx::LoggerPtr;

using namespace std;
using namespace MPF;
using namespace COMPONENT;

std::string PersonDetection::GetDetectionType() {
    return "PERSON";
}

bool PersonDetection::Init() {
    if (!initialized) {
        int rc;
        string run_dir = GetRunDirectory();
        string plugin_path = run_dir + "/OcvPersonDetection";
        string config_path = plugin_path + "/config";

        personLogger = log4cxx::Logger::getLogger("OcvPersonDetection");
        string config_file = config_path + "/mpfOcvPersonDetection.ini";
        LOG4CXX_DEBUG(personLogger, "Looking for config file in" << config_path.c_str());
        rc = LoadConfig(config_file, parameters);
        if (rc) {
            LOG4CXX_ERROR(personLogger, "Failed to load config file: " << config_file)
            return false;
        }

        initialized = true;

        imshow_on = parameters["IMSHOW_ON"].toUInt();


        LOG4CXX_DEBUG(personLogger, "Imshow_on: " << imshow_on);
    } else {
        LOG4CXX_DEBUG(personLogger, "Previously initialized");
    }

    return true;
}

bool PersonDetection::Close() {
    initialized = 0;
    return true;
}


vector<MPFVideoTrack> PersonDetection::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_TRACE(personLogger, "[" << job.job_name << "] Beginning the GetDetectionsFromVideo() function");
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Data_uri: " << job.data_uri);
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Start_index: " << job.start_frame);
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Stop_index: " << job.stop_frame);

        MPFVideoCapture video_capture(job, true, true);

        vector<MPFVideoTrack> tracks = GetDetectionsFromVideoCapture(job, video_capture);

        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }

        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, personLogger);
    }

}


vector<MPFVideoTrack> PersonDetection::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job, MPFVideoCapture &video_capture) {

    int total_frames = video_capture.GetFrameCount();
    LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Video frames: " << total_frames);

    Mat frame;
    int frame_index = 0;

    if (imshow_on) {
        cv::namedWindow("PersonTracker", 1);
    }

    //  Create the detector and track manager.
    HogDetector detector;
    TrakerManager mTrack(&detector, frame, 5);

    vector<MPFVideoTrack> tracks;
    LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Starting video processing");
    while (video_capture.Read(frame)) {
        if (frame.empty() || frame.rows == 0 || frame.cols == 0) {
            LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Empty frame encountered at frame " << video_capture.GetCurrentFramePosition());
            break;
        }

        //  Look for people.
        mTrack.doWork(frame, frame_index, tracks);

        //  Update the tracks.
        UpdateTracks(frame_index, tracks);

        //	Update the screen.
        if (imshow_on) {
            imshow("PersonTracker", frame);
            cv::waitKey(10);
        }

        frame_index++;
    }

    //  There can still be running tracks when the video ends or the stop index has been hit
    CloseAnyOpenTracks(frame_index, tracks);

    //  Release resources.
    video_capture.Release();
    if (imshow_on) {
        cv::destroyWindow("PersonTracker");
    }

    //  Report.
    LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Total_detections_count: " << tracks.size());

    LOG4CXX_INFO(personLogger, "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(tracks.size()) << " tracks.");

    return tracks;
}



vector<MPFImageLocation> PersonDetection::GetDetections(const MPFImageJob &job) {
    try {
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Image file: " << job.data_uri);

        if (imshow_on) {
            cv::namedWindow("PersonTracker", 1);
        }

        //	Load the image.
        MPFImageReader image_reader(job);
        cv::Mat image = image_reader.GetImage();

        //	Get the detections.
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Getting detections");
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        vector<cv::Rect> found;
        hog.detectMultiScale(image, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

        vector<MPFImageLocation> locations;
        cv::Rect imageRect(cv::Point(0, 0), image.size());
        for (const cv::Rect &detection : found) {
            cv::Rect intersection = detection & imageRect;
            locations.emplace_back(intersection.x, intersection.y, intersection.width, intersection.height);
        }

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        //	Display the image with detections.
        if (imshow_on) {
            cv::Mat raw_image = cv::imread(job.data_uri, cv::IMREAD_IGNORE_ORIENTATION + cv::IMREAD_COLOR);
            for (auto &location : locations) {
                rectangle(raw_image, Rect(location.x_left_upper, location.y_left_upper, location.width, location.height), Scalar(255, 255, 0));
            }
            imshow("PersonTracker", raw_image);
            cv::waitKey(1000);
            cv::destroyWindow("PersonTracker");
        }

        // Report.
        LOG4CXX_INFO(personLogger,
                     "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(locations.size())
                         << " detections.");

        return locations;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, personLogger);
    }
}

void PersonDetection::UpdateTracks(int frame_index, vector <MPFVideoTrack> &tracks) {
    for (vector<MPFVideoTrack>::iterator it = tracks.begin(); it != tracks.end(); it++) {
        if ((*it).stop_frame == -1 && frame_index - (*it).start_frame != static_cast<int>((*it).frame_locations.size()) - 1) {
            Rect t_rect = ImageLocationToCvRect((*it).frame_locations[frame_index-1]); // copy previous detection
            MPFImageLocation detection(t_rect.x, t_rect.y, t_rect.width, t_rect.height, static_cast<float>(-1));
            (*it).frame_locations.insert(pair<int, MPFImageLocation>(frame_index, detection));
        }
    }
}

void PersonDetection::CloseAnyOpenTracks(int frame_index, vector <MPFVideoTrack> &tracks) {
    for (vector<MPFVideoTrack>::iterator it = tracks.begin(); it != tracks.end(); it++) {
        if ((*it).stop_frame == -1) {
            (*it).stop_frame = frame_index-1;
        }
    }
}

MPFImageLocation PersonDetection::CvRectToImageLocation(const Rect &rect) {
    return MPFImageLocation(rect.x, rect.y, rect.width, rect.height);
}

Rect PersonDetection::ImageLocationToCvRect(const MPFImageLocation &detection) {
    return Rect(detection.x_left_upper, detection.y_left_upper, detection.width,
                detection.height);
}



void PersonDetection::logPerson(const MPFImageLocation& detection, const std::string& job_name) {
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] XLeftUpper: " << detection.x_left_upper);
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] YLeftUpper: " << detection.y_left_upper);
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] Width:      " << detection.width);
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] Height:     " << detection.height);
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] Confidence: " << detection.confidence);
}

void PersonDetection::logTrack(const MPFVideoTrack& track, const std::string& job_name) {
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] StartFrame: " << track.start_frame);
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] StopFrame:  " << track.stop_frame);
    int count;
    for (std::map<int, MPFImageLocation>::const_iterator it = track.frame_locations.begin(); it != track.frame_locations.end(); ++it) {
        LOG4CXX_DEBUG(personLogger, "[" << job_name << "] Person # " << count);
        logPerson(it->second, job_name);
        count++;
    }
}


MPF_COMPONENT_CREATOR(PersonDetection);
MPF_COMPONENT_DELETER();
