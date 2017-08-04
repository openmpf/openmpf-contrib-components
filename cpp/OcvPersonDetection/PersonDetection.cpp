/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2017 The MITRE Corporation. All Rights Reserved.                 *
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

#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>

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

using log4cxx::LogManager;

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

        log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
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
        output_video = parameters["OUTPUT_VIDEO"].toUInt();
        output_image = parameters["OUTPUT_IMAGE"].toUInt();
        output_base_path = parameters["OUTPUT_BASE_PATH"].toUtf8().constData();


        LOG4CXX_DEBUG(personLogger, "Imshow_on: " << imshow_on);
        LOG4CXX_DEBUG(personLogger, "Output_video: " << output_video);
        LOG4CXX_DEBUG(personLogger, "Output_image: " << output_image);
        LOG4CXX_DEBUG(personLogger, "Output_base_path: " << output_base_path);
    } else {
        LOG4CXX_DEBUG(personLogger, "Previously initialized");
    }

    return true;
}

bool PersonDetection::Close() {
    initialized = 0;
    return true;
}


MPFDetectionError PersonDetection::GetDetections(const MPFVideoJob &job, vector<MPFVideoTrack> &tracks) {
    try {
        int frame_interval = DetectionComponentUtils::GetProperty<int>(job.job_properties, "FRAME_INTERVAL", 1);

        LOG4CXX_TRACE(personLogger, "[" << job.job_name << "] Beginning the GetDetectionsFromVideo() function");
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Data_uri: " << job.data_uri);
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Start_index: " << job.start_frame);
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Stop_index: " << job.stop_frame);

        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(personLogger, "[" << job.job_name << "] Input video file path is empty");
            return MPF_INVALID_DATAFILE_URI;
        }
        int frame_skip = (frame_interval > 0) ? frame_interval : 1;

        MPFVideoCapture video_capture(job);
        if (!video_capture.IsOpened()) {
            LOG4CXX_ERROR(personLogger, "[" << job.job_name << "] Video failed to open.");
            return MPF_COULD_NOT_OPEN_DATAFILE;
        }

        MPFDetectionError detection_result = GetDetectionsFromVideoCapture(job, frame_skip, video_capture, tracks);

        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }

        //	Write the video.
        if (output_video) {
            writeDetectionToVideo(job.start_frame, job.stop_frame, job.data_uri, video_capture.GetFrameCount(), frame_skip, job.job_name, tracks);
        }

        return detection_result;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, personLogger);
    }

}


MPFDetectionError PersonDetection::GetDetectionsFromVideoCapture(const MPFVideoJob &job,
                                                                 const int frame_skip,
                                                                 MPFVideoCapture &video_capture,
                                                                 vector<MPFVideoTrack> &tracks) {

    //get frame count -  use total_frames to check the start_index and stop_index
    //to make sure they are within the video bounds
    int total_frames = video_capture.GetFrameCount();
    LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Total video frames: " << total_frames);

    video_capture.SetFramePosition(job.start_frame);

    // Do some initialization
    Mat frame, frame_draw, gray, prev_gray;
    int frame_index = job.start_frame;

    if (imshow_on) {
        cv::namedWindow("PersonTracker", 1);
    }

    //  Create the detector and track manager.
    HogDetector *detector = new HogDetector();
    if (detector == NULL) {
        LOG4CXX_ERROR(personLogger, "[" << job.job_name << "] The HogDetector failed to initialize");
        return MPF_OTHER_DETECTION_ERROR_TYPE;
    }
    TrakerManager *mTrack = new TrakerManager(detector, frame, 5);
    if (mTrack == NULL) {
        LOG4CXX_ERROR(personLogger, "[" << job.job_name << "] The Tracker failed to initialize");
        return MPF_OTHER_DETECTION_ERROR_TYPE;
    }

    LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Starting video processing");
    while (video_capture.GetCurrentFramePosition() < qMin(total_frames, job.stop_frame + 1)) {
        //  Get the frame.
        video_capture >> frame;
        if (frame.empty() || frame.rows == 0 || frame.cols == 0) {
            LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Empty frame encountered at frame " << video_capture.GetCurrentFramePosition());
            break;
        }

        //  Look for people.
        mTrack->doWork(frame, frame_index, tracks);

        //  Update the tracks.
        UpdateTracks(frame_index, tracks);

        //	Update the screen.
        if (imshow_on) {
            imshow("PersonTracker", frame);
            cvWaitKey(10);
        }

        //  Proceed to the correct next frame.
        video_capture.SetFramePosition(video_capture.GetCurrentFramePosition() + frame_skip - 1);
        frame_index += frame_skip;
    }

    //  There can still be running tracks when the video ends or the stop index has been hit
    CloseAnyOpenTracks(frame_index, tracks);

    //  Release resources.
    delete detector;
    delete mTrack;
    video_capture.Release();
    if (imshow_on) {
        cv::destroyWindow("PersonTracker");
    }

    //  Report.
    LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Total_detections_count: " << tracks.size());

    LOG4CXX_INFO(personLogger, "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(tracks.size()) << " detections.");

    return MPF_DETECTION_SUCCESS;
}



MPFDetectionError PersonDetection::GetDetections(const MPFImageJob &job, vector<MPFImageLocation> &locations) {
    try {
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Image file: " << job.data_uri);

        // Check that the input arguments are sensible
        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(personLogger, "[" << job.job_name << "] The data_uri was empty.");
            return MPF_INVALID_DATAFILE_URI;
        }

        //	Do some initialization.
        cv::Mat image_gray;
        if (imshow_on) {
            cv::namedWindow("PersonTracker", 1);
        }

        //	Load the image.
        MPFImageReader image_reader(job);
        cv::Mat image = image_reader.GetImage();

        if (image.empty()) {
            LOG4CXX_ERROR(personLogger, "[" << job.job_name << "] The image did not load properly.");
            return MPF_IMAGE_READ_ERROR;
        }

        cvtColor(image, image_gray, CV_BGR2GRAY);

        //	Get the detections.
        LOG4CXX_DEBUG(personLogger, "[" << job.job_name << "] Getting detections");
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        vector<cv::Rect> found;
        hog.detectMultiScale(image, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

        //	Record the detections.
        for (unsigned int j = 0; j < found.size(); j++) {
            cv::Rect person = found[j];
            int ix, iy, iw, ih;
            ix = (found[j].x >= 0) ? found[j].x : 0;
            iy = (found[j].y >= 0) ? found[j].y : 0;
            iw = (found[j].x + found[j].width < image.cols) ? found[j].width : image.cols - 1;
            ih = (found[j].y + found[j].height < image.rows) ? found[j].height : image.rows - 1;

            MPFImageLocation this_face(ix, iy, iw, ih, static_cast<float>(-1));
            locations.push_back(this_face);
        }

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        //	Display the image with detections.
        if (imshow_on || output_image) {
            cv::Mat raw_image = cv::imread(job.data_uri, CV_LOAD_IMAGE_IGNORE_ORIENTATION + CV_LOAD_IMAGE_COLOR);
            for (auto &location : locations) {
                rectangle(raw_image, Rect(location.x_left_upper, location.y_left_upper, location.width, location.height), Scalar(255, 255, 0));
            }
            if (imshow_on) {
                imshow("PersonTracker", raw_image);
                cvWaitKey(1000);
            }
            if (output_image) {
                imwrite(output_base_path + "/out.jpg", raw_image);
            }
        }

        //	Release resources.
        if (imshow_on) {
            cv::destroyWindow("PersonTracker");
        }

        // Report.
        LOG4CXX_INFO(personLogger,
                     "[" << job.job_name << "] Processing complete. Found " << static_cast<int>(locations.size())
                         << " detections.");

        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, personLogger);
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

vector <Rect> PersonDetection::GetRectsAtFrameIndex(int frame_index, const std::vector <MPFVideoTrack> &tracks,
                                                    std::vector <int> &track_indexes) {
    vector <Rect> object_rects;
    int track_index = 0;
    for (vector<MPFVideoTrack>::const_iterator it = tracks.begin(); it != tracks.end(); ++it) {
        if (frame_index >= it->start_frame && frame_index <= it->stop_frame) {
            Rect object_rect(ImageLocationToCvRect(it->frame_locations.at(frame_index)));
            object_rects.push_back(object_rect);
            track_indexes.push_back(track_index);
        }
        ++track_index;
    }
    return object_rects;
}

vector <Rect> PersonDetection::GetRectsAtFrameIndex(int frame_index, const std::vector <MPFVideoTrack> &tracks) {
    vector <Rect> object_rects;
    int track_index = 0;
    for (vector<MPFVideoTrack>::const_iterator object_track = tracks.begin(); object_track != tracks.end(); ++object_track) {
        if (frame_index >= object_track->start_frame && frame_index <= object_track->stop_frame) {
            Rect object_rect(ImageLocationToCvRect(object_track->frame_locations.at(frame_index)));
            object_rects.push_back(object_rect);
        }
        ++track_index;
    }
    return object_rects;
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
        LOG4CXX_DEBUG(personLogger, "[" << job_name << "] Person # " << cout);
        logPerson(it->second, job_name);
        count++;
    }
}

void PersonDetection::writeDetectionToVideo(const int start_index, const int stop_index, const std::string& data_uri, const int frame_count,
                                            const int frame_interval, const std::string& job_name, std::vector<MPFVideoTrack>& detections) {
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] Video file: " << data_uri);

    // Check that the input arguments are sensible
    if (data_uri.empty()) {
        LOG4CXX_ERROR(personLogger, "[" << job_name << "] The data_uri was empty.");
        return;
    }

    //  Open the input video.
    cv::VideoCapture cap;
    cap.open(data_uri);
    if (!cap.isOpened()) {
        LOG4CXX_ERROR(personLogger, "[" << job_name << "] Video failed to open.");
        return;
    }
    cap.set(CV_CAP_PROP_POS_FRAMES, start_index);

    //	Do some initialization.
    int frame_index = start_index;
    int stop = (stop_index == 0) ? INT_MAX : stop_index;
    VideoWriter outputVideo;
    Mat frame;

    string name = output_base_path + "/out.avi";
    int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
    double fps = cap.get(CV_CAP_PROP_FPS) / static_cast<double>(frame_interval);
    LOG4CXX_DEBUG(personLogger, "[" << job_name << "] FPS: " << fps);

    int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    Size size = Size(width, height);
    outputVideo.open(name, fourcc, fps, size);

    if (!outputVideo.isOpened()) {
        LOG4CXX_ERROR(personLogger, "[" << job_name << "] Output video failed to open.");
        return;
    }

    while (cap.get(CV_CAP_PROP_POS_FRAMES) < qMin(frame_count, stop)) {
        //  Get the frame.
        cap >> frame;
        if (frame.empty()) {
            LOG4CXX_ERROR(personLogger, "[" << job_name << "] The frame was not properly retrieved from the video.");
            return;
        }

        //	Draw the detections onto the frame.
        vector <Rect> results = GetRectsAtFrameIndex(frame_index, detections);
        for (int i = 0; i < results.size(); i++) {
            rectangle(frame, results[i], Scalar(255, 255, 0));
        }

        //	Write the frame to video.
        LOG4CXX_TRACE(personLogger, "[" << job_name << "] Writing frame to video.");
        outputVideo << frame;

        //  Proceed to the correct next frame.
        cap.set(CV_CAP_PROP_POS_FRAMES, cap.get(CV_CAP_PROP_POS_FRAMES) + (frame_interval - 1));
        frame_index += frame_interval;
    }
}

MPF_COMPONENT_CREATOR(PersonDetection);
MPF_COMPONENT_DELETER();
