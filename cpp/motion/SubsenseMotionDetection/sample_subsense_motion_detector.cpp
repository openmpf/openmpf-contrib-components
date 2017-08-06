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
 * Copyright 2017 The MITRE Corporation                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <QCoreApplication>

#include "MotionDetection_Subsense.h"

using namespace MPF;
using namespace COMPONENT;

int main(int argc, char *argv[])
{
    QCoreApplication *this_app = new QCoreApplication(argc, argv);
    std::string app_dir = (this_app->applicationDirPath()).toStdString();
    delete this_app;
    /*
     * Parameters:
     *  char *data_uri: URI indicating the location of the input image or video.
     *  int start_frame, stop_frame: if the media type is video, then these are
     *                               required parameters, that give the start and stop
     *                               frames for detection.
     *  int frame_interval: if the media type is video, then this is an optional
     *                      parameter, that gives the frame interval for tracking. The default
     *                      is 1.
     *  int segment_video:  if the media type is video, then this optional parameter
     *                      allows the user to request that the video be split into two
     *                      contiguous segments to be processed sequentially.  A value of 1
     *                      requests segmentation.  Any other value will be ignored, and the
     *                      video will be processed from the start frame to the stop frame
     *                      in one segment.
     */

    MPFDetectionDataType media_type = VIDEO;
    int start_frame = 0, end_frame = 0;
    int frame_interval = 1;
    bool segment_video = false;

    start_frame = atoi(argv[2]);
    if (start_frame < 0) {
        std::cout << "start frame must be greater than or equal to zero: start frame = ["
                  << start_frame << "]" << std::endl;
        exit(-1);
    }

    end_frame = atoi(argv[3]);
    if (end_frame < start_frame) {
        std::cout << "stop_frame must be greater than or equal to start frame: start frame = ["
                  << start_frame << "] stop frame = [" << end_frame << "]" << std::endl;
        exit(-1);
    }

    //optional frame_interval
    if (argc > 4) {
        frame_interval = atoi(argv[4]);
        if(frame_interval < 1)
        {
            std::cout << "the frame_interval: " << frame_interval << " must be greater than 1" << std::endl;
            exit(-1);
        }
    }

    //optional segment_video
    if (argc > 5) {
        int flag = atoi(argv[5]);
        if (flag == 1) segment_video = true;
    }

    std::cout << "start_frame = " << start_frame << std::endl;
    std::cout << "stop_frame = " << end_frame << std::endl;
    std::cout << "frame_interval = " << frame_interval << std::endl;
    std::cout << "segment_video = " << segment_video << std::endl;

    /****************************/
    /* Configure and Initialize */
    /****************************/

    MotionDetection_Subsense detector;
    detector.SetRunDirectory(app_dir + "/plugin");
    if ( !detector.Init() ) {
        std::cout << "Subsense motion detection component initialization failed, exiting." << std::endl;
        exit (-1);
    }

    std::string data_file(argv[1]);
    std::cout << "file = " << data_file << std::endl;

    MPFDetectionError rc = MPF_DETECTION_SUCCESS;
    std::vector<MPFVideoTrack> tracks;
    int num_frames, segment_length;
    int first_seg_start, first_seg_end;
    int second_seg_start, second_seg_end;
    std::map<std::string, std::string> algorithm_properties;
    std::ostringstream stringStream;
    stringStream << frame_interval;

    algorithm_properties["FRAME_INTERVAL"] = stringStream.str();

    // algorithm_properties["USE_MOTION_TRACKING"] = std::to_string(1);
    // algorithm_properties["VERBOSE"] = std::to_string(2);
    // algorithm_properties["ROTATION"] = std::to_string(270);

    if (segment_video) {
        //split into two segments
        num_frames = end_frame - start_frame + 1;
        segment_length = (num_frames+1)/2;
        first_seg_start = start_frame;
        first_seg_end = start_frame + segment_length - 1;
        second_seg_start = first_seg_start + segment_length;
        second_seg_end = second_seg_start + segment_length - 1;
        if ((2*segment_length) > num_frames)
            second_seg_end -= 1;
        std::cout << "num_frames: " << num_frames << " segment_length: " << segment_length << std::endl;
        std::cout << "first_seg_start: " << first_seg_start << " first_seg_end: " << first_seg_end << std::endl;
        std::cout << "second_seg_start: " << second_seg_start << " second_seg_end: " << second_seg_end << std::endl;

        MPFVideoJob first_seg_job("Testing", data_file, first_seg_start, first_seg_end, algorithm_properties, { });
        rc = detector.GetDetections(first_seg_job, tracks);
        if (rc != MPF_DETECTION_SUCCESS) {
            std::cout << "Subsense motion detection component failed first segment: rc = " << rc << std::endl;
        }

        tracks.clear();
        MPFVideoJob second_seg_job("Testing", data_file, second_seg_start, second_seg_end, algorithm_properties, { });
        rc = detector.GetDetections(second_seg_job, tracks);
        if (rc != MPF_DETECTION_SUCCESS) {
            std::cout << "Subsense motion detection component failed second segment: rc = " << rc << std::endl;
        }
    }
    else {
        MPFVideoJob job("Testing", data_file, start_frame, end_frame, algorithm_properties, { });
        rc = detector.GetDetections(job, tracks);
        if (rc != MPF_DETECTION_SUCCESS) {
            std::cout << "Subsense motion detection component failed: rc = " << rc << std::endl;
        }
        else {
            std::cout << "number of tracks found = " << tracks.size() << std::endl;
            for (int i = 0; i < tracks.size(); i++) {
                std::cout << "track# " << i << std::endl;
                std::stringstream ss;
                ss << tracks[i].start_frame << ", " << tracks[i].stop_frame << ", " << tracks[i].confidence;
                std::cout << ss.str() << std::endl;
            }
        }
    }

    if ( !detector.Close() ) {
        std::cout << "Subsense motion detector failed in closing." << std::endl;
    }

    exit(rc);
}
