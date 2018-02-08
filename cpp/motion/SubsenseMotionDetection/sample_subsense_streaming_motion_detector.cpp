/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

#include "StreamingMotionDetection_Subsense.h"

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include <QCoreApplication>

using namespace std;

using namespace MPF;
using namespace COMPONENT;

// Main program to run the SuBSENSE streaming motion detection in
// standalone mode.

bool parseLong(const char *str, long &val);

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cout << "Usage (VIDEO): " << argv[0] << " <uri> <segment_length> <number of segments>" << endl;
        return EXIT_FAILURE;
    }

    long segment_length, num_segments;

    if (!parseLong(argv[2], segment_length)) {
        cout << "Failure: Could not parse segment length input argument" << endl;
        return EXIT_FAILURE;
    }
    cout << "Using segment length: " << segment_length << endl;

    if (!parseLong(argv[3], num_segments)) {
        cout << "Failure: Could not parse number of segments input argument" << endl;
        return EXIT_FAILURE;
    }
    cout << "Processing " << num_segments << " segments" << endl;

    QCoreApplication *this_app = new QCoreApplication(argc, argv);
    string app_dir = (this_app->applicationDirPath()).toStdString() + "/plugin";
    delete this_app;

    try {
        Properties job_props, media_props;
        MPFStreamingVideoJob job("TEST", app_dir, job_props, media_props);
        SubsenseStreamingDetection* detection_engine = new SubsenseStreamingDetection(job);

        // process stream

        int segment_frame_count = 0;
        int delay = 1;  //milliseconds
        int frame_index = 0;

        cv::VideoCapture cap(argv[1]); 
        cv::Mat frame;

        // get first frame
        while (cap.read(frame)) { 
            if (!frame.empty()) {
                break;
            } else {
                std::cout << "empty frame" << std::endl;
            }
        }

        for (int seg_index = 0; seg_index < num_segments; seg_index++) {
            int start_frame = frame_index;
            int stop_frame = frame_index + (segment_length-1);
            VideoSegmentInfo seg_info(seg_index,
                                      start_frame, stop_frame,
                                      frame.cols, frame.rows);

            detection_engine->BeginSegment(seg_info);

            do {
                if (frame.empty()) {
                    std::cout << "empty frame" << std::endl;
                } else {

                    bool activity_alert = detection_engine->ProcessFrame(frame, frame_index);

                    if (activity_alert) {
                        cout << "Activity alert for frame #" << frame_index << endl;
                    }
                    // cv::imshow("frame", frame);
                    // cv::waitKey(delay);

                    segment_frame_count++;
                    frame_index++;
                }
            } while (cap.read(frame) && (segment_frame_count < segment_length));

            vector<MPFVideoTrack> tracks = detection_engine->EndSegment();

            cout << "Number of video tracks = " << tracks.size() << endl;

            for (int i = 0; i < tracks.size(); i++) {
                cout << "\nVideo track " << i << "\n"
                     << "   start frame = " << start_frame + tracks[i].start_frame << "\n"
                     << "   stop frame = " << start_frame + tracks[i].stop_frame << "\n"
                     << "   number of locations = " << tracks[i].frame_locations.size() << "\n"
                     << "   confidence = " << tracks[i].confidence << endl;

                for (auto it : tracks[i].frame_locations) {
                    cout << "   Image location frame = " << start_frame + it.first << "\n"
                         << "      x left upper = " << it.second.x_left_upper << "\n"
                         << "      y left upper = " << it.second.y_left_upper << "\n"
                         << "      width = " << it.second.width << "\n"
                         << "      height = " << it.second.height << "\n"
                         << "      confidence = " << it.second.confidence << endl;
                }
            }
            segment_frame_count = 0;
        }
        std::cout << "end of stream" << std::endl;
        delete detection_engine;

    } catch (std::exception &e) {
        cout << "Exception caught in main: " << e.what() << endl;
    }

}

bool parseLong(const char *str, long &val) {

    errno = 0;
    char *temp;
    val = strtol(str, &temp, 0);

    if (temp == str || *temp != '\0' || ((val == LONG_MIN || val == LONG_MAX) && errno == ERANGE)) {
        cerr << "Could not convert '" << str << "' to long." << endl;
        return false;
    }

    return true;
}
