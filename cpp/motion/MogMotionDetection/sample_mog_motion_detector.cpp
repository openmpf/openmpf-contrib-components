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
 * Copyright 2016 The MITRE Corporation                                       *
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

#include <string>
#include <vector>
#include <iostream>
#include "MotionDetection_MOG2.h"

using namespace MPF;
using namespace COMPONENT;

using std::to_string;

int main(int argc, char* argv[]) {

    if ((4 != argc)) {
        std::cout << "Usage: " << argv[0] << " VIDEO_FILE_URI START_FRAME STOP_FRAME" << std::endl;
        return 0;
    }

    std::string uri(argv[1]);
    int start_frame = atoi(argv[2]);
    int stop_frame = atoi(argv[3]);

    if ((start_frame < 0) || (stop_frame <= start_frame)) {
        std::cerr << "Start frame must be greater than or equal 0, and stop_frame must be greater than start_frame" << std::endl;
        return MPF_INVALID_STOP_FRAME;
    }

    Properties algorithm_properties;

    algorithm_properties["USE_MOTION_TRACKING"] = to_string(1);

    Properties media_properties;
    std::string job_name("Test MOG Motion");
    MPFVideoJob job(job_name, uri, start_frame, stop_frame, algorithm_properties, media_properties);

    // Instantiate the component
    MotionDetection_MOG2 component;

    component.SetRunDirectory("plugin");

    if (!component.Init()) {
        std::cout << "Component initialization failed, exiting." << std::endl;
        return EXIT_FAILURE;
    }

    // Declare the vector of tracks to be filled in by the component.
    std::vector<MPFVideoTrack> tracks;

    // Pass the job to the video capture component
    MPFDetectionError rc = MPF_DETECTION_SUCCESS;

    rc = component.GetDetections(job, tracks);
    if (rc == MPF_DETECTION_SUCCESS) {
        std::cout << "Number of video tracks = " << tracks.size() << std::endl;

        for (int i = 0; i < tracks.size(); i++) {
            std::cout << "Video track number " << i << "\n"
                      << "   start frame = " << tracks[i].start_frame << "\n"
                      << "   stop frame = " << tracks[i].stop_frame << "\n"
                      << "   number of locations = " << tracks[i].frame_locations.size() << "\n"
                      << "   confidence = " << tracks[i].confidence << std::endl;

            for (auto it : tracks[i].frame_locations) {
                std::cout << "   Image location frame = " << it.first << "\n"
                          << "      x left upper = " << it.second.x_left_upper << "\n"
                          << "      y left upper = " << it.second.y_left_upper << "\n"
                          << "      width = " << it.second.width << "\n"
                          << "      height = " << it.second.height << "\n"
                          << "      confidence = " << it.second.confidence << std::endl;
            }
        }
    }
    else {
        std::cout << "GetDetections failed: " << rc << std::endl;
    }

    return 0;
}


