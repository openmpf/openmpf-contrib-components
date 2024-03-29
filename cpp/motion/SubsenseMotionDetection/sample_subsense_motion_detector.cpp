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
 * Copyright 2023 The MITRE Corporation                                       *
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

#include "MotionDetection_Subsense.h"

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include <detectionComponentUtils.h>

using namespace std;

using namespace MPF;
using namespace COMPONENT;

// Main program to run the SuBSENSE motion detection in standalone mode.

void processImage(MPFDetectionComponent *detection_engine, int argc, char* argv[]);
void processVideo(MPFDetectionComponent *detection_engine, int argc, char* argv[]);

int main(int argc, char* argv[]) {
    try {

        if (argc != 2 && argc != 4 && argc != 5) {
            cout << "Usage (IMAGE): " << argv[0] << " <uri>" << endl;
            cout << "Usage (VIDEO): " << argv[0] << " <uri> <start_index> <end_index> <detection_interval (optional)>"
                 << endl;
            return EXIT_FAILURE;
        }

        std::string app_dir = DetectionComponentUtils::GetAppDir(argv[0]);

        MotionDetection_Subsense subsense_motion_detection;
        MPFDetectionComponent *detection_engine = &subsense_motion_detection;
        detection_engine->SetRunDirectory(app_dir + "/plugin");

        if (!detection_engine->Init()) {
            cerr << "Failed to initialize." << endl;
            return EXIT_FAILURE;
        }

        if (argc == 2) {
            processImage(detection_engine, argc, argv);
        }
        else {
            processVideo(detection_engine, argc, argv);
        }

        if (!detection_engine->Close()) {
            cerr << "Failed to close." << endl;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}

void processImage(MPFDetectionComponent *detection_engine, int argc, char* argv[]) {

    MPFImageJob job("Testing", argv[1], { }, { });

    vector<MPFImageLocation> locations = detection_engine->GetDetections(job);
    cout << "Number of detections: " << locations.size() << endl;
}

void processVideo(MPFDetectionComponent *detection_engine, int argc, char* argv[]) {

    // get detection interval if argument is present
    long detection_interval = 1;
    if (argc > 4) {
        detection_interval = std::stol(argv[4]);
    }
    cout << "Using detection interval: " << detection_interval << endl;

    map<string, string> algorithm_properties;
    algorithm_properties.insert(pair<string, string>("FRAME_INTERVAL", to_string(detection_interval)));

    algorithm_properties["USE_MOTION_TRACKING"] = to_string(1);
    // algorithm_properties["VERBOSE"] = to_string(2);
    // algorithm_properties["ROTATION"] = to_string(270);
    // algorithm_properties["DISTANCE_CONFIDENCE_WEIGHT_FACTOR"] = to_string(0.75);
    // algorithm_properties["SIZE_CONFIDENCE_WEIGHT_FACTOR"] = to_string(0.25);

    MPFVideoJob job("Testing", argv[1], stoi(argv[2]), stoi(argv[3]), algorithm_properties, { });
    vector<MPFVideoTrack> tracks = detection_engine->GetDetections(job);

    cout << "Number of video tracks = " << tracks.size() << endl;
    for (int i = 0; i < tracks.size(); i++) {
        cout << "\nVideo track " << i << "\n"
             << "   start frame = " << tracks[i].start_frame << "\n"
             << "   stop frame = " << tracks[i].stop_frame << "\n"
             << "   number of locations = " << tracks[i].frame_locations.size() << "\n"
             << "   confidence = " << tracks[i].confidence << endl;

        for (auto it : tracks[i].frame_locations) {
            cout << "   Image location frame = " << it.first << "\n"
                 << "      x left upper = " << it.second.x_left_upper << "\n"
                 << "      y left upper = " << it.second.y_left_upper << "\n"
                 << "      width = " << it.second.width << "\n"
                 << "      height = " << it.second.height << "\n"
                 << "      confidence = " << it.second.confidence << endl;
        }
    }
}

