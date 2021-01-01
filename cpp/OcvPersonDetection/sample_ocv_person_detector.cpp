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

//#include <log4cxx/simplelayout.h>
//#include <log4cxx/consoleappender.h>
//#include <log4cxx/fileappender.h>
//#include <log4cxx/logmanager.h>
#include <QCoreApplication>

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

#include "PersonDetection.h"

using namespace MPF;
using namespace COMPONENT;

int main(int argc, char* argv[]) {
    try {
        QCoreApplication* this_app = new QCoreApplication(argc, argv);
        std::string app_dir = (this_app->applicationDirPath()).toStdString();
        delete this_app;

        PersonDetection *personDetection = new PersonDetection();
        personDetection->SetRunDirectory(app_dir  + "/plugin");

        if (!personDetection->Init()) {
            printf("\nError: OCV Face Detection failed to initialize \n\n");
            return 1;
        }

        MPFDetectionComponent *detection = personDetection;
        if (detection == NULL) {
            printf("The detection component failed to initialize\n");
            return 1;
        }

        std::map<std::string, std::string> algorithm_properties;

        std::string executableName = argv[0];
        std::string uri = (argc >= 2) ? argv[1] : "null";
        int startFrame = (argc >= 4) ? atoi(argv[2]) : 0;
        int stopFrame = (argc >= 4) ? atoi(argv[3]) : 0;
        int detectionInterval = (argc >= 5) ? atoi(argv[4]) : 1;

        printf("\tExecutable Name:     %s\n", executableName.c_str());
        printf("\tURI:                 %s\n", uri.c_str());
        printf("\tStart Frame:         %d\n", startFrame);
        printf("\tStop Frame:          %d\n", stopFrame);
        printf("\tDetection Interval: %d\n", detectionInterval);

        printf("\nchecking for tracks... \n\n");

        if (argc == 4 || argc == 5) {

            std::ostringstream stringStream;
            stringStream << detectionInterval;

            algorithm_properties["FRAME_INTERVAL"] = stringStream.str();

            MPFVideoJob job("Testing", uri, startFrame, stopFrame, algorithm_properties, { });

            std::vector<MPFVideoTrack> tracks = detection->GetDetections(job);

            if (!tracks.empty()) {
                printf("\n\n----Tracks---- \n");
                for (unsigned int i = 0; i < tracks.size(); i++) {
                    printf("track index: %i \n", i);
                    printf("track start frame: %i \n", tracks[i].start_frame);
                    printf("track end frame: %i \n", tracks[i].stop_frame);
                    printf("locations size : %i \n", static_cast<int>(tracks[i].frame_locations.size()));;

                    int count = 0;
                    for (std::map<int, MPFImageLocation>::const_iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it) {
                        printf("\tdetection index: %i \n", count);
                        printf("\tframe num: %i\n", it->first);
                        printf("\tconfidence: %d\n", it->second.confidence);
                        printf("\tcorner: %i %i\n", it->second.x_left_upper, it->second.y_left_upper);
                        printf("\tsize: %i %i\n", it->second.width, it->second.height);
                        count++;
                    }
                }
            } else {
                printf("\n\n--No tracks found--\n");
            }
        } else if (argc == 2) {
            MPFImageJob job("Testing", uri, algorithm_properties, { });
            std::vector<MPFImageLocation> locations = detection->GetDetections(job);

            if (!locations.empty()) {
                printf("\n\n----Locations---- \n");
                for (unsigned int i = 0; i < locations.size(); i++) {
                    printf("location index: %i \n", i);
                    printf("\tcorner: %i %i\n", locations[i].x_left_upper,locations[i].y_left_upper);
                    printf("\tsize: %i %i\n", locations[i].width,locations[i].height);
                }
            } else {
                printf("\n\n--No locations found--\n");
            }
        } else {
            printf("\nError: The number of command line parameters was incorrect: %d \n\n", argc);
            printf("\t\tUsage (IMAGE): %s <uri> \n", executableName.c_str());
            printf("\t\tUsage (VIDEO): %s <uri> <start_frame> <end_frame> <detection_interval (optional)>\n", executableName.c_str());
        }

        printf("Deleting the detector in main.cpp\n");
        delete detection;

        return 0;
    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
