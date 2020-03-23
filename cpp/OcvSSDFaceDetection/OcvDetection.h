/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2019 The MITRE Corporation                                       *
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


#ifndef OPENMPF_COMPONENTS_OCVDETECTION_H
#define OPENMPF_COMPONENTS_OCVDETECTION_H

#include <string>
#include <vector>
#include <utility>

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

class OcvDetection {
public:
    OcvDetection();
    ~OcvDetection(){};

    std::vector<std::pair<cv::Rect, float>> DetectFacesHaar(const cv::Mat &frame_gray, int min_face_size = 48);
    std::vector<std::pair<cv::Rect, float>> DetectFacesSSD(const cv::Mat &frame_color, int min_face_size = 48);

    bool Init(std::string &run_directory);

private:
    std::string face_cascade_path;
    cv::CascadeClassifier face_cascade;

    std::string face_ssd_config_path;
    std::string face_ssd_model_path;
    cv::dnn::Net face_ssd_net;

    bool initialized;

    log4cxx::LoggerPtr openFaceDetectionLogger;

    void GroupRectanglesMod(std::vector<cv::Rect>& rectList, int groupThreshold, double eps, std::vector<int>* weights, std::vector<double>* levelWeights);
};


#endif //OPENMPF_COMPONENTS_OCVDETECTION_H
