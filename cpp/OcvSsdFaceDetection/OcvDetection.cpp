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

/* GroupRectanglesMod is a modified version of groupRectangles included in the
   OpenCV source and for that reason the copyright notice below is provided */

////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
////////////////////////////////////////////////////////////////////////////////////////


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "detectionComponentUtils.h"
#include "OcvDetection.h"

using namespace MPF::COMPONENT;


/******************************************************************************/
OcvDetection::OcvDetection(const string &pluginPath):
    _pluginPath(pluginPath),
    _log(log4cxx::Logger::getLogger("OcvSSDFaceDetection")){
        
    string  tfModelPath = pluginPath + "/data/opencv_face_detector_uint8.pb";
    string tfConfigPath = pluginPath + "/data/opencv_face_detector.pbtxt";
    string errMsg = "Failed to load TF model: " + tfConfigPath + ", " + tfModelPath;

    try{
        _ssdNet = cv::dnn::readNetFromTensorflow(tfModelPath, tfConfigPath);
    }catch(const runtime_error& re){
        LOG4CXX_FATAL(_log, errMsg << " Runtime error: " << re.what());
        THROW_EXCEPTION(errMsg);
    }catch(const exception& ex){
        LOG4CXX_FATAL(_log, errMsg << " Exception: " << ex.what());
        THROW_EXCEPTION(errMsg);
    }catch(...){
        LOG4CXX_FATAL(_log, errMsg << " Unknown failure occurred. Possible memory corruption");
        THROW_EXCEPTION(errMsg);
    }
}

/** ****************************************************************************
* Detect objects using SSD DNN opencv detector network.   
* 
* \param bgrFrame      BGR color image in which to detect objects
* \param minBBoxSide   minimum bounding box size to keep
* \param confThresh    minimum confidence score [0..1] to keep detection
*
* \returns vector of bounding boxes and conf scores for detected objects
*
***************************************************************************** */
MPFImageLocationVec OcvDetection::detect(const cv::Mat &bgrFrame,
                                         const int      minBBoxSide,
                                         const float    confThresh) {

    MPFImageLocationVec detections;

    const double inScaleFactor = 1.0;
    const cv::Size blobSize(300, 300);
    const cv::Scalar meanVal(104.0, 117.0, 124.0);  // BGR mean pixel color

    cv::Mat inputBlob = cv::dnn::blobFromImage(bgrFrame,      // BGR image
                                               inScaleFactor, // no pixel value scaline (e.g. 1.0/255.0)
                                               blobSize,      // expected network input size: 300x300
                                               meanVal,       // mean BGR pixel value
                                               true,          // swap RB channels
                                               false          // center crop
                                               );
    _ssdNet.setInput(inputBlob,"data");
    cv::Mat detection = _ssdNet.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++){
        float conf = detectionMat.at<float>(i, 2);
        if(conf > confThresh){
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * bgrFrame.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * bgrFrame.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * bgrFrame.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * bgrFrame.rows);
            int width  = x2 - x1;
            int height = y2 - y1;
            if(   width  >= minBBoxSide
               && height >= minBBoxSide){
              detections.push_back(MPFImageLocation(x1, y1, width, height, conf));
            }
        }
    }
    return detections;
}