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


#ifndef OPENMPF_COMPONENTS_OCVSsdFACEDETECTION_H
#define OPENMPF_COMPONENTS_OCVSsdFACEDETECTION_H

#include <log4cxx/logger.h>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <dlib/image_processing.h>
#include <dlib/opencv.h>

// MPF sdk header files 

#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

#include "OcvSsdFaceDetection_types.h"
#include "OcvSsdFaceDetection_streamio.h"
#include "OcvSsdFaceDetection_JobConfig.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  class Detection: public MPFImageLocation{
    cvPoint2fVec landmarks;
    cv::Mat      thumbnail;
    cv::Mat      feature;
  };


  /***************************************************************************/
  class OcvSsdFaceDetection : public MPFImageAndVideoDetectionComponentAdapter {

    public:
      bool Init() override;
      bool Close() override;
      string GetDetectionType(){return "FACE";};
      MPFDetectionError GetDetections(const MPFVideoJob &job, MPFVideoTrackVec    &tracks)    override;
      MPFDetectionError GetDetections(const MPFImageJob &job, MPFImageLocationVec &locations) override;

    private:

      log4cxx::LoggerPtr             _log;          ///< log object
      cv::dnn::Net                   _ssdNet;       ///< single shot DNN face detector network
      cv::Ptr<cv::face::FacemarkLBF> _facemark;     ///< landmark detector
      dlib::shape_predictor          _shape_predictor;  ///< landmark detector
      cv::dnn::Net                   _openFaceNet;  ///< feature generator

      void _detect(const JobConfig     &cfg,
                   MPFImageLocationVec &locations,
                   const cv::Mat       &bgrFrame);   ///< get bboxes with conf. scores

      cvPoint2fVecVec _getLandmarks(const JobConfig     &cfg,
                                    MPFImageLocationVec &locations,
                                    const cv::Mat       &bgrFrame); ///< get face landmarks

      void _drawLandmarks(cv::Mat &im,
                          cvPoint2fVec &landmarks); ///< draw landmarks on image

      cvMatVec _getThumbnails(const JobConfig       &cfg,
                              const cv::Mat         &bgrFrame,
                              const cvPoint2fVecVec &landmarks); ///< get thumbnails for detections              
        
      cvMatVec _getFeatures(const JobConfig &cfg,
                            const cvMatVec  &thumbnails); ///< get features from thumbnails

      float _featureDistance(const cv::Mat &a,
                             const cv::Mat &b){return norm(a,b,cv::NORM_L2);}  ///< compute feature distance between two features

/*
      MPFDetectionError GetDetectionsFromVideoCapture(const MPFVideoJob &job,
                                                      MPFVideoCapture   &video_capture,
                                                      MPFVideoTrackVec  &tracks);

      MPFDetectionError GetDetectionsFromImageData(const MPFImageJob   &job,
                                                   cv::Mat             &image_data,
                                                   MPFImageLocationVec &locations);

*/
  };
 }
}
#endif //OPENMPF_COMPONENTS_OCVFACEDETECTION_H
