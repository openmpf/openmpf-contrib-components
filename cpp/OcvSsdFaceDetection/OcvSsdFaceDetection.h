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

#include <map>
#include <string>
#include <vector>

#include <QHash>
#include <QString>
#include <log4cxx/logger.h>

#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>

#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <MPFDetectionComponent.h>
#include <MPFVideoCapture.h>



#include "OcvDetection.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  /* **************************************************************************
  * Conveniance operator to dump MPFLocation to a stream
  *************************************************************************** */ 
  std::ostream& operator<< (std::ostream& out, const MPFImageLocation& l) {
    out << "[" << l.x_left_upper << "," << l.y_left_upper << "]-("
               << l.width << "," << l.height << "):" << l.confidence << " ";
    return out;
  } 

  /* **************************************************************************
  * Configuration parameters populsted with appropriate values / defaults
  *************************************************************************** */
  class JobConfig{
    public:
      size_t minDetectionSize;  ///< minimum boounding box dimension
      float  confThresh;        ///< detection confidence threshold

      JobConfig(const log4cxx::LoggerPtr log,const MPFJob &job);
  };
  
  /***************************************************************************/
  class OcvSsdFaceDetection : public MPFImageAndVideoDetectionComponentAdapter {

    public:
      bool Init() override;
      bool Close() override;
      string GetDetectionType();
      MPFDetectionError GetDetections(const MPFVideoJob &job, MPFVideoTrackVec    &tracks)    override;
      MPFDetectionError GetDetections(const MPFImageJob &job, MPFImageLocationVec &locations) override;

    private:
      OcvDetection* _detectorPtr = NULL;
      log4cxx::LoggerPtr _log;

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
