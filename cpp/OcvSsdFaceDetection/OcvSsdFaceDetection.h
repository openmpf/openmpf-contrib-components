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

// MPF sdk header files 
#include "detectionComponentUtils.h"
#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;
  typedef vector<MPFVideoTrack>    MPFVideoTrackVec;     ///< vector of MPFViseoTracks
  typedef vector<MPFImageLocation> MPFImageLocationVec;  ///< vector of MPFImageLocations

  /* **************************************************************************
  * Conveniance operator to dump MPFLocation to a stream
  *************************************************************************** */ 
  std::ostream& operator<< (std::ostream& out, const MPFImageLocation& l) {
    out << "[" << l.x_left_upper << "," << l.y_left_upper << "]-("
               << l.width        << "," << l.height       << "):"
               << l.confidence   << " ";
    return out;
  }

  /** ****************************************************************************
  * Conveniance << operator template for dumping vectors
  ***************************************************************************** */
  template<typename T>
  std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "{";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i){
      out << v[i];
      if(i != last) out << ", ";
    }
    out << "}";
    return out;
  }

  /** ****************************************************************************
  * shorthands for getting configuration from environment variables if not
  * provided by job configuration
  ***************************************************************************** */
  template<typename T>
  T getEnv(const Properties &p, const string &k, const T def){
    auto iter = p.find(k);
    if (iter == p.end()) {
      const char* env_p = getenv(k.c_str());
      if(env_p != NULL){
        map<string,string> envp;
        envp.insert(pair<string,string>(k,string(env_p)));
        return DetectionComponentUtils::GetProperty<T>(envp,k,def);
      }else{
        return def;
      }
    }
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }

  /** ****************************************************************************
  * shorthands for getting MPF properies of various types
  ***************************************************************************** */
  template<typename T>
  T get(const Properties &p, const string &k, const T def){
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }
  /** ****************************************************************************
  * Macro for throwing exception so we can see where in the code it happened
  ****************************************************************************** */
  #define THROW_EXCEPTION(MSG){                                  \
  string path(__FILE__);                                         \
  string f(path.substr(path.find_last_of("/\\") + 1));           \
  throw runtime_error(f + "[" + to_string(__LINE__)+"] " + MSG); \
  }

  /* **************************************************************************
  * Configuration parameters populated with appropriate values / defaults
  *************************************************************************** */
  class JobConfig{
    public:
      static log4cxx::LoggerPtr _log;  ///< shared log opbject
      size_t minDetectionSize;         ///< minimum boounding box dimension
      float  confThresh;               ///< detection confidence threshold
      JobConfig();
      JobConfig(const MPFJob &job);
  };
  
  /* **************************************************************************
  * Conveniance operator to dump JobConfig to a stream
  *************************************************************************** */ 
  std::ostream& operator<< (std::ostream& out, const JobConfig& cfg) {
    out << "{"
        << "\"minDetectionSize\": " << cfg.minDetectionSize 
        << "\"confThresh\":" << cfg.confThresh 
        << "}";
    return out;
  }

  /***************************************************************************/
  class OcvSsdFaceDetection : public MPFImageAndVideoDetectionComponentAdapter {

    public:
      bool Init() override;
      bool Close() override;
      string GetDetectionType(){return "FACE";};
      MPFDetectionError GetDetections(const MPFVideoJob &job, MPFVideoTrackVec    &tracks)    override;
      MPFDetectionError GetDetections(const MPFImageJob &job, MPFImageLocationVec &locations) override;

    private:

      log4cxx::LoggerPtr _log;     ///< log object
      cv::dnn::Net       _ssdNet;  ///< single shot DNN face detector network

      void _detect(const JobConfig     &cfg,
                   MPFImageLocationVec &locations,
                   const cv::Mat       &bgrFrame);   ///< get bboxes with conf. scores 

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
