#ifndef OCVFACEDETECTION_STREAMIO_H
#define OCVFACEDETECTION_STREAMIO_H

#include <log4cxx/logger.h>
#include <iostream>

#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

#include "OcvSsdFaceDetection_JobConfig.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  /* **************************************************************************
  * Conveniance operator to dump MPFLocation to a stream
  *************************************************************************** */ 
  ostream& operator<< (ostream& out, const MPFImageLocation& l) {
    out << "[" << l.x_left_upper << "," << l.y_left_upper << "]-("
               << l.width        << "," << l.height       << "):"
               << l.confidence   << " "
              //<< l.detection_properties.at("LANDMARKS")
               ;
    return out;
  }

  /* **************************************************************************
  * Conveniance operator to dump MPFTrack to a stream
  *************************************************************************** */ 
  ostream& operator<< (ostream& out, const MPFVideoTrack& t) {
    out << t.start_frame << endl;
    out << t.stop_frame  << endl;
    for(auto& p:t.frame_locations){
      out << p.second.x_left_upper << "," << p.second.y_left_upper << ","
          << p.second.width << "," << p.second.height << endl;
    }
    return out;
  }

  /** ****************************************************************************
  * Conveniance << operator template for dumping vectors
  ***************************************************************************** */
  template<typename T>
  ostream& operator<< (ostream& out, const vector<T>& v) {
    out << "{";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i){
      out << v[i];
      if(i != last) out << ", ";
    }
    out << "}";
    return out;
  }

  /* **************************************************************************
  * Conveniance operator to dump JobConfig to a stream
  *************************************************************************** */ 
  ostream& operator<< (ostream& out, const JobConfig& cfg) {
    out << "{"
        << "\"minDetectionSize\": " << cfg.minDetectionSize 
        << "\"confThresh\":" << cfg.confThresh 
        << "}";
    return out;
  }

 }
}

#endif //OPENMPF_COMPONENTS_OCVFACEDETECTION_H