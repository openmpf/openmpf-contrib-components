#ifndef OCVSSDFACEDETECTION_TYPES_H
#define OCVSSDFACEDETECTION_TYPES_H

#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  typedef vector<MPFVideoTrack>    MPFVideoTrackVec;     ///< vector of MPFVideoTracks
  typedef vector<MPFImageLocation> MPFImageLocationVec;  ///< vector of MPFImageLocations
  typedef vector<cv::Mat>          cvMatVec;             ///< vector of OpenCV matricie/images
  typedef vector<cv::Rect>         cvRectVec;            ///< vector of OpenCV rectangles
  typedef vector<cv::Point>        cvPointVec;           ///< vector of OpenCV points     
  typedef vector<cv::Point2f>      cvPoint2fVec;         ///< vector of OpenCV 2D float points
  typedef vector<cvPoint2fVec>     cvPoint2fVecVec;      ///< vector of vectors of OpenCV 2D float points

  /** **************************************************************************
  *   Dump MPFLocation to a stream
  *************************************************************************** */ 
  inline
  ostream& operator<< (ostream& out, const MPFImageLocation& l) {
    out << "[" << l.x_left_upper << "," << l.y_left_upper << "]-("
               << l.width        << "," << l.height       << "):"
               << l.confidence   << " "
              //<< l.detection_properties.at("LANDMARKS")
               ;
    return out;
  }

  /** **************************************************************************
  *   Dump MPFTrack to a stream
  *************************************************************************** */ 
  inline
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
  *   Dump vectors to a stream
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

 }
}



#endif