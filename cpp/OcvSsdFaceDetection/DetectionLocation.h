#ifndef OCVSSDFACEDETECTION_DETECTIONLOCATION_H
#define OCVSSDFACEDETECTION_DETECTIONLOCATION_H

#include "types.h" 

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  class DetectionLocation: public MPFImageLocation{

    public:
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors

      cvPoint2fVec landmarks;                    ///< vector of landmarks (e.g. eyes, nose, etc.)
      cv::Mat      thumbnail;                    ///< 96x96 image comprising an aligned thumbnail
      cv::Mat      feature;                      ///< DNN feature for matching-up detections

  };

  typedef vector<DetectionLocation> DetectionLocationVec;  ///< vector of DetectionLocations

  /* **************************************************************************
  * Conveniance operator to dump MPFLocation to a stream
  *************************************************************************** */ 
  ostream& operator<< (ostream& out, const DetectionLocation& d) {
    out << "[" << (MPFImageLocation)d 
               << " L:" << d.landmarks << " F["
               << d.feature.size() << "] T["
               << d.thumbnail.rows << "," << d.thumbnail.cols << "]";
    return out;
  }

 }
}



#endif