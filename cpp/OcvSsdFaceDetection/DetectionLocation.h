#ifndef OCVSSDFACEDETECTION_DETECTIONLOCATION_H
#define OCVSSDFACEDETECTION_DETECTIONLOCATION_H

#include "types.h" 

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  class DetectionLocation: public MPFImageLocation{

    public:
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors

      cv::Point2f  center;                       ///< bounding box center normalized to image size
      size_t       frameIdx;                     ///< frame index of detection (for videos)

      cvPoint2fVec landmarks;                    ///< vector of landmarks (e.g. eyes, nose, etc.)
      cv::Mat      thumbnail;                    ///< 96x96 image comprising an aligned thumbnail
      cv::Mat      feature;                      ///< DNN feature for matching-up detections

      DetectionLocation(int x,int y,int width,int height,float conf,
                        cv::Point2f center, size_t frameIdx):
        MPFImageLocation(x,y,width,height,conf),
        center(center),
        frameIdx(frameIdx){};

  };

  typedef vector<DetectionLocation> DetectionLocationVec;       ///< vector of DetectionLocations
  typedef DetectionLocationVec      Track;                      ///< a track is an ordered list of detections
  typedef list<Track>               TrackList;                  ///< list of tracks
  
  /* **************************************************************************
  * Conveniance operator to dump MPFLocation to a stream
  *************************************************************************** */ 
  ostream& operator<< (ostream& out, const DetectionLocation& d) {
    out << "[" << (MPFImageLocation)d 
              // << " L:" << d.landmarks << " F["
               << d.feature.size() << "] T["
               << d.thumbnail.rows << "," << d.thumbnail.cols << "]";
    return out;
  }

  ostream& operator<< (ostream& out, const Track& t) {
    DetectionLocation d = t.back();
    out << "("<<t.size()<<") ... [" << (MPFImageLocation)d 
               << d.feature.size() << "] T["
               << d.thumbnail.rows << "," << d.thumbnail.cols << "]";
    return out;
  }

 }
}



#endif