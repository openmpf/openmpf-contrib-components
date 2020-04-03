#ifndef OCVFACEDETECTION_TYPES_H
#define OCVFACEDETECTION_TYPES_H

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

 }
}

#endif