#ifndef OCVSSDFACEDETECTION_DETECTIONLOCATION_H
#define OCVSSDFACEDETECTION_DETECTIONLOCATION_H

#include <log4cxx/logger.h>

#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include "types.h" 
#include "JobConfig.h"

#define CALL_MEMBER_FUNC(object,ptrToMember)  ((object).*(ptrToMember))

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  class DetectionLocation;

  typedef vector<DetectionLocation> DetectionLocationVec;       ///< vector of DetectionLocations
  typedef DetectionLocationVec      Track;                      ///< track as an ordered list of detections
  typedef list<Track>               TrackList;                  ///< list of tracks
  typedef float (DetectionLocation::*DetectionLocationCostFunc)(const DetectionLocation &d); ///< cost member function pointer type

  class DetectionLocation: public MPFImageLocation{

    public: 
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors for MPFImageLocation

      const cv::Point2f  center;                 ///< bounding box center normalized to image dimensions
      const size_t     frameIdx;                 ///< frame index frame where detection is located (for videos)

      static bool Init(log4cxx::LoggerPtr log,
                       const string       plugin_path);                     ///< setup class shared memebers

      static DetectionLocationVec createDetections(const JobConfig &cfg);   ///< created detection objects from image frame

      const cvPoint2fVec&  getLandmarks() const;                            ///< get landmark points for detection   
      const cv::Mat&       getThumbnail() const;                            ///< get thumbnail image for detection
      const cv::Mat&       getFeature()   const;                            ///< get DNN features for detection 

      float               iou(const DetectionLocation &d) const;            ///< compute intersection over union
      float          frameGap(const DetectionLocation &d) const;            ///< compute temporal frame gap
      float center2CenterDist(const DetectionLocation &d) const;            ///< compute normalized center to center distance
      float       featureDist(const DetectionLocation &d) const;            ///< compute deep feature similarity distance
      
      void drawLandmarks(cv::Mat &img, const cv::Scalar drawColor) const;   ///< draw landmark point on image
      void releaseBGRFrame(){_bgrFramePtr.reset();}                         ///< release reference to image frame 

    private:

      static log4cxx::LoggerPtr                _log;                ///< shared log opbject
      static unique_ptr<cv::dnn::Net>          _ssdNetPtr;          ///< single shot DNN face detector network
      static cv::Ptr<cv::face::FacemarkLBF>    _facemarkPtr;        ///< landmark detector
      static unique_ptr<dlib::shape_predictor> _shapePredFuncPtr;   ///< landmark detector
      static unique_ptr<cv::dnn::Net>          _openFaceNetPtr;     ///< feature generator


      shared_ptr<cv::Mat>  _bgrFramePtr;                  ///< pointer to frame associated with detection
      mutable cvPoint2fVec _landmarks;                    ///< vector of landmarks (e.g. eyes, nose, etc.)
      mutable cv::Mat      _thumbnail;                    ///< 96x96 image comprising an aligned thumbnail
      mutable cv::Mat      _feature;                      ///< DNN feature for matching-up detections

      DetectionLocation(int x,int y,int width,int height,float conf,
                        cv::Point2f center, size_t frameIdx,
                        shared_ptr<cv::Mat> bgrFramePtr):
        MPFImageLocation(x,y,width,height,conf),
        center(center),
        frameIdx(frameIdx),
        _bgrFramePtr(bgrFramePtr){};              ///< private constructor for createDetections()
  };
 
  ostream& operator<< (ostream& out, const MPF::COMPONENT::DetectionLocation& d); 
  ostream& operator<< (ostream& out, const MPF::COMPONENT::Track& t);

 }
}

#endif