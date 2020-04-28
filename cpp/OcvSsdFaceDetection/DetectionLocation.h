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

  typedef vector<unique_ptr<DetectionLocation>> DetectionLocationPtrVec;  ///< vector of DetectionLocation pointers
  typedef DetectionLocationPtrVec               Track;                    ///< track as vector of detection pointers
  typedef list<unique_ptr<Track>>               TrackPtrList;             ///< list of track pointers
  typedef float (DetectionLocation::*DetectionLocationCostFunc)(const DetectionLocation &d) const; ///< cost member function pointer type

  class DetectionLocation: public MPFImageLocation{

    public: 
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors for MPFImageLocation

      const cv::Point2f  center;                 ///< bounding box center normalized to image dimensions
      const size_t     frameIdx;                 ///< frame index frame where detection is located (for videos)

      static bool Init(log4cxx::LoggerPtr log, const string plugin_path);    ///< setup class shared memebers
      static DetectionLocationPtrVec createDetections(const JobConfig &cfg); ///< created detection objects from image frame

      const cvPoint2fVec&  getLandmarks() const;                             ///< get landmark points for detection   
      const cv::Mat&       getThumbnail() const;                             ///< get thumbnail image for detection
      const cv::Mat&       getFeature()   const;                             ///< get DNN features for detection 

      float           iouDist(const DetectionLocation &d) const;             ///< 1 - compute intersection over union
      float         frameDist(const DetectionLocation &d) const;             ///< compute temporal frame gap
      float center2CenterDist(const DetectionLocation &d) const;             ///< compute normalized center to center distance
      float       featureDist(const DetectionLocation &d) const;             ///< compute deep feature similarity distance
      
      void drawLandmarks(cv::Mat &img, const cv::Scalar drawColor) const;    ///< draw landmark point on image
      void releaseBGRFrame(){_bgrFrame.release();};                                                ///< release reference to image frame 

    private:

      static log4cxx::LoggerPtr                _log;                ///< shared log opbject
      static cv::dnn::Net                      _ssdNet;             ///< single shot DNN face detector network
      static cv::dnn::Net                      _openFaceNet   ;     ///< feature generator
      static cv::Ptr<cv::face::FacemarkLBF>    _facemarkPtr;        ///< landmark detector
      static unique_ptr<dlib::shape_predictor> _shapePredFuncPtr;   ///< landmark detector

      cv::Mat  _bgrFrame;                                 ///< frame associated with detection (openCV memory managed :( )
      mutable cvPoint2fVec _landmarks;                    ///< vector of landmarks (e.g. eyes, nose, etc.)
      mutable cv::Mat      _thumbnail;                    ///< 96x96 image comprising an aligned thumbnail
      mutable cv::Mat      _feature;                      ///< DNN feature for matching-up detections

      DetectionLocation(int x,int y,int width,int height,float conf,
                        cv::Point2f center, size_t frameIdx,
                        cv::Mat bgrFrame):
        MPFImageLocation(x,y,width,height,conf),
        center(center),
        frameIdx(frameIdx),
        _bgrFrame(bgrFrame)
        {};              ///< private constructor for createDetections()
  };
 
  /** **************************************************************************
  * Conveniance operator to dump MPFLocation to a stream
  *************************************************************************** */
  inline 
  ostream& operator<< (ostream& out, const DetectionLocation& d) {
    out  << "[" << (MPFImageLocation)d 
              // << " L:" << d.landmarks << " F["    
                << d.getFeature().size() << "] T["
                << d.getThumbnail().rows << "," << d.getThumbnail().cols << "]";
    return out;
  }

  /** **************************************************************************
  * Conveniance operator to dump MPF::COMPONENT::Track to a stream
  *************************************************************************** */
  inline 
  ostream& operator<< (ostream& out, const Track& t) {
    out << "<f"   << t.front()->frameIdx << (MPFImageLocation)(*t.front())
        << "...f" << t.back()->frameIdx << (MPFImageLocation)(*t.back()) 
        << ">("<<t.size()<<")";
    return out;
  }

 }
}

#endif