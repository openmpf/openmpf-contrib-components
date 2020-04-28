
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "DetectionLocation.h"

using namespace MPF::COMPONENT;
 
// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr                DetectionLocation::_log;
cv::dnn::Net                      DetectionLocation::_ssdNet;                     ///< single shot DNN face detector network
cv::dnn::Net                      DetectionLocation::_openFaceNet;                ///< feature generator
unique_ptr<dlib::shape_predictor> DetectionLocation::_shapePredFuncPtr  = NULL;   ///< landmark detector function pointer
cv::Ptr<cv::face::FacemarkLBF>    DetectionLocation::_facemarkPtr;                ///< landmark detector



/** ****************************************************************************
*  Draw polylines to visualize landmark features
*
* \param im        image to draw landmarks on
* \param landmarks all landmark point only some of which will be drawn
* \param start     start landmark point index for polyline
* \param end       end landmark point index for polyline
* \param isClosed  if true polyline draw a closed shape (end joined to start)
* \param drawColor color to use for drawing
*
***************************************************************************** */
void drawPolyline(cv::Mat &im, const cvPoint2fVec &landmarks,
                  const int start, const int end, bool isClosed = false,
                  const cv::Scalar drawColor = cv::Scalar(255, 200,0)){
    cvPointVec points;
    for (int i = start; i <= end; i++){
        points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
    }
    cv::polylines(im, points, isClosed, drawColor, 2, 16);
}

/** ****************************************************************************
* Visualize landmark point on image by drawing them.  If 68 landmarks are
* available, they are drawn as polygons, otherwise just as points
*
* \param im        image to draw landmarks on
* \param drawColor color to use for drawing
*
***************************************************************************** */
void DetectionLocation::drawLandmarks(cv::Mat &img,
                   const cv::Scalar drawColor = cv::Scalar(255, 200,0)) const {
  cvPoint2fVec landmarks = getLandmarks();
  if(landmarks.size() == 68){
    drawPolyline(img, landmarks,  0, 16, false, drawColor);    // Jaw line
    drawPolyline(img, landmarks, 17, 21, false, drawColor);    // Left eyebrow
    drawPolyline(img, landmarks, 22, 26, false, drawColor);    // Right eyebrow
    drawPolyline(img, landmarks, 27, 30, false, drawColor);    // Nose bridge
    drawPolyline(img, landmarks, 30, 35, true,  drawColor);    // Lower nose
    drawPolyline(img, landmarks, 36, 41, true,  drawColor);    // Left eye
    drawPolyline(img, landmarks, 42, 47, true,  drawColor);    // Right Eye
    drawPolyline(img, landmarks, 48, 59, true,  drawColor);    // Outer lip
    drawPolyline(img, landmarks, 60, 67, true,  drawColor);    // Inner lip
  }else { 
    for(size_t i = 0; i < landmarks.size(); i++){
      cv::circle(img,landmarks[i],3, drawColor, cv::FILLED);
    }
  }
}

/** **************************************************************************
* Compute 1 - Intersection Over Union metric for two detections comprised of 
* 1 - the ratio of the area of the intersection of the detection rectangels
* divided by the area of the union of the detection rectangles. 
* 
* \param   d second detection 
* \returns   1- intersection over union [0.0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::iouDist(const DetectionLocation &d) const {
	int ulx = max(x_left_upper         , d.x_left_upper           );
	int uly = max(y_left_upper         , d.y_left_upper           );
	int lrx = min(x_left_upper + width , d.x_left_upper + d.width );
	int lry = min(y_left_upper + height, d.y_left_upper + d.height);

	float inter_area = max(0, lrx - ulx + 1) * max(0, lry - uly + 1);	
  float union_area = width * height + d.width * d.height - inter_area;
  return 1.0f - inter_area / union_area;
}

/** **************************************************************************
* Compute the temporal distance in frame counts between two detections
*
* \param   d second detection 
* \returns   absolute difference in frame indecies
*
*************************************************************************** */
float DetectionLocation::frameDist(const DetectionLocation &d) const {
  if(frameIdx > d.frameIdx){
    return frameIdx - d.frameIdx;
  }else{
    return d.frameIdx - frameIdx;
  }
}

/** **************************************************************************
* Compute euclidian center to distance center distance from normalized centers
*
* \param   d second detection 
* \returns   normalized center to center distance [0 ... Sqrt(2)]
*
*************************************************************************** */
float  DetectionLocation::center2CenterDist(const DetectionLocation &d) const {
  float dx = center.x - d.center.x;
  float dy = center.y - d.center.y;
  return sqrt( dx*dx + dy*dy );
}


/** **************************************************************************
* Compute feature distance (similarity) between two detection feature vectors
*
* \param   d second detection 
* \returns cos distance [0 ... 1.0]
*
* \note Feature vectors are expected to be of unit magnitude
*
*************************************************************************** */
float DetectionLocation::featureDist(const DetectionLocation &d) const { 
//  return static_cast<float>(norm(getFeature(), d.getFeature(),cv::NORM_L2));
  return 1.0f - max(0.0f,static_cast<float>(getFeature().dot(d.getFeature())));
}

/** **************************************************************************
* Lazy accessor method to get/compute landmark points
* 5-Landmark detector returns outside and inside eye corners and bottom of nose
* 68-Landmark detector returns "standard" facial landmarks (see data/landmarks.png)
*
* \returns landmarks 
* 
*************************************************************************** */
const cvPoint2fVec& DetectionLocation::getLandmarks() const {
  if(_landmarks.empty()){
    // OpenCV 68-landmark detector
    // cvRectVec detections;                                       
    // for(auto &l:locations){
    //   detections.push_back(cv::Rect(l.x_left_upper,l.y_left_upper,l.width,l.height));
    // }
    // cvPoint2fVecVec landmarks;
    // if(! _facemark->fit(cfg.bgrFrame, detections, landmarks)){
    //   LOG4CXX_WARN(_log,"failed to generate facial landmarks");
    // }

    // DLib 5-landmark detector
    try{
      dlib::cv_image<dlib::bgr_pixel> cimg(_bgrFrame);
      dlib::full_object_detection
        shape = (*_shapePredFuncPtr)(cimg, dlib::rectangle(x_left_upper,               // left
                                                            y_left_upper,              // top
                                                            x_left_upper + width-1,    // right
                                                            y_left_upper + height-1)); // bottom
                                                        
      for(size_t i=0; i<shape.num_parts(); i++){
        dlib::point pt = shape.part(i);                                          //LOG4CXX_TRACE(_log, "lm[" << i << "]: " << shape.part(i));                  
        _landmarks.push_back(cv::Point2f(pt.x(),pt.y()));   
      }
    }catch(...){
      exception_ptr eptr = current_exception();                                                              
      LOG4CXX_FATAL(_log, "failed to determine landmarks");
      rethrow_exception(eptr);      
    }
  }
  return _landmarks;
}

/** **************************************************************************
* Lazy accessor method to get/copy-create 92x92 thumbnail image for 
* feature generation
*
* \returns thumbnail image of detection
*
*************************************************************************** */
const cv::Mat& DetectionLocation::getThumbnail() const {
  if(_thumbnail.empty()){
    const int  THUMBNAIL_SIZE_X = 96;
    const int  THUMBNAIL_SIZE_Y = 96;
    const cv::Size THUMB_SIZE(THUMBNAIL_SIZE_X,THUMBNAIL_SIZE_Y);
      // Landmark indices for OpenFace nn4.v1
      // const size_t lmIdx[] = {39,42,57};
      // const cv::Mat dst = (cv::Mat_<float>(3,2)
      //                      << 0.36378494 * THUMBNAIL_SIZE_X, 0.17794687 * THUMBNAIL_SIZE_Y,
      //                         0.61896320 * THUMBNAIL_SIZE_X, 0.17277813 * THUMBNAIL_SIZE_Y,
      //                         0.50020397 * THUMBNAIL_SIZE_X, 0.75058440 * THUMBNAIL_SIZE_Y); 
      
      // Landmark indecies for OpenFace nn4.v2, nn4.small1.v1, nn4.small2.v1
      //const size_t lmIdx[] = {36,45,33};  // if using 68 pt landmarks
      const size_t lmIdx[] = {2,0,4};       // if using 5 pt landmarks
      const cv::Mat dst =  (cv::Mat_<float>(3,2)
                            << 0.1941570 * THUMBNAIL_SIZE_X, 0.16926692 * THUMBNAIL_SIZE_Y,
                               0.7888591 * THUMBNAIL_SIZE_X, 0.15817115 * THUMBNAIL_SIZE_Y,
                               0.4949509 * THUMBNAIL_SIZE_X, 0.51444140 * THUMBNAIL_SIZE_Y);

      cv::Mat src = cv::Mat_<float>(3,2);
      for(size_t r=0; r<3; r++){
        float* rowPtr = src.ptr<float>(r);
        rowPtr[0] = getLandmarks()[lmIdx[r]].x; 
        rowPtr[1] = getLandmarks()[lmIdx[r]].y;      
      }
      cv::Mat xfrm = cv::getAffineTransform(src,dst);
      try{
        _thumbnail = cv::Mat(THUMB_SIZE,_bgrFrame.type());
        cv::warpAffine(_bgrFrame,_thumbnail,cv::getAffineTransform(src,dst),
                      THUMB_SIZE,cv::INTER_CUBIC,cv::BORDER_REPLICATE);
      }catch (...) { 
        exception_ptr eptr = current_exception();                                                              
        LOG4CXX_FATAL(_log, "failed to generate thumbnail");
        rethrow_exception(eptr);
      }

  }
  return _thumbnail;
}

/** **************************************************************************
* Lazy accessor method to get/compute feature fector based on thumbnail 
*
* \returns unit magnitude feature vector
* 
*************************************************************************** */
const cv::Mat& DetectionLocation::getFeature() const {
  if(_feature.empty()){
    float aspect_ratio = width / height;
    if(   x_left_upper > 0
       && y_left_upper > 0
       && x_left_upper + width  < _bgrFrame.cols - 1
       && y_left_upper + height < _bgrFrame.rows - 1
       || (0.8 < aspect_ratio && aspect_ratio < 1.2)){
      const double inScaleFactor = 1.0 / 255.0;
      const cv::Size blobSize(96, 96);
      const cv::Scalar meanVal(0.0, 0.0, 0.0);  // BGR mean pixel color 
      cv::Mat inputBlob = cv::dnn::blobFromImage(getThumbnail(), // BGR image
                                                inScaleFactor,   // no pixel value scaline (e.g. 1.0/255.0)
                                                blobSize,        // expected network input size: 90x90
                                                meanVal,         // mean BGR pixel value
                                                true,            // swap RB channels
                                                false);          // center crop
      _openFaceNet.setInput(inputBlob);
      _feature = _openFaceNet.forward().clone();             // need to clone as mem gets reused

    }else{                                                   // can't trust feature of detections on the edge
                                                             LOG4CXX_TRACE(_log,"'Zero-feature' for detection at frame edge with poor aspect ratio = " << aspect_ratio);                 
      _feature = cv::Mat::zeros(1, 128, CV_32F);             // zero feature will wipe out any dot products...
    }
  }
  return _feature;
}

/** ****************************************************************************
* Detect objects using SSD DNN opencv face detector network   
* 
* \param  cfg job configuration setting including iamge frame
* 
* \returns found face detections that meet size requirments.
*
* \note each detection hang on to a reference to the bgrFrame which
*       should be released once no longer needed (i.e. features care computed)
*
*************************************************************************** */
DetectionLocationPtrVec DetectionLocation::createDetections(const JobConfig &cfg){
  DetectionLocationPtrVec detections;

  const double inScaleFactor = 1.0;
  const cv::Size blobSize(300, 300);
  const cv::Scalar meanVal(104.0, 117.0, 124.0);  // BGR mean pixel color       

  cv::Mat inputBlob = cv::dnn::blobFromImage(cfg.bgrFrame,  // BGR image
                                            inScaleFactor, // no pixel value scaline (e.g. 1.0/255.0)
                                            blobSize,      // expected network input size: 300x300
                                            meanVal,       // mean BGR pixel value
                                            true,          // swap RB channels
                                            false          // center crop
                                            );
  _ssdNet.setInput(inputBlob,"data");
  cv::Mat detection = _ssdNet.forward("detection_out");
  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

  for(int i = 0; i < detectionMat.rows; i++){
    float conf = detectionMat.at<float>(i, 2);
    if(conf > cfg.confThresh){
      cv::Point2f ul(detectionMat.at<float>(i, 3),detectionMat.at<float>(i, 4));
      cv::Point2f lr(detectionMat.at<float>(i, 5),detectionMat.at<float>(i, 6));
      cv::Point2f wh = lr - ul;
      int x1     = static_cast<int>(ul.x * cfg.bgrFrame.cols);
      int y1     = static_cast<int>(ul.y * cfg.bgrFrame.rows);
      int width  = static_cast<int>(wh.x * cfg.bgrFrame.cols);
      int height = static_cast<int>(wh.y * cfg.bgrFrame.rows);

      if(    width  >= cfg.minDetectionSize
          && height >= cfg.minDetectionSize){
        detections.push_back(unique_ptr<DetectionLocation>(
          new DetectionLocation(x1, y1, width, height, conf, (ul + lr) / 2.0,
                                cfg.frameIdx, cfg.bgrFrame)));                 LOG4CXX_TRACE(_log,"detection:" << *detections.back());
      }
    }                                                                               
  }
  return detections;
}

/** **************************************************************************
* Setup class shared static configurations and initialize / load shared 
* detectors and feature generator objects.
*************************************************************************** */
bool DetectionLocation::Init(log4cxx::LoggerPtr log, string plugin_path){

  _log = log;

  // Load SSD Tensor Flow Network
  string  tf_model_path = plugin_path + "/data/opencv_face_detector_uint8.pb";
  string tf_config_path = plugin_path + "/data/opencv_face_detector.pbtxt";
  string lbf_model_path = plugin_path + "/data/lbfmodel.yaml";
  
  //string  sp_model_path = plugin_path + "/data/shape_predictor_68_face_landmarks.dat";
  string  sp_model_path = plugin_path + "/data/shape_predictor_5_face_landmarks.dat";
  string  tr_model_path = plugin_path + "/data/nn4.small2.v1.t7";

  string err_msg = "Failed to load models: " + tf_config_path + ", " + tf_model_path + ", " + lbf_model_path;

  try{
      // load detector net
      _ssdNet = cv::dnn::readNetFromTensorflow(tf_model_path, tf_config_path);
      
      // load landmark finder
      _facemarkPtr = cv::face::FacemarkLBF::create();
      _facemarkPtr->loadModel(lbf_model_path);

      _shapePredFuncPtr = unique_ptr<dlib::shape_predictor>(new dlib::shape_predictor());
      dlib::deserialize(sp_model_path) >> *_shapePredFuncPtr;

      // load feature generator
      _openFaceNet = cv::dnn::readNetFromTorch(tr_model_path);

  }catch(const runtime_error& re){                                           LOG4CXX_FATAL(_log, err_msg << " Runtime error: " << re.what());
      return false;
  }catch(const exception& ex){                                               LOG4CXX_FATAL(_log, err_msg << " Exception: " << ex.what());
      return false;
  }catch(...){                                                               LOG4CXX_FATAL(_log, err_msg << " Unknown failure occurred. Possible memory corruption");
      return false;
  } 

  return true;
}
