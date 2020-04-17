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

#include "OcvSsdFaceDetection.h"

#include <algorithm>
#include <stdexcept>
#include <limits.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgcodecs.hpp>

#include <log4cxx/xml/domconfigurator.h>

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QString>

// 3rd party code
#include "munkres.h"

// MPF-SDK header files 
#include "Utils.h"
#include "MPFSimpleConfigLoader.h"
#include "detectionComponentUtils.h"


using namespace MPF::COMPONENT;

// Temporary initializer for static member variable
log4cxx::LoggerPtr JobConfig::_log = log4cxx::Logger::getRootLogger();


string format(cv::Mat m){
  stringstream ss;
  ss << m;
  string str = ss.str();
  str.erase(remove(str.begin(),str.end(),'\n'),str.end());
  return str;
}

/** ****************************************************************************
* Draw polylines for landmark features 
***************************************************************************** */
const cv::Scalar DRAW_COLOR(255, 200,0);
void drawPolyline(cv::Mat &im, const cvPoint2fVec &landmarks,
                  const int start, const int end, bool isClosed = false){
    cvPointVec points;
    for (int i = start; i <= end; i++){
        points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
    }
    cv::polylines(im, points, isClosed, DRAW_COLOR, 2, 16);
}

/** ****************************************************************************
* Draw landmark features
***************************************************************************** */
cv::Mat OcvSsdFaceDetection::_drawLandmarks(const JobConfig      &cfg,
                                            DetectionLocationVec &detections){
  cv::Mat img = cfg.bgrFrame.clone();
  for(auto &det:detections){
    if(det.landmarks.size() == 68){
      drawPolyline(img, det.landmarks, 0, 16);           // Jaw line
      drawPolyline(img, det.landmarks, 17, 21);          // Left eyebrow
      drawPolyline(img, det.landmarks, 22, 26);          // Right eyebrow
      drawPolyline(img, det.landmarks, 27, 30);          // Nose bridge
      drawPolyline(img, det.landmarks, 30, 35, true);    // Lower nose
      drawPolyline(img, det.landmarks, 36, 41, true);    // Left eye
      drawPolyline(img, det.landmarks, 42, 47, true);    // Right Eye
      drawPolyline(img, det.landmarks, 48, 59, true);    // Outer lip
      drawPolyline(img, det.landmarks, 60, 67, true);    // Inner lip
    }else { 
      for(size_t i = 0; i < det.landmarks.size(); i++){
        cv::circle(img,det.landmarks[i],3, DRAW_COLOR, cv::FILLED);
      }
    }
  }
  return img;
}

/** ****************************************************************************
* Initialize SSD face detector module by setting up paths and reading configs
* configuration variables are turned into environment variables for late
* reference
*
* \returns   true on success
***************************************************************************** */
bool OcvSsdFaceDetection::Init() {
    string plugin_path    = GetRunDirectory() + "/OcvSsdFaceDetection";
    string config_path    = plugin_path       + "/config";

    log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
    _log = log4cxx::Logger::getLogger("OcvSsdFaceDetection");                  LOG4CXX_DEBUG(_log,"Initializing OcvSSDFaceDetector");
    JobConfig::_log = _log;                                                    //LOG4CXX_TRACE(_log, cv::getBuildInformation() << std::endl);


    // read config file and create update any missing env variables
    string config_params_path = config_path + "/mpfOcvSsdFaceDetection.ini";
    QHash<QString,QString> params;
    if(LoadConfig(config_params_path, params)) {                               LOG4CXX_ERROR(_log, "Failed to load the OcvSsdFaceDetection config from: " << config_params_path);
        return false;
    }                                                                          LOG4CXX_TRACE(_log,"read config file:" << config_params_path);
    for(auto p = params.begin(); p != params.end(); p++){
      const string key = p.key().toStdString();                                   
      const string val = p.value().toStdString();                              LOG4CXX_TRACE(_log,"Config    Vars:" << key << "=" << val);
      const char* env_val = getenv(key.c_str());                               LOG4CXX_TRACE(_log,"Veryfying ENVs:" << key << "=" << env_val);
      if(env_val == NULL){
        if(setenv(key.c_str(), val.c_str(), 0) !=0){                           LOG4CXX_ERROR(_log, "Failed to convert config to env variable: " << key << "=" << val);
          return false;
        }
      }else if(string(env_val) != val){                                        LOG4CXX_INFO(_log, "Keeping existing env variable:" << key << "=" << string(env_val));
      }
    }

    // Load SSD Tensor Flow Network
    string  tf_model_path = plugin_path + "/data/opencv_face_detector_uint8.pb";
    string tf_config_path = plugin_path + "/data/opencv_face_detector.pbtxt";
    string lbf_model_path = plugin_path + "/data/lbfmodel.yaml";
    //string  sp_model_path = plugin_path + "/data/shape_predictor_68_face_landmarks.dat";
    string  sp_model_path = plugin_path + "/data/shape_predictor_5_face_landmarks.dat";
    string  tr_model_path = plugin_path + "/data/nn4.small2.v1.t7";

    string err_msg = "Failed to load models: " + tf_config_path + ", " + tf_model_path + ", " + lbf_model_path;

    try{
        // load detecore net
        _ssdNet = cv::dnn::readNetFromTensorflow(tf_model_path, tf_config_path);
        
        // load landmark finder
        _facemark = cv::face::FacemarkLBF::create();
        _facemark->loadModel(lbf_model_path);

        dlib::deserialize(sp_model_path) >> _shape_predictor;

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

/** ****************************************************************************
* Clean up and release any created detector objects
*
* \returns   true on success
***************************************************************************** */
bool OcvSsdFaceDetection::Close() {
    return true;
}

/** ****************************************************************************
* Detect objects using SSD DNN opencv detector network.   
* 
* \param         cfg       Config settings
* \param[in,out] locations detections to add to    
*
***************************************************************************** */
void OcvSsdFaceDetection::_detect(const JobConfig      &cfg,
                                  DetectionLocationVec &detections){
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
        detections.push_back(DetectionLocation(x1, y1, width, height, conf,
                                               (ul + lr) / 2.0, cfg.frameIdx));
      }
    }                                                                               
  }                                              
}

/** ****************************************************************************
* Get landmark points for detections.   
* 
* \param         cfg       Config settings
* \param[in,out] detections to find landmarks from   
*
* \returns landmarks for each detection  
*
***************************************************************************** */
void OcvSsdFaceDetection::_findLandmarks(const JobConfig      &cfg,
                                         DetectionLocationVec &detections){
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
  cvPoint2fVecVec landmarks;
  dlib::cv_image<dlib::bgr_pixel> cimg(cfg.bgrFrame);
  for(auto &det:detections){
    dlib::full_object_detection
       shape = _shape_predictor(cimg, dlib::rectangle(det.x_left_upper,                  // left
                                                      det.y_left_upper,                  // top
                                                      det.x_left_upper + det.width-1,    // right
                                                      det.y_left_upper + det.height-1)); // bottom
                                                      
    for(size_t i=0; i<shape.num_parts(); i++){
      dlib::point pt = shape.part(i);                                          //LOG4CXX_TRACE(_log, "lm[" << i << "]: " << shape.part(i));                  
      det.landmarks.push_back(cv::Point2f(pt.x(),pt.y()));   
    }
  }
  // remove detections that failed to produce landmarks 5 landmarks
  detections.erase(remove_if(detections.begin(),
                             detections.end(),
                             [](DetectionLocation const& d){return d.landmarks.size()<5;}),
                   detections.end());
}

/** ****************************************************************************
* Get aligned thumbnails from detections using landmarks.   
* 
* \param         cfg        Config settings
* \param[in,out] detections to which features will be added    
*
*
***************************************************************************** */
void OcvSsdFaceDetection::_createThumbnails(const JobConfig      &cfg,
                                            DetectionLocationVec &detections){
  const float  THUMBNAIL_SIZE = 96;
  const cv::Size THUMB_SIZE(THUMBNAIL_SIZE,THUMBNAIL_SIZE);

  // Landmark indices for OpenFace nn4.v1
  // const size_t lmIdx[] = {39,42,57};
  // const cv::Mat dst =  THUMBNAIL_SIZE * (cv::Mat_<float>(3,2)
  //                                         << 0.36378494, 0.17794687,
  //                                            0.61896320, 0.17277813,
  //                                            0.50020397, 0.75058440); 
  
  // Landmark indecies for OpenFace nn4.v2, nn4.small1.v1, nn4.small2.v1
  //const size_t lmIdx[] = {36,45,33};  // 68 pt landmarks
  const size_t lmIdx[] = {2,0,4};       // 5 pt landmarks
  const cv::Mat dst = THUMBNAIL_SIZE * (cv::Mat_<float>(3,2)
                                         << 0.1941570 , 0.16926692,
                                            0.7888591 , 0.15817115,
                                            0.4949509 , 0.51444140);
  cv::Mat src = cv::Mat_<float>(3,2);
  
  for(auto &det:detections){
    for(size_t r=0; r<3; r++){
      float* rowPtr = src.ptr<float>(r);
      rowPtr[0] = det.landmarks[lmIdx[r]].x; 
      rowPtr[1] = det.landmarks[lmIdx[r]].y;      
    }
    cv::Mat xfrm = cv::getAffineTransform(src,dst);
    det.thumbnail = cv::Mat(THUMB_SIZE,cfg.bgrFrame.type());
    cv::warpAffine(cfg.bgrFrame,det.thumbnail,cv::getAffineTransform(src,dst),
                   THUMB_SIZE,cv::INTER_CUBIC,cv::BORDER_REPLICATE);
  }
}

/** ****************************************************************************
* Get similarity features from thumbnails   
* 
* \param         cfg        config settings
* \param         thumbnails landmark aligned thumbnails
*
* \returns features for detections
*
***************************************************************************** */
void OcvSsdFaceDetection::_calcFeatures(const JobConfig &cfg,
                                       DetectionLocationVec &detections){
  const double inScaleFactor = 1.0 / 255.0;
  const cv::Size blobSize(96, 96);
  const cv::Scalar meanVal(0.0, 0.0, 0.0);  // BGR mean pixel color 

  for(auto &det:detections){
    cv::Mat inputBlob = cv::dnn::blobFromImage(det.thumbnail, // BGR image
                                              inScaleFactor,  // no pixel value scaline (e.g. 1.0/255.0)
                                              blobSize,       // expected network input size: 300x300
                                              meanVal,        // mean BGR pixel value
                                              true,           // swap RB channels
                                              false);         // center crop
    _openFaceNet.setInput(inputBlob);
    det.feature = _openFaceNet.forward().clone();             // need to clone as mem gets reused
  }
}

/** ****************************************************************************
* Computes distance squared between centers of location bounding boxes
***************************************************************************** */
float iou(const DetectionLocation &a, const DetectionLocation &b){
  
	// determine the (x, y)-coordinates of the intersection rectangle
	int ulx = max(a.x_left_upper, b.x_left_upper);
	int uly = max(a.y_left_upper, b.y_left_upper);
	int lrx = min(a.x_left_upper + a.width,  b.x_left_upper + b.width);
	int lry = min(a.y_left_upper + a.height, b.y_left_upper + b.height);

	// compute the area of intersection rectangle
	float inter_area = max(0, lrx - ulx + 1) * max(0, lry - uly + 1);	
  float union_area = a.width * a.height + b.width * b.height - inter_area;

  return inter_area / union_area;
}

/** ****************************************************************************
* Computes distance squared between centers of location bounding boxes
***************************************************************************** */
int OcvSsdFaceDetection::_calcAssignmentCost(const JobConfig         &cfg,
                                             const DetectionLocation &a,
                                             const DetectionLocation &b){

                                                                              LOG4CXX_TRACE(_log, "trk = " << a);
                                                                              LOG4CXX_TRACE(_log, "det = " << b);
  // frame gap cost
  float frameGapCost = (a.frameIdx > b.frameIdx) ? a.frameIdx-b.frameIdx
                                                 : b.frameIdx-a.frameIdx; 
  if(frameGapCost > cfg.maxFrameGap){                                         LOG4CXX_TRACE(_log,"frameGapCost="<< frameGapCost <<" > " << cfg.maxFrameGap << "=max");
    return INT_MAX;
  } 

  // intersection over union cost
  float iouCost = 1.0f - iou(a,b);
  if(iouCost > cfg.maxIOUDist){                                               LOG4CXX_TRACE(_log,"iouCost="<< iouCost <<" > " << cfg.maxIOUDist << "=max");
    return INT_MAX;
  }

  // center to center distance cost
  cv::Point2f d = a.center - b.center;
  d.x *= cfg.widthOdiag;
  d.y *= cfg.heightOdiag;
  float centerDistCost = sqrt(d.x*d.x + d.y*d.y);
  if(centerDistCost > cfg.maxCenterDist){                                     LOG4CXX_TRACE(_log,"centerDistCost="<< centerDistCost <<" > " << cfg.maxCenterDist << "=max");
    return INT_MAX;
  }                  

  // openFace feature cost
  float featureCost = norm(a.feature, b.feature,cv::NORM_L2);                                                                     
  if(featureCost > cfg.maxFeatureDist){                                       LOG4CXX_TRACE(_log,"featureCost="<< featureCost <<" > " << cfg.maxFeatureDist << "=max");
    return INT_MAX;
  }
                                                                              LOG4CXX_TRACE(_log, "costs [frame,iou,centd,feat] = [" << frameGapCost << "," << iouCost << "," << centerDistCost << "," << featureCost << "]");
  return static_cast<int>(1000.0 * (
                          cfg.featureWeight    * featureCost + 
                          cfg.centerDistWeight * centerDistCost +
                          cfg.frameGapWeight   * frameGapCost));
}

/** ****************************************************************************
* Computes distance squared between centers of location bounding boxes
***************************************************************************** */
cv::Mat_<int> OcvSsdFaceDetection::_calcAssignemntMatrix(const JobConfig            &cfg,
                                                         const TrackList            &tracks,
                                                         const DetectionLocationVec &detections){
  // rows -> tracks, cols -> detections
  cv::Mat_<int> costs(tracks.size(),detections.size());
  size_t r = 0;
  for(auto &track:tracks){
    for(size_t c=0; c<costs.cols; c++){
      costs[r][c] = _calcAssignmentCost(cfg, track.back(), detections[c]);
    }
    r++;
  }                                                                           LOG4CXX_TRACE(_log,"Cost Matrix[" << costs.rows << "," << costs.cols << "]: " << format(costs));

  Munkres lap;
  cv::Mat_<int> am = costs.clone();
  lap.solve(am);                                                              LOG4CXX_TRACE(_log,"Solved Matrix: " << format(am));                                       

  // knock out assignments that are too costly (i.e. new track needed)
  for(size_t r=0; r<costs.rows; r++){
    //int* costRowPtr = costs.ptr<int>(r);
    for(size_t c=0; c<costs.cols; c++){
      if(costs[r][c] == INT_MAX){
        am[r][c] = -1;                   // remove potential assignment (-1)
      }
    }
  }                                                                           LOG4CXX_TRACE(_log,"Modified Matrix: " << format(am));
  return am;
}

/** ****************************************************************************
* Computes distance squared between centers of location bounding boxes
***************************************************************************** */
void OcvSsdFaceDetection::_assignDetections2Tracks(const JobConfig       &cfg,
                                                   TrackList             &tracks,
                                                   DetectionLocationVec  &detections,
                                                   const cv::Mat_<int>   &assignmentMatrix){
  DetectionLocation dummy(-1,-1,-1,-1);
  dummy.frameIdx = -1;
  size_t r=0;                                                   
  for(auto &track:tracks){
    const int* rowPtr = assignmentMatrix.ptr<int>(r);
    for(size_t c=0; c<assignmentMatrix.cols; c++){
      if(rowPtr[c] == 0){
        track.push_back(detections[c]);              LOG4CXX_TRACE(_log,"assigning det: " << detections[c] << "to track " << track);
        detections[c] = dummy;
      }
    }
    r++;
  }

  // remove detections that were assigned to tracks
  detections.erase(remove_if(detections.begin(),
                             detections.end(),
                             [](DetectionLocation const& d){return d.frameIdx == -1;}),
                   detections.end());

}

/** ****************************************************************************
* Read an image and get object detections and features
*
* \param          job     MPF Image job
* \param[in,out]  locations  locations collection to which detections will be added
*
* \returns  an MPF error constant or MPF_DETECTION_SUCCESS
*
* \note prior to returning the features are base64 encoded ?
*
***************************************************************************** */
MPFDetectionError OcvSsdFaceDetection::GetDetections(const MPFImageJob   &job,
                                                     MPFImageLocationVec &locations) {

  try {                                                                        LOG4CXX_DEBUG(_log, "[" << job.job_name << "Data URI = " << job.data_uri);

    JobConfig cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS) return cfg.lastError;

    DetectionLocationVec detections;
    _detect(cfg, detections);                                                  LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of faces detected = " << detections.size());

    for(auto &det:detections){                                                 
      cfg.ReverseTransform(det);
      locations.push_back(det);
    }

  }catch(const runtime_error& re){                                             LOG4CXX_FATAL(_log, "[" << job.job_name << "] runtime error: " << re.what());
    return Utils::HandleDetectionException(job, _log);
  }catch(const exception& ex){                                                 LOG4CXX_FATAL(_log, "[" << job.job_name << "] exception: " << ex.what());
    return Utils::HandleDetectionException(job, _log);
  }catch (...) {                                                               LOG4CXX_FATAL(_log, "[" << job.job_name << "] unknown error");
    return Utils::HandleDetectionException(job, _log);                         
  }
                                                                               LOG4CXX_DEBUG(_log,"[" << job.job_name << "] complete.");
  return MPF_DETECTION_SUCCESS;
}

/** ****************************************************************************
* Convert track to MPFVideoTracks
*
* \param          job     MPF Video job
* \param[in,out]  tracks  Tracks collection to which detections will be added
*
* \returns
*
***************************************************************************** */
MPFVideoTrack OcvSsdFaceDetection::_convert_track(const Track &track){

  MPFVideoTrack mpf_track;
  mpf_track.start_frame = track.front().frameIdx;
  mpf_track.stop_frame  = track.back().frameIdx;
  stringstream start_feature;
  stringstream stop_feature;
  start_feature << track.front().feature;
  stop_feature << track.back().feature;
  mpf_track.detection_properties["START_FEATURE"] = start_feature.str();
  mpf_track.detection_properties["STOP_FEATURE"]  = stop_feature.str();
  for(auto &det:track){
    mpf_track.confidence += det.confidence;                                                    
    mpf_track.frame_locations.insert(mpf_track.frame_locations.end(),{det.frameIdx,det});
  }
  mpf_track.confidence /= static_cast<float>(track.size());

  return mpf_track;
}

/** ****************************************************************************
* Read frames from a video, get object detections and make tracks
*
* \param          job     MPF Video job
* \param[in,out]  tracks  Tracks collection to which detections will be added
*
* \returns  an MPF error constant or MPF_DETECTION_SUCCESS
*
***************************************************************************** */
MPFDetectionError OcvSsdFaceDetection::GetDetections(const MPFVideoJob &job,
                                                     MPFVideoTrackVec  &mpf_tracks) {

  try{ 

    TrackList tracks;

    JobConfig cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS) return cfg.lastError;

    while(cfg.nextFrame()) {                                                   LOG4CXX_TRACE(_log, "processing frame " << cfg.frameIdx);                                
      DetectionLocationVec detections;
      _detect(cfg, detections);
      if(detections.size() > 0){                                               // found some detections in current frame
        tracks.remove_if([&](const Track& t){                                  // remove any tracks too far in the past
          if(cfg.frameIdx - t.back().frameIdx > cfg.maxFrameGap){
            mpf_tracks.push_back(_convert_track(t));                           LOG4CXX_TRACE(_log,"droping old track: " << t);
            return true;
          }else{
            return false;
          }
        }); 

        _findLandmarks(cfg, detections);                                       // get their landmarks 
        _createThumbnails(cfg, detections);                                    // create aligned thumbnails
        _calcFeatures(cfg, detections);                                        // calculate features from thumbnails
 

        if(tracks.size() >= 0 ){                                               // some current tracks exist, so determine potential assignments
          cv::Mat_<int> am = _calcAssignemntMatrix(cfg,tracks,detections);     //  assign detection to current tracks
          _assignDetections2Tracks(cfg,tracks,detections, am);
        }
        for(auto &det:detections){                                             // make any unassigned detections into new tracks
            tracks.push_back({det});                                           LOG4CXX_TRACE(_log,"created new track " << tracks.back());
        }
      }
    }                                                                          LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of tracks detected = " << tracks.size());

    for(auto &track:tracks){
      mpf_tracks.push_back(_convert_track(track));
    }

    for(auto &mpf_track:mpf_tracks){
      cfg.ReverseTransform(mpf_track);
    }

  }catch(const runtime_error& re){                                             LOG4CXX_FATAL(_log, "[" << job.job_name << "] runtime error: " << re.what());
    return Utils::HandleDetectionException(job, _log);
  }catch(const exception& ex){                                                 LOG4CXX_FATAL(_log, "[" << job.job_name << "] exception: " << ex.what());
    return Utils::HandleDetectionException(job, _log);
  }catch (...) {                                                               LOG4CXX_FATAL(_log, "[" << job.job_name << "] unknown error");
    return Utils::HandleDetectionException(job, _log);                         
  }
                                                                               LOG4CXX_DEBUG(_log,"[" << job.job_name << "] complete.");
  return MPF_DETECTION_SUCCESS;
}

MPF_COMPONENT_CREATOR(OcvSsdFaceDetection);
MPF_COMPONENT_DELETER();
