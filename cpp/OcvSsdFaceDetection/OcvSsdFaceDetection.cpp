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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgcodecs.hpp>

#include <log4cxx/xml/domconfigurator.h>

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QString>

// MPF sdk header files 
#include "Utils.h"
#include "MPFSimpleConfigLoader.h"
#include "MPFImageReader.h"
#include "MPFVideoCapture.h"
#include "detectionComponentUtils.h"


using namespace MPF::COMPONENT;

// Temporary initializer for static member variable
log4cxx::LoggerPtr JobConfig::_log = log4cxx::Logger::getRootLogger();



/** ****************************************************************************
* Default constructor
***************************************************************************** */
JobConfig::JobConfig():
  minDetectionSize(45),
  confThresh(0.65){}                    

/** ****************************************************************************
* Parse setting out of MPFJob
***************************************************************************** */
JobConfig::JobConfig(const MPFJob &job) : JobConfig() {
  const Properties jpr = job.job_properties;
  minDetectionSize = getEnv<int>  (jpr,"MIN_DETECTION_SIZE", minDetectionSize);       LOG4CXX_TRACE(_log, "MIN_DETECTION_SIZE: " << minDetectionSize);
  confThresh       = getEnv<float>(jpr,"DETECTION_CONFIDENCE_THRESHOLD", confThresh); LOG4CXX_TRACE(_log, "DETECTION_CONFIDENCE_THRESHOLD: " << confThresh);
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
void OcvSsdFaceDetection::_drawLandmarks(cv::Mat &im, cvPoint2fVec &landmarks){
  if (landmarks.size() == 68){
    drawPolyline(im, landmarks, 0, 16);           // Jaw line
    drawPolyline(im, landmarks, 17, 21);          // Left eyebrow
    drawPolyline(im, landmarks, 22, 26);          // Right eyebrow
    drawPolyline(im, landmarks, 27, 30);          // Nose bridge
    drawPolyline(im, landmarks, 30, 35, true);    // Lower nose
    drawPolyline(im, landmarks, 36, 41, true);    // Left eye
    drawPolyline(im, landmarks, 42, 47, true);    // Right Eye
    drawPolyline(im, landmarks, 48, 59, true);    // Outer lip
    drawPolyline(im, landmarks, 60, 67, true);    // Inner lip
  }else { 
    for(int i = 0; i < landmarks.size(); i++){
      cv::circle(im,landmarks[i],3, DRAW_COLOR, cv::FILLED);
    }
  }
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
    JobConfig::_log = _log;                                                    LOG4CXX_TRACE(_log, cv::getBuildInformation() << std::endl);


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
* \param         bgrFrame  BGR color image in which to detect objects
*
***************************************************************************** */
void OcvSsdFaceDetection::_detect(const JobConfig     &cfg,
                                  MPFImageLocationVec &locations,
                                  const cv::Mat       &bgrFrame){
  const double inScaleFactor = 1.0;
  const cv::Size blobSize(300, 300);
  const cv::Scalar meanVal(104.0, 117.0, 124.0);  // BGR mean pixel color       

  cv::Mat inputBlob = cv::dnn::blobFromImage(bgrFrame,      // BGR image
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
      int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * bgrFrame.cols);
      int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * bgrFrame.rows);
      int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * bgrFrame.cols);
      int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * bgrFrame.rows);
      int width  = x2 - x1;
      int height = y2 - y1;
      if(   width  >= cfg.minDetectionSize
          && height >= cfg.minDetectionSize){
        locations.push_back(MPFImageLocation(x1, y1, width, height, conf));
      }
    }
  }                                              
}

/** ****************************************************************************
* Get landmark points for detections.   
* 
* \param         cfg       Config settings
* \param[in,out] locations detections to find landmarks from   
* \param         bgrFrame  BGR color image in which detections were found
*
* \returns landmarks for each detection  
*
***************************************************************************** */
cvPoint2fVecVec OcvSsdFaceDetection::_getLandmarks(const JobConfig     &cfg,
                                       MPFImageLocationVec &locations,
                                       const cv::Mat       &bgrFrame){
  // cvRectVec detections;                                       
  // for(auto &l:locations){
  //   detections.push_back(cv::Rect(l.x_left_upper,l.y_left_upper,l.width,l.height));
  // }
  // cvPoint2fVecVec landmarks;
  // if(! _facemark->fit(bgrFrame, detections, landmarks)){
  //   LOG4CXX_WARN(_log,"failed to generate facial landmarks");
  // }

   cvPoint2fVecVec landmarks;
  vector<dlib::rectangle> detections;
  dlib::cv_image<dlib::bgr_pixel> cimg(bgrFrame);
  for(auto &l:locations){
    dlib::full_object_detection
       shape = _shape_predictor(cimg, dlib::rectangle(l.x_left_upper,             // left
                                                      l.y_left_upper,             // top
                                                      l.x_left_upper + l.width-1, // right
                                                      l.y_left_upper + l.height-1 // bottom
                                                      ));
    cvPoint2fVec lm;
    for(size_t i=0; i<shape.num_parts(); i++){
      dlib::point pt = shape.part(i);                                          LOG4CXX_TRACE(_log, "lm[" << i << "]: " << shape.part(i));
      lm.push_back(cv::Point2f(pt.x(),pt.y()));
    }
    landmarks.push_back(lm);
  }

  return landmarks;
}

/** ****************************************************************************
* Get aligned thumbnails from detections using landmarks.   
* 
* \param         cfg       Config settings
* \param[in,out] locations detections to which features will be added    
* \param         bgrFrame  BGR color image in which detections were found
* \param         landmarks landmark points used for alignement
*
* \returns thumbnails for detections
*
***************************************************************************** */
cvMatVec OcvSsdFaceDetection::_getThumbnails(const JobConfig       &cfg,
                                             const cv::Mat         &bgrFrame,
                                             const cvPoint2fVecVec &landmarks){
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
  cvMatVec thumbnails;
  cv::Mat src = cv::Mat_<float>(3,2);
  
  for(auto &lm:landmarks){
    for(size_t r=0; r<3; r++){
      float* rowPtr = src.ptr<float>(r);
      rowPtr[0] = lm[lmIdx[r]].x; 
      rowPtr[1] = lm[lmIdx[r]].y;      
    }
    cv::Mat xfrm = cv::getAffineTransform(src,dst);
    cv::Mat thumbnail(THUMB_SIZE,bgrFrame.type());
    cv::warpAffine(bgrFrame,thumbnail,cv::getAffineTransform(src,dst),
                   THUMB_SIZE,cv::INTER_CUBIC,cv::BORDER_REPLICATE);
    thumbnails.push_back(thumbnail);
  }

  return thumbnails;
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
cvMatVec OcvSsdFaceDetection::_getFeatures(const JobConfig &cfg,
                                           const cvMatVec  &thumbnails){
  const double inScaleFactor = 1.0 / 255.0;
  const cv::Size blobSize(96, 96);
  const cv::Scalar meanVal(0.0, 0.0, 0.0);  // BGR mean pixel color 

  cvMatVec features;
  for(auto tmb:thumbnails){
    cv::Mat inputBlob = cv::dnn::blobFromImage(tmb,      // BGR image
                                              inScaleFactor, // no pixel value scaline (e.g. 1.0/255.0)
                                              blobSize,      // expected network input size: 300x300
                                              meanVal,       // mean BGR pixel value
                                              true,          // swap RB channels
                                              false);        // center crop
    _openFaceNet.setInput(inputBlob);
    cv::Mat feature = _openFaceNet.forward().clone();
    features.push_back(feature);
  }
  return features;

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
    if(job.data_uri.empty()) {                                                 LOG4CXX_ERROR(_log, "[" << job.job_name << "Invalid image url");
        return MPF_INVALID_DATAFILE_URI;
    }

    MPFImageReader imreader(job);
    cv::Mat img(imreader.GetImage());

    if(img.empty()){                                                           LOG4CXX_ERROR(_log, "[" << job.job_name << "] Could not read image file: " << job.data_uri);
      return MPF_IMAGE_READ_ERROR;
    }                                                                          LOG4CXX_DEBUG(_log, "[" << job.job_name << "] img.width  = " << img.cols);
                                                                               LOG4CXX_DEBUG(_log, "[" << job.job_name << "] img.height = " << img.rows);
    JobConfig cfg(job);
    size_t next_idx = locations.size();
    _detect(cfg, locations, img);
    size_t new_size = locations.size();                                        LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of faces detected = " << (new_size - next_idx));

    for(size_t i = next_idx; i < new_size; i++){                               LOG4CXX_TRACE(_log, "[" << job.job_name << "] location[" << i << "]" << locations[i]);
      imreader.ReverseTransform(locations[i]);
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
* Read frames from a video, get object detections and make tracks
*
* \param          job     MPF Video job
* \param[in,out]  tracks  Tracks collection to which detections will be added
*
* \returns  an MPF error constant or MPF_DETECTION_SUCCESS
*
***************************************************************************** */
MPFDetectionError OcvSsdFaceDetection::GetDetections(const MPFVideoJob &job, MPFVideoTrackVec &tracks) {

  try{                                                                         LOG4CXX_DEBUG(_log, "[" << job.job_name << "Data URI = " << job.data_uri);

    if(job.data_uri.empty()) {                                                 LOG4CXX_ERROR(_log, "[" << job.job_name << "Invalid video url");
        return MPF_INVALID_DATAFILE_URI;
    }

    MPFVideoCapture video_capture(job, true, true);
    if( !video_capture.IsOpened() ){                                           LOG4CXX_ERROR(_log, "[" << job.job_name << "] Could not initialize capturing");
      return MPF_COULD_NOT_OPEN_DATAFILE;
    }

    Properties jpr = job.job_properties;
    JobConfig cfg(job);
    size_t next_idx = tracks.size();

    cv::Mat frame;
    

    int frameIdx = 0;
    while (video_capture.Read(frame)) {
      MPFImageLocationVec locations;
      _detect(cfg, locations, frame);
      if(locations.size() > 0){
        if(tracks.size() == 0 ){
          tracks.push_back(MPFVideoTrack());
          tracks.back().start_frame = frameIdx;
        }
        for(auto &loc:locations){
          tracks.back().frame_locations.insert(pair<int,MPFImageLocation>(frameIdx,loc));
        }
        tracks.back().stop_frame = frameIdx;
      }
      frameIdx++;  
    }

    size_t new_size = tracks.size();                                           LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of tracks detected = " << (new_size - next_idx));
    for(size_t i = next_idx; i < new_size; i++){                               LOG4CXX_TRACE(_log, "[" << job.job_name << "] track[" << i << "]" << endl << tracks[i]);
      video_capture.ReverseTransform(tracks[i]);
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

/*
MPFDetectionError OcvSsdFaceDetection::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job,
        MPFVideoCapture &video_capture,
        vector<MPFVideoTrack> &tracks) {


    long total_frames = video_capture.GetFrameCount();
    LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Total video frames: " << total_frames);

    int frame_index = 0;

    Mat frame, frame_draw, frame_draw_pre_verified;
    //need to store the previous frame
    Mat gray, prev_gray;

    while (video_capture.Read(frame)) {

        //Convert to grayscale
        gray = Utils::ConvertToGray(frame);

        //look for new faces
        //vector <pair<Rect, float>> faces = ocv_detection.DetectFacesHaar(gray, min_face_size);
        RectScorePairVec faces = _detectorPtr->detectFaces(frame, min_face_size);


        int track_index = -1;
        for(auto track = current_tracks.begin(); track != current_tracks.end(); ++track) {
            ++track_index;
            //assume track lost at start
            track->track_lost = true;

            if (track->previous_points.empty()) {
                //should never get here!! - current points of first detection will be swapped to previous!!
                //kill the track if this case does occur
                LOG4CXX_TRACE(_log, "[" << job.job_name
                                                           << "] Track contains no previous points - killing tracks");
                continue;
            }

            vector <Point2f> new_points;
            vector <uchar> status;
            vector <float> err;
            //get new points
            calcOpticalFlowPyrLK(prev_gray, gray, track->previous_points,
                                 new_points, status, err); //, win_size, 3, term_criteria, 0, 0.001);

            //stores the detected rect properly matched up to the current track
            //if certain requirements are met
            pair<Rect, int> correct_detected_rect_pair;
            correct_detected_rect_pair.first = Rect(0, 0, 0, 0);

            //set to true if template matching is used to keep the track going
            //this will be used to determine if points can be redetected
            bool track_recovered = false;

            if (new_points.empty()) {
                //no new points - this track will be killed
                LOG4CXX_TRACE(_log, "[" << job.job_name
                                                           << "] Optical flow could not find any new points - killing track");
                continue;
            }else {
                //don't want to display any of the drawn points or bounding boxes of tracks that aren't kept
                frame_draw_pre_verified = frame_draw.clone();

                //store the correct bounding box here
                //Rect final_bounding_box(0,0,0,0);

                //GOAL: try to find the current detection with the most contained points
                //the goal is to use the opencv bounding box rather than estimate one using
                //an enclosed circle and a bounding rect
                //it will also be good to remove points not in this bounding box to improve the continued tracking
                //Rect correct_detected_rect(0,0,0,0);

                int points_within_detected_rect = 0;
                //store the percentage of intersection for each detected face
                map <int, float> rect_point_percentage_map;
                int face_rect_index = 0;
                for (vector <pair<Rect, float>>::iterator face_rect = faces.begin(); face_rect != faces.end(); ++face_rect) {
                    points_within_detected_rect = 0;

                    for (unsigned i = 0; i < new_points.size(); i++) { //TODO: could also use the err vector (from calcOpticalFlowPyrLK) with a float threshold
                        if (face_rect->first.contains(new_points[i])) {
                            ++points_within_detected_rect;
                        }
                    }

                    //if points were found inside one of the rects then it needs to be stored in the map
                    if (points_within_detected_rect > 0) {
                        rect_point_percentage_map[face_rect_index] = (static_cast<float>(points_within_detected_rect) /
                                                                      static_cast<float>(new_points.size()));
                    }

                    ++face_rect_index;
                }

                //TODO: could combine this with the loop above
                //also probably want the best intersection percentage to be > 75 or 80 percent
                // - if only a few points are intersecting then there is clearly an issue
                //float best_intersection_percentage = 0.0;
                //must be greater than 0.75
                float best_intersection_percentage = 0.75;
                for (map<int, float>::iterator map_iter = rect_point_percentage_map.begin();
                     map_iter != rect_point_percentage_map.end(); ++map_iter) {
                    float intersection_percentage(map_iter->second);
                    if (intersection_percentage > best_intersection_percentage) {
                        best_intersection_percentage = intersection_percentage;
                        correct_detected_rect_pair = faces[map_iter->first];
                    }
                }

                //if enters here means there is good point intersection with a detected rect
                if (correct_detected_rect_pair.first.area() > 0) {
                    //have to clear the old points first!
                    track->current_points.clear();
                    //only add new points if they are a '1' in the status vector and within the correctly detected rect!!
                    //TODO: could also use the err vector with a float threshold
                    for (unsigned i = 0; i < status.size(); i++) {
                        if (static_cast<int>(status[i])) {
                            //TODO: keep track of how many missed detections per track - think about stopping the track if not
                            //detecting the face for a few frames in a row
                            //only keep if within the correct detected face rect
                            if (correct_detected_rect_pair.first.contains(new_points[i])) {
                                track->current_points.push_back(new_points[i]);
                                circle(frame_draw_pre_verified, new_points[i], 2, Scalar(255, 255, 255), CV_FILLED);
                            }
                            else {
                                circle(frame_draw_pre_verified, new_points[i], 2, Scalar(0, 0, 255), CV_FILLED);
                            }
                        }
                    }
                }
                else {
                    //if turning the additional three displays on then make sure they are killed when finished tracking or detecting!
                    //Display("current frame at lost face", frame);

                    //have to clear the old points first! - points are still in the new points vector
                    track->current_points.clear();

                    //add all of the points
                    for (unsigned i = 0; i < new_points.size(); i++) {
                        track->current_points.push_back(new_points[i]);
                        //draw the points as red
                        circle(frame_draw_pre_verified, new_points[i], 2, Scalar(0, 0, 255), CV_FILLED);
                    }

                    //draw a circle around the points
                    Point2f center;
                    float radius;
                    minEnclosingCircle(track->current_points, center, radius);
                    //radius is increased
                    circle(frame_draw_pre_verified, center, radius * 1.2, Scalar(255, 255, 255), 1, 8);

                    //get a bound rect using the current points
                    Rect face_rect = boundingRect(track->current_points);
                    //check to see how small the bounding rect is
                    if (face_rect.height < 32) //face_rect.width < 32 ||
                    {
                        //if face smaller than 32 pixels then we don't want to keep it
                        LOG4CXX_TRACE(_log, "[" << job.job_name
                                                                   << "] Face too small to track - killing track");
                        continue;
                    }

                    //increase size since the size was decreased when detecting the points
                    Rect upscaled_face = GetUpscaledFaceRect(face_rect);
                    rectangle(frame_draw_pre_verified, face_rect, Scalar(255, 255, 0));

                    LOG4CXX_TRACE(_log, "[" << job.job_name << "] Getting template match");

                    //try template matching
                    //make sure the template is not out of bounds
                    AdjustRectToEdges(upscaled_face, gray);

                    //get first face - TODO: might not be the best face - should use the quality tool if available
                    //or could use the last face
                    Rect last_face_rect = Utils::ImageLocationToCvRect(track->face_track.frame_locations.rbegin()->second); // last element in map
                    Mat templ = prev_gray(last_face_rect);

                    //Display("last face", templ);

                    Rect match_rect = GetMatch(frame, gray, templ);
                    Mat new_frame_copy = frame.clone();

                    rectangle(new_frame_copy, match_rect, Scalar(255, 255, 255), 2);
                    //draw the previous face even though it was from the previous frame
                    rectangle(new_frame_copy, last_face_rect, Scalar(0, 255, 0), 2);

                    LOG4CXX_TRACE(_log, "[" << job.job_name << "] Match rect area: "
                                                               << match_rect.area());

                    Rect match_intersection = match_rect & last_face_rect; //opencv allows for this operation
                    rectangle(new_frame_copy, match_intersection, Scalar(255, 0, 0), 2);

                    //Display("intersection", new_frame_copy);

                    LOG4CXX_TRACE(_log, "[" << job.job_name << "] Finished getting match");

                    //look for a certain percentage of intersection
                    if (match_intersection.area() > 0) {
                        float intersection_rate = static_cast<float>(match_intersection.area()) /
                                                  static_cast<float>(last_face_rect.area());

                        LOG4CXX_TRACE(_log, "[" << job.job_name << "] Intersection rate: "
                                                                   << intersection_rate);

                        //was at 0.5 - should be much higher - don't want to be very permissive here - the template
                        //matching is not that good - TODO: need some sort of score could use openbr to do matching
                        if (intersection_rate < 0.7) {
                            continue;
                        }
                    }
                    else {
                        continue;
                    }

                    //NEED TO TRIM THE POINTS TO THE TEMPLATE MATCH RECT!!!
                    //some of the calc optical flow next points will start to get away!!
                    //TODO: make sure to set the actual rect to the template rect

                    //set the rescaled face that is used for the object detection to the template match
                    correct_detected_rect_pair.first = match_rect;

                    //set to true to make sure we don't redetect points in this case
                    track_recovered = true;

                    int points_within_template_match_rect = 0;
                    vector <Point2f> temp_points_copy(track->current_points);
                    //have to be cleared once again
                    track->current_points.clear();

                    //for(unsigned i=0; i < track->current_points.size(); i++)
                    for (unsigned i = 0; i < temp_points_copy.size(); i++) {
                        //USING the match intersection rect - might fix some issues with there is a transition in the video
                        //- might want to use the intserection for the actual face rect as well
                        if (match_rect.contains(temp_points_copy[i])) {
                            //TODO: need to check the count of these points - also need to kill the track if a bounding
                            //rect of these new points gets to be too small!!!
                            track->current_points.push_back(temp_points_copy[i]);
                            ++points_within_template_match_rect;

                        }
                    }
                }


                //TODO: does it seem necessary to do this again?
                //check corrected rect area again to check point percent
                if (correct_detected_rect_pair.first.area() > 0) {
                    //now can check the point percentage!
                    float current_point_percent = static_cast<float>(track->current_points.size()) /
                                                  static_cast<float>(track->init_point_count);

                    //TODO: probably need an intersection rate specific to the track continuation
                    //update current track info
                    track->current_point_count = static_cast<int>(track->current_points.size());
                    track->current_point_percent = current_point_percent;

                    if (current_point_percent < min_point_percent) {
                        //lost too many of original points - kill track
                        //set something to continue onto the feature matching portion - no reason to kill the track here
                        LOG4CXX_TRACE(_log, "[" << job.job_name
                                                                   << "] Lost too many points, current percent: " << current_point_percent);
                        continue;
                    }
                }
            }

            bool redetect_feature_points = true;

            //if a face bounding box is now set and we can redetect feature points
            //if track is not recovered
            if(correct_detected_rect_pair.first.area() > 0 && redetect_feature_points &&
               !track_recovered && track->current_point_percent < min_redetect_point_perecent) {
                LOG4CXX_TRACE(_log, "[" << job.job_name
                                                           << "] Attempting to redetect feature points");
                vector <KeyPoint> keypoints;
                Mat mask = GetMask(gray, correct_detected_rect_pair.first);
                //search for keypoints in the frame using a mask
                feature_detector->detect(gray, keypoints, mask);

                //calcOpticalFlowPyrLK uses float points - no need to store the KeyPoint vector - convert
                track->current_points.clear(); //why not clear before
                KeyPoint::convert(keypoints, track->current_points);

                //min init point count should be different for each detector!
                //TODO: not sure if I want to kill the track here - just don't update the init point count and continue
                if(keypoints.size() < min_init_point_count)
                {
                    LOG4CXX_TRACE(_log, "[" << job.job_name << "] Not enough initial points: " <<
                                                               static_cast<int>(track->current_points.size()));

                    //set the init to min init point count because we are now below that
                    track->init_point_count = min_init_point_count;
                    track->current_point_count = static_cast<int>(track->current_points.size());

                    //now need to re-check the percentage - TODO: put this in a member function
                    float current_point_percent = static_cast<float>(track->current_points.size()) /
                                                  static_cast<float>(track->init_point_count);
                    //set track info for display
                    track->current_point_count = static_cast<int>(track->current_points.size());
                    track->current_point_percent = current_point_percent;

                    if (current_point_percent < min_point_percent) {
                        //lost too many of original points - kill track
                        LOG4CXX_TRACE(_log, "[" << job.job_name
                                                                   << "] Lost too many points below min point percent, "
                                                                   << "current percent: " << current_point_percent);
                        continue;
                    }

                    LOG4CXX_TRACE(_log, "[" << job.job_name
                                                               << "] Keeping track below min_init_point_count with current percent: "
                                                               << current_point_percent);
                }
                else {
                    //reset the init and current point count also!
                    track->init_point_count = static_cast<int>(track->current_points.size());
                    track->current_point_count = track->init_point_count;
                }

                //must update the last detection index!
                track->last_face_detected_index = frame_index;
            }

            //at this point if the correct detected rect has an area we can keep the track
            if (correct_detected_rect_pair.first.area() > 0) {
                rectangle(frame_draw_pre_verified, correct_detected_rect_pair.first, Scalar(0, 255, 0), 2);

                //don't want to store a face that isn't within the bounds of the image
                AdjustRectToEdges(correct_detected_rect_pair.first, gray);
                MPFImageLocation fd = Utils::CvRectToImageLocation(correct_detected_rect_pair.first);

                fd.confidence = static_cast<float>(correct_detected_rect_pair.second);
                //can finally store the MPFImageLocation
                track->face_track.frame_locations.insert(pair<int, MPFImageLocation>(frame_index, fd));
                track->face_track.confidence = std::max(track->face_track.confidence, fd.confidence);
            }
            else {
                continue;
            }

            //if makes it here then we want to keep the track!!
            track->track_lost = false;
            //can also set the frame_draw to frame_draw_pre_verified Mat - TODO: should think of showing the pre verified Mat if the
            //track is lost
            frame_draw = frame_draw_pre_verified.clone();
        }

        //check if there is intersection between new objects and existing tracks
        //if not then add the new tracks
        for (unsigned i = 0; i < faces.size(); ++i) {
            int intersection_index = -1;
            if (!IsExistingTrackIntersection(faces[i].first, intersection_index)) {
                Track track_new;

                //set face detection
                Rect face(faces[i].first);
                AdjustRectToEdges(face, gray);

                float first_face_confidence = static_cast<float>(faces[i].second);

                bool use_face = false;
                if (first_face_confidence > min_initial_confidence) {
                    use_face = true;
                }
                else {

                    LOG4CXX_TRACE(_log, "[" << job.job_name
                                                               << "] Detected face does not meet initial quality: " << first_face_confidence);
                }

                if (use_face) {
                    //if the face meets quality or we don't care about quality
                    //the keypoints can now be detected

                    vector <KeyPoint> keypoints;
                    Mat mask = GetMask(gray, faces[i].first);
                    //search for keypoints in the frame using a mask
                    feature_detector->detect(gray, keypoints, mask);

                    //min init point count should be different for each detector!
                    if(keypoints.size() < min_init_point_count)
                    {
                        LOG4CXX_TRACE(_log, "[" << job.job_name << "] Not enough initial points: "
                                                                   << static_cast<int>(keypoints.size()));

                        continue;
                    }

                    //set first keypoints and mat
                    track_new.first_detected_keypoints = keypoints;
                    track_new.first_gray_frame = gray.clone();

                    //calcOpticalFlowPyrLK uses float points - no need to store the KeyPoint vector - convert
                    KeyPoint::convert(keypoints, track_new.current_points);

                    //set start frame and initial point count
                    track_new.face_track.start_frame = frame_index;
                    //set first face detection index
                    track_new.last_face_detected_index = frame_index;
                    track_new.init_point_count = static_cast<int>(track_new.current_points.size());

                    //set face detection
                    Rect face(faces[i].first);
                    AdjustRectToEdges(face, gray);
                    MPFImageLocation first_face_detection(face.x, face.y, face.width, face.height);
                    //first_face_confidence is already a float value
                    first_face_detection.confidence = first_face_confidence;

                    //add the first detection
                    track_new.face_track.frame_locations.insert(pair<int, MPFImageLocation>(frame_index, first_face_detection));
                    track_new.face_track.confidence = std::max(track_new.face_track.confidence,
                                                               first_face_detection.confidence);
                    //add the new track
                    current_tracks.push_back(track_new);

                    LOG4CXX_TRACE(_log, "[" << job.job_name << "] Creating new track");
                }


            }

        }

        vector <Track> tracks_to_keep;
        for(vector<Track>::iterator track = current_tracks.begin(); track != current_tracks.end(); ++track)
        {
            if(track->track_lost)
            {
                LOG4CXX_TRACE(_log, "[" << job.job_name << "] Killing track");

                //did not pass the rules to continue this frame_index, it ended on the previous index
                track->face_track.stop_frame = frame_index - 1;

                //only saving tracks lasting more than 1 frame to eliminate badly started tracks
                if (track->face_track.stop_frame - track->face_track.start_frame > 1) {
                    saved_tracks.push_back(*track);
                }
            }
            else {
                tracks_to_keep.push_back(*track);
            }
        }

        //now clear the current tracks and add the tracks to keep
        current_tracks.clear();
        for (vector<Track>::iterator track = tracks_to_keep.begin(); track != tracks_to_keep.end(); track++) {
            current_tracks.push_back(*track);
        }

        //set previous frame
        prev_gray = gray.clone();
        //swap points
        for (vector<Track>::iterator it = current_tracks.begin(); it != current_tracks.end(); it++) {
            swap(it->current_points, it->previous_points);
        }

        ++frame_index;
    }

    CloseAnyOpenTracks(video_capture.GetFrameCount() - 1);

    //set tracks reference!
    for (unsigned int i = 0; i < saved_tracks.size(); i++) {
        tracks.push_back(saved_tracks[i].face_track);
    }

    //clear any internal structures that could carry over before the destructor is called
    //these can be cleared - the data has been moved to tracks
    current_tracks.clear();
    saved_tracks.clear();

    LOG4CXX_INFO(_log, "[" << job.job_name << "] Processing complete. Found "
                                              << static_cast<int>(tracks.size()) << " tracks.");

    if (verbosity > 0) {
        //now print tracks if available
        if(!tracks.empty())
        {
            for(unsigned int i=0; i<tracks.size(); i++)
            {
                LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Track index: " << i);
                LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Track start index: " << tracks[i] .start_frame);
                LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Track end index: " << tracks[i] .stop_frame);

                for (map<int, MPFImageLocation>::const_iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it)
                {
                    LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Frame num: " << it->first);
                    LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Bounding rect: (" << it->second .x_left_upper << ","
                                                               <<  it->second.y_left_upper << "," << it->second.width << "," << it->second.height << ")");
                    LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Confidence: " << it->second.confidence);
                }
            }
        }
        else
        {
            LOG4CXX_DEBUG(_log, "[" << job.job_name << "] No tracks found");
        }
    }

    return MPF_DETECTION_SUCCESS;
}


void OcvSsdFaceDetection::LogDetection(const MPFImageLocation& face, const string& job_name){
    LOG4CXX_DEBUG(_log, "[" << job_name << "] XLeftUpper: " << face.x_left_upper);
    LOG4CXX_DEBUG(_log, "[" << job_name << "] YLeftUpper: " << face.y_left_upper);
    LOG4CXX_DEBUG(_log, "[" << job_name << "] Width:      " << face.width);
    LOG4CXX_DEBUG(_log, "[" << job_name << "] Height:     " << face.height);
    LOG4CXX_DEBUG(_log, "[" << job_name << "] Confidence: " << face.confidence);
}
*/

MPF_COMPONENT_CREATOR(OcvSsdFaceDetection);
MPF_COMPONENT_DELETER();
