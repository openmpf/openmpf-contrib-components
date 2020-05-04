/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
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

#include <log4cxx/xml/domconfigurator.h>

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QString>

// 3rd party code for solving assignment problem
#include "munkres.h"

// MPF-SDK header files 
#include "Utils.h"
#include "MPFSimpleConfigLoader.h"
#include "detectionComponentUtils.h"


using namespace MPF::COMPONENT;

// Temporary initializer for static member variable
log4cxx::LoggerPtr JobConfig::_log = log4cxx::Logger::getRootLogger();

/** ****************************************************************************
 *  print out opencv matrix on a single line
 * 
 * \param   m matric to serialize to single line string
 * \returns single line string representation of matrix
 * 
***************************************************************************** */
string format(cv::Mat m){
  stringstream ss;
  ss << m;
  string str = ss.str();
  str.erase(remove(str.begin(),str.end(),'\n'),str.end());
  return str;
}

/** ****************************************************************************
*  Initialize SSD face detector module by setting up paths and reading configs
*  configuration variables are turned into environment variables for late
*  reference
*
* \returns   true on success
***************************************************************************** */
bool OcvSsdFaceDetection::Init() {
    string plugin_path    = GetRunDirectory() + "/OcvSsdFaceDetection";
    string config_path    = plugin_path       + "/config";

    log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
    _log = log4cxx::Logger::getLogger("OcvSsdFaceDetection");                  LOG4CXX_DEBUG(_log,"Initializing OcvSSDFaceDetector");
    JobConfig::_log = _log;                                                  //LOG4CXX_TRACE(_log, cv::getBuildInformation() << std::endl);

    // read config file and create or update any missing env variables
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

    return DetectionLocation::Init(_log, plugin_path);
    
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
*   Compute cost matrix and solve it to give detection to track assignment matrix
*
* \param COST_FUNC     cost function to use 
* \param TrackPtrList  list of existing tracks to consider for assignment
* \param detections    vector of detections that need assigned to tracks
* \param maxCost       maximum assignment cost, if exeeded the particular 
*                      detection to track assignemnt will be removed from result
* \returns assignment matrix am[track,detection] with dim (# tracks x # detections)
*          if am[x,y]==0 then detection[y] should be assigned to track[x]
* 
***************************************************************************** */
template<DetectionLocationCostFunc COST_FUNC>
cv::Mat_<int> OcvSsdFaceDetection::_calcAssignemntMatrix(const TrackPtrList                &tracks,
                                                         const DetectionLocationPtrVec     &detections,
                                                         const float                        maxCost = INT_MAX){
  // rows -> tracks, cols -> detections
  cv::Mat_<int> costs(tracks.size(),detections.size());
  size_t r = 0;
  for(auto &track:tracks){
    int* row = costs.ptr<int>(r);
    for(size_t c=0; c<costs.cols; c++){
      if(track->back()->frameIdx < detections[c]->frameIdx){
        float cost = CALL_MEMBER_FUNC(*(track->back()),COST_FUNC)(*detections[c]);
        row[c] = (cost <= maxCost) ? 1000 * cost : INT_MAX;
      }else{
        row[c] = INT_MAX;
      }
    }
    r++;
  }                                                                           LOG4CXX_TRACE(_log,"Cost Matrix[tr=" << costs.rows << ",det=" << costs.cols << "]: " << format(costs));

  Munkres assignmentSolver;  
  cv::Mat_<int> am = costs.clone();
  assignmentSolver.solve(am);                                                 //LOG4CXX_TRACE(_log,"Solved Matrix: " << format(am));                                       

  for(int rr=0; rr<costs.rows; rr++){                                         
    int* cost_row = costs.ptr<int>(rr);
    int*   am_row = am.ptr<int>(rr);
    for(int cc=0; cc<costs.cols; cc++){
      if(cost_row[cc] == INT_MAX) am_row[cc] = -1;                            // knock out assignments that are too costly (i.e. new track needed)
    }
  }
                                                                              LOG4CXX_TRACE(_log,"Assignment Matrix: " << format(am));
  return am;
}

/** ****************************************************************************
*   Move detections to tails of tracks according to assignment matrix
*
* \param [in,out] tracks           list of tracks
* \param [in,out] detections       vector of detections
* \param          assignmentMatrix (tracks x detections) matrix with 0 in 
*                                  position corresponding to an assignment
*
* \note detections that are assigned are be removed from the detections vector
*
***************************************************************************** */
void OcvSsdFaceDetection::_assignDetections2Tracks(TrackPtrList             &tracks,
                                                   DetectionLocationPtrVec  &detections,
                                                   const cv::Mat_<int>      &assignmentMatrix){
  size_t r=0;                                                   
  for(auto &track:tracks){
    const int* rowPtr = assignmentMatrix.ptr<int>(r);
    for(size_t c=0; c<assignmentMatrix.cols; c++){
      if(rowPtr[c] == 0){                                                    LOG4CXX_TRACE(_log,"assigning det: " << "f" << detections[c]->frameIdx << " " <<  ((MPFImageLocation)*(detections[c])) << "to track " << *track);
        track->back()->releaseBGRFrame();                                    // no longer need previous frame data.
        track->push_back(move(detections[c]));                                 
      }
    }
    r++;
  }

  // remove detections that were assigned to tracks
  detections.erase(remove_if(detections.begin(),
                             detections.end(),
                             [](unique_ptr<DetectionLocation> const& d){return !d;}),
                   detections.end());

}

/** ****************************************************************************
* Read an image and get object detections and features
*
* \param          job     MPF Image job
* \param[in,out]  locations  locations collection to which detections will be added
*
* \returns MPF error constant or MPF_DETECTION_SUCCESS
*
***************************************************************************** */
MPFDetectionError OcvSsdFaceDetection::GetDetections(const MPFImageJob   &job,
                                                     MPFImageLocationVec &locations) {

  try {                                                                        LOG4CXX_DEBUG(_log, "[" << job.job_name << "Data URI = " << job.data_uri);
    JobConfig cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS) return cfg.lastError;

    DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfg);
                                                                               LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of faces detected = " << detections.size());
    for(auto &det:detections){
      MPFImageLocation loc = *det;
      det.reset();                                                             // release frame object                                                      
      cfg.ReverseTransform(loc);
      locations.push_back(loc);
    }

  }catch(const runtime_error& re){                                             
    LOG4CXX_FATAL(_log, "[" << job.job_name << "] runtime error: " << re.what());
    return Utils::HandleDetectionException(job, _log);
  }catch(const exception& ex){                                                 
    LOG4CXX_FATAL(_log, "[" << job.job_name << "] exception: " << ex.what());
    return Utils::HandleDetectionException(job, _log);
  }catch (...) {                                                               
    LOG4CXX_FATAL(_log, "[" << job.job_name << "] unknown error");
    return Utils::HandleDetectionException(job, _log);                         
  }
                                                                               LOG4CXX_DEBUG(_log,"[" << job.job_name << "] complete.");
  return MPF_DETECTION_SUCCESS;
}

/** ****************************************************************************
* Convert track (list of detection ptrs) to an MPFVideoTrack object
*
* \param[in,out]  tracks  Tracks collection to which detections will be added
*                         
* \returns MPFVideoTrack object resulting from conversion
* 
* \note detection pts are released on conversion and confidence is assigned
*       as the average of the detection confidences
*
***************************************************************************** */
MPFVideoTrack OcvSsdFaceDetection::_convert_track(Track &track){

  MPFVideoTrack mpf_track;
  
  mpf_track.start_frame = track.front()->frameIdx;
  mpf_track.stop_frame  = track.back()->frameIdx;
  
  stringstream start_feature;
  stringstream stop_feature;
  start_feature << track.front()->getFeature();  // make sure we have computed features to serialize
  stop_feature << track.back()->getFeature();    //  for the start and end detections.
  mpf_track.detection_properties["START_FEATURE"] = start_feature.str();
  mpf_track.detection_properties["STOP_FEATURE"]  = stop_feature.str();

  for(auto &det:track){
    mpf_track.confidence += det->confidence;                                                    
    mpf_track.frame_locations.insert(mpf_track.frame_locations.end(),{det->frameIdx,*det});
    det.reset();
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
* \returns   MPF_DETECTION_SUCCESS or another MPF error constant on failure
*
***************************************************************************** */
MPFDetectionError OcvSsdFaceDetection::GetDetections(const MPFVideoJob &job,
                                                     MPFVideoTrackVec  &mpf_tracks) {

  try{ 

    TrackPtrList trackPtrs;

    JobConfig cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS) return cfg.lastError;

    size_t detectTrigger = 0;                                                                           
    while(cfg.nextFrame()) {                                                   LOG4CXX_TRACE(_log, ".");
                                                                               LOG4CXX_TRACE(_log, "processing frame " << cfg.frameIdx);                                                                             
      // remove any tracks too far in the past
      trackPtrs.remove_if([&](unique_ptr<Track>& tPtr){                      
        if(cfg.frameIdx - tPtr->back()->frameIdx > cfg.maxFrameGap){           LOG4CXX_TRACE(_log,"droping old track: " << *tPtr);
          mpf_tracks.push_back(_convert_track(*tPtr));                       
          return true;
        }
        return false;
      });

      if(detectTrigger == 0){                                                  LOG4CXX_TRACE(_log,"checking for new detections");
        DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfg); // look for new detections
      
        if(detections.size() > 0){   // found some detections in current frame
          if(trackPtrs.size() >= 0 ){  // not all tracks were dropped  
                                                                               LOG4CXX_TRACE(_log, detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");
            // intersection over union tracking and assignment
            cv::Mat_<int> am = _calcAssignemntMatrix<&DetectionLocation::iouDist>(trackPtrs,detections,cfg.maxIOUDist);     
            _assignDetections2Tracks(trackPtrs, detections, am);               LOG4CXX_TRACE(_log,"IOU assignment complete"); 

            // feature-based tracking tracking and assignment
            if(detections.size() > 0){                                         LOG4CXX_TRACE(_log, detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");  
              am = _calcAssignemntMatrix<&DetectionLocation::featureDist>(trackPtrs,detections,cfg.maxFeatureDist);
              _assignDetections2Tracks(trackPtrs, detections, am);             LOG4CXX_TRACE(_log,"Feature assignment complete");
            }

            // center-to-center distance tracking and assignemnt
            if(detections.size() > 0){                                         LOG4CXX_TRACE(_log, detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks"); 
              am = _calcAssignemntMatrix<&DetectionLocation::center2CenterDist>(trackPtrs,detections,cfg.maxCenterDist);
              _assignDetections2Tracks(trackPtrs, detections, am);             LOG4CXX_TRACE(_log,"Center2Center assignment complete");
            }

          }
                                                                               LOG4CXX_TRACE(_log, detections.size() <<" detections left for new tracks");
          // any detection not assigned up to this point becomes a new track
          for(auto &det:detections){                                           // make any unassigned detections into new tracks
              trackPtrs.push_back(unique_ptr<Track>(new Track()));             // create a new empty track
              det->getFeature();                                               // start of tracks always get feature calculated
              det->releaseBGRFrame();                                          // once features are calculated, won't need frame data any more.
              trackPtrs.back()->push_back(move(det));                          LOG4CXX_TRACE(_log,"created new track " << *(trackPtrs.back()));
          }
        }
      }

      // check any tracks that didn't get a detection and use tracker to continue them if possible
      for(auto &trackPtr:trackPtrs){
        if(trackPtr->back()->frameIdx < cfg.frameIdx){  // no detections for track in current frame, try tracking
          unique_ptr<DetectionLocation> detPtr = trackPtr->back()->ocvTrackerPredict(cfg);
          if(detPtr){  // tracker returned something
            trackPtr->back()->releaseBGRFrame();
            trackPtr->push_back(move(detPtr));
          }
        }
      }

      detectTrigger++;
      detectTrigger = detectTrigger % (cfg.detFrameInterval + 1);

    }                                                                          LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of tracks detected = " << trackPtrs.size());

    // convert any remaining active tracks to MPFVideoTracks
    for(auto &trackPtr:trackPtrs){
      mpf_tracks.push_back(_convert_track(*trackPtr));
    }

    // reverse transform all mpf tracks
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
