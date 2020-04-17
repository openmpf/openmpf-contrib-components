
#include "JobConfig.h"

using namespace MPF::COMPONENT;

void JobConfig::_parse(const MPFJob &job){
  const Properties jpr = job.job_properties;
  minDetectionSize = getEnv<int>  (jpr,"MIN_DETECTION_SIZE",             minDetectionSize);  LOG4CXX_TRACE(_log, "MIN_DETECTION_SIZE: " << minDetectionSize);
  confThresh       = getEnv<float>(jpr,"DETECTION_CONFIDENCE_THRESHOLD", confThresh);        LOG4CXX_TRACE(_log, "DETECTION_CONFIDENCE_THRESHOLD: " << confThresh);
  
  featureWeight    = getEnv<float>(jpr,"TRACKING_FEATURE_WEIGHT",        featureWeight);     LOG4CXX_TRACE(_log, "TRACKING_FEATURE_WEIGHT: " << featureWeight);
  centerDistWeight = getEnv<float>(jpr,"TRACKING_CENTER_DIST_WEIGHT",    centerDistWeight);  LOG4CXX_TRACE(_log, "TRACKING_CENTER_DIST_WEIGHT: " << centerDistWeight);
  frameGapWeight   = getEnv<float>(jpr,"TRACKING_FRAME_GAP_WEIGHT",      frameGapWeight);    LOG4CXX_TRACE(_log, "TRACKING_FRAME_GAP_WEIGHT: " << frameGapWeight);
  iouWeight        = getEnv<float>(jpr,"TRACKING_IOU_WEIGHT",            iouWeight);         LOG4CXX_TRACE(_log, "TRACKING_IOU_WEIGHT: " << iouWeight);

  maxFeatureDist   = getEnv<float>(jpr,"TRACKING_MAX_FEATURE_DIST",      maxFeatureDist);    LOG4CXX_TRACE(_log, "TRACKING_MAX_FEATURE_DIST: " << maxFeatureDist);
  maxFrameGap      = getEnv<int>  (jpr,"TRACKING_MAX_FRAME_GAP",         maxFrameGap);       LOG4CXX_TRACE(_log, "TRACKING_MAX_FRAME_GAP: " << maxFrameGap);
  maxCenterDist    = getEnv<float>(jpr,"TRACKING_MAX_CENTER_DIST",       maxCenterDist);     LOG4CXX_TRACE(_log, "TRACKING_MAX_CENTER_DIST: " << maxCenterDist);
  maxIOUDist       = getEnv<float>(jpr,"TRACKING_MAX_IOU_DIST",          maxIOUDist);        LOG4CXX_TRACE(_log, "TRACKING_MAX_IOU_DIST: " << maxIOUDist);

  float weightsum = featureWeight + centerDistWeight + frameGapWeight + iouWeight;
  iouWeight        /= weightsum;
  featureWeight    /= weightsum;
  centerDistWeight /= weightsum;
  frameGapWeight   /= weightsum;
}

/* **************************************************************************
* Default constructor
*************************************************************************** */ 
ostream& operator<< (ostream& out, const JobConfig& cfg) {
  out << "{"
      << "\"minDetectionSize\": " << cfg.minDetectionSize 
      << "\"confThresh\":" << cfg.confThresh 
      << "}";
  return out;
}

/* **************************************************************************
* Default constructor
*************************************************************************** */ 
JobConfig::JobConfig():
  minDetectionSize(45),
  confThresh(0.65),
  featureWeight(0.5),
  centerDistWeight(0.25),
  frameGapWeight(0.25),
  maxFeatureDist(0.5),
  maxCenterDist(0.5),
  maxFrameGap(30),
  frameIdx(-1),
  lastError(MPF_DETECTION_SUCCESS),
  _imreaderPtr(NULL),
  _videocapPtr(NULL){}                    

/* **************************************************************************
* ImageJob constructor
*************************************************************************** */ 
JobConfig::JobConfig(const MPFImageJob &job):
  JobConfig() {
                                                                               LOG4CXX_DEBUG(_log, "[" << job.job_name << "Data URI = " << job.data_uri);
  _parse(job);
  
  if(job.data_uri.empty()) {                                                   LOG4CXX_ERROR(_log, "[" << job.job_name << "Invalid image url");
    lastError = MPF_INVALID_DATAFILE_URI;
  }else{
    _imreaderPtr = new MPFImageReader(job);
    bgrFrame = _imreaderPtr->GetImage();
    if(bgrFrame.empty()){                                                      LOG4CXX_ERROR(_log, "[" << job.job_name << "] Could not read image file: " << job.data_uri);
      lastError = MPF_IMAGE_READ_ERROR;
    }                                                                          LOG4CXX_DEBUG(_log, "[" << job.job_name << "] image.width  = " << bgrFrame.cols);
                                                                               LOG4CXX_DEBUG(_log, "[" << job.job_name << "] image.height = " << bgrFrame.rows);

  }
}

/* **************************************************************************
* VideoJob constructor
*************************************************************************** */ 
JobConfig::JobConfig(const MPFVideoJob &job):
  JobConfig() {

  _parse(job);

  if(job.data_uri.empty()) {                                                   LOG4CXX_ERROR(_log, "[" << job.job_name << "Invalid video url");
    lastError = MPF_INVALID_DATAFILE_URI;
  }else{
    _videocapPtr = new MPFVideoCapture(job, true, true);
    if(!_videocapPtr->IsOpened()){                                             LOG4CXX_ERROR(_log, "[" << job.job_name << "] Could not initialize capturing");
      lastError = MPF_COULD_NOT_OPEN_DATAFILE;
    }
    // pre-compute diagonal normalization factor for distance normalizations
    cv::Size fs = _videocapPtr->GetFrameSize();
    float diag  = sqrt(fs.width*fs.width + fs.height*fs.height); 
    widthOdiag  = fs.width  / diag; 
    heightOdiag = fs.height / diag;
  }
}

/* **************************************************************************
* Destructor
*************************************************************************** */ 
JobConfig::~JobConfig(){
  if(_imreaderPtr != NULL){
    delete _imreaderPtr;
  }
  if(_videocapPtr != NULL){
    _videocapPtr->Release();
    delete _videocapPtr;
  }
}