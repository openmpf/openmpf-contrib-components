
#ifndef OCVSSDFACEDETECTION_JOBCONFIG_H
#define OCVSSDFACEDETECTION_JOBCONFIG_H

#include <log4cxx/logger.h>

#include "detectionComponentUtils.h"
#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"
#include "MPFImageReader.h"
#include "MPFVideoCapture.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;


  /** ****************************************************************************
  *   get MPF properies of various types
  ***************************************************************************** */
  template<typename T>
  T get(const Properties &p, const string &k, const T def){
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }

  /** ****************************************************************************
  *   get configuration from environment variables if not
  *   provided by job configuration
  ***************************************************************************** */
  template<typename T>
  T getEnv(const Properties &p, const string &k, const T def){
    auto iter = p.find(k);
    if (iter == p.end()) {                       
      const char* env_p = getenv(k.c_str());
      if(env_p != NULL){                          
        map<string,string> envp;
        envp.insert(pair<string,string>(k,string(env_p)));
        return DetectionComponentUtils::GetProperty<T>(envp,k,def);
      }else{                                      
        return def;                            
      }
    }
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }

  /** ****************************************************************************
  *   exception macro so we can see where in the code it happened
  ****************************************************************************** */
  #define THROW_EXCEPTION(MSG){                                  \
  string path(__FILE__);                                         \
  string f(path.substr(path.find_last_of("/\\") + 1));           \
  throw runtime_error(f + "[" + to_string(__LINE__)+"] " + MSG); \
  }

  /* **************************************************************************
  *   Configuration parameters populated with appropriate values & defaults
  *************************************************************************** */
  class JobConfig{
    public:
      static log4cxx::LoggerPtr _log;  ///< shared log opbject
      size_t minDetectionSize;         ///< minimum boounding box dimension
      float  confThresh;               ///< detection confidence threshold
      long   detFrameInterval;         ///< number of frames between looking for new detection (tracking only)    

      float  maxFeatureDist;           ///< maximum feature distance to maintain track continuity
      float  maxCenterDist;            ///< maximum spatial distance normalized by diagonal to maintain track continuity
      long   maxFrameGap;              ///< maximum temporal distance (frames) to maintain track continuity
      float  maxIOUDist;               ///< maximum for (1 - Intersection/Union) to maintain track continuity

      float   widthOdiag;              ///< image (width/diagonal)
      float   heightOdiag;             ///< image (height/diagonal)
      size_t  frameIdx;                ///< index of current frame
      cv::Mat bgrFrame;                ///< current BGR image frame

      MPFDetectionError   lastError;   ///< last MPF error that should be returned    

      JobConfig();
      JobConfig(const MPFImageJob &job);
      JobConfig(const MPFVideoJob &job);
      ~JobConfig();

      void ReverseTransform(MPFImageLocation loc){_imreaderPtr->ReverseTransform(loc);}
      void ReverseTransform(MPFVideoTrack  track){_videocapPtr->ReverseTransform(track);}
      bool nextFrame();

    private:
      unique_ptr<MPFImageReader>  _imreaderPtr;
      unique_ptr<MPFVideoCapture> _videocapPtr;

      void _parse(const MPFJob &job);


  };
  
  ostream& operator<< (ostream& out, const JobConfig& cfg);  ///< Dump JobConfig to a stream

 }
}

#endif