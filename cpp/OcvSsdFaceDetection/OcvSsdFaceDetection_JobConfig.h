
#ifndef OCVFACEDETECTION_JOBCONFIG_H
#define OCVFACEDETECTION_JOBCONFIG_H

#include <log4cxx/logger.h>

#include "detectionComponentUtils.h"
#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;


  /** ****************************************************************************
  * shorthands for getting MPF properies of various types
  ***************************************************************************** */
  template<typename T>
  T get(const Properties &p, const string &k, const T def){
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }

  /** ****************************************************************************
  * shorthands for getting configuration from environment variables if not
  * provided by job configuration
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
  * Macro for throwing exception so we can see where in the code it happened
  ****************************************************************************** */
  #define THROW_EXCEPTION(MSG){                                  \
  string path(__FILE__);                                         \
  string f(path.substr(path.find_last_of("/\\") + 1));           \
  throw runtime_error(f + "[" + to_string(__LINE__)+"] " + MSG); \
  }

  /* **************************************************************************
  * Configuration parameters populated with appropriate values / defaults
  *************************************************************************** */
  class JobConfig{
    public:
      static log4cxx::LoggerPtr _log;  ///< shared log opbject
      size_t minDetectionSize;         ///< minimum boounding box dimension
      float  confThresh;               ///< detection confidence threshold
      JobConfig();
      JobConfig(const MPFJob &job);
  };
  
 }
}

#endif