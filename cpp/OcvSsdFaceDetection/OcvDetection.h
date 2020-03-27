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


#ifndef OPENMPF_COMPONENTS_OCVDETECTION_H
#define OPENMPF_COMPONENTS_OCVDETECTION_H

#include <string>
#include <vector>
#include <utility>

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

namespace MPF{
 namespace COMPONENT{

    using namespace std;

    typedef vector<MPFVideoTrack>    MPFVideoTrackVec;     ///< vector of MPFViseoTracks
    typedef vector<MPFImageLocation> MPFImageLocationVec;  ///< vector of MPFImageLocations

    /** ****************************************************************************
    * Macro for throwing exception so we can see where in the code it happened
    ****************************************************************************** */
    #define THROW_EXCEPTION(MSG){                                  \
    string path(__FILE__);                                         \
    string f(path.substr(path.find_last_of("/\\") + 1));           \
    throw runtime_error(f + "[" + to_string(__LINE__)+"] " + MSG); \
    }


    /** *********************************************************************** 
     * Class to encapsulate the SSD detector
    ************************************************************************* */
    class OcvDetection {
      public:
        OcvDetection(const string &plugin_path);

        MPFImageLocationVec detect(const cv::Mat &bgrFrame,
                                   const int      minBBoxSide = 48,
                                   const float    confThresh  = 0.65
                                   );                            ///< get bboxes with conf. scores 

      private:
        log4cxx::LoggerPtr _log;                                 ///< log object
        string             _pluginPath;                          ///< mpf plugin base path 
        cv::dnn::Net       _ssdNet;                              ///< single shot DNN face detector network

    };

 }
}
#endif //OPENMPF_COMPONENTS_OCVDETECTION_H
