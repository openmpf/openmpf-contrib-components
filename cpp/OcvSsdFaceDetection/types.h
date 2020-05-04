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

#ifndef OCVSSDFACEDETECTION_TYPES_H
#define OCVSSDFACEDETECTION_TYPES_H

#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  typedef vector<MPFVideoTrack>    MPFVideoTrackVec;     ///< vector of MPFVideoTracks
  typedef vector<MPFImageLocation> MPFImageLocationVec;  ///< vector of MPFImageLocations
  typedef vector<cv::Mat>          cvMatVec;             ///< vector of OpenCV matricie/images
  typedef vector<cv::Rect>         cvRectVec;            ///< vector of OpenCV rectangles
  typedef vector<cv::Point>        cvPointVec;           ///< vector of OpenCV points     
  typedef vector<cv::Point2f>      cvPoint2fVec;         ///< vector of OpenCV 2D float points
  typedef vector<cvPoint2fVec>     cvPoint2fVecVec;      ///< vector of vectors of OpenCV 2D float points

  /** **************************************************************************
  *   Dump MPFLocation to a stream
  *************************************************************************** */ 
  inline
  ostream& operator<< (ostream& out, const MPFImageLocation& l) {
    out << "[" << l.x_left_upper << "," << l.y_left_upper << "]-("
               << l.width        << "," << l.height       << "):"
               << l.confidence   << " "
              //<< l.detection_properties.at("LANDMARKS")
               ;
    return out;
  }

  /** **************************************************************************
  *   Dump MPFTrack to a stream
  *************************************************************************** */ 
  inline
  ostream& operator<< (ostream& out, const MPFVideoTrack& t) {
    out << t.start_frame << endl;
    out << t.stop_frame  << endl;
    for(auto& p:t.frame_locations){
      out << p.second.x_left_upper << "," << p.second.y_left_upper << ","
          << p.second.width << "," << p.second.height << endl;
    }
    return out;
  }

  /** ****************************************************************************
  *   Dump vectors to a stream
  ***************************************************************************** */
  template<typename T>
  ostream& operator<< (ostream& out, const vector<T>& v) {
    out << "{";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i){
      out << v[i];
      if(i != last) out << ", ";
    }
    out << "}";
    return out;
  }

 }
}



#endif