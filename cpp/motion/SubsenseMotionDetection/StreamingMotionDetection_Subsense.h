/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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


#ifndef OPENMPF_CONTRIB_COMPONENTS_MOTION_SUBSENSE_STREAMING_DETECTION_H
#define OPENMPF_CONTRIB_COMPONENTS_MOTION_SUBSENSE_STREAMING_DETECTION_H

#include <map>
#include <string>

#include <opencv2/opencv.hpp>

#include <log4cxx/logger.h>

#include "SubSense/BackgroundSubtractorSuBSENSE.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include <struck.h>
#pragma GCC diagnostic pop

#include <MPFStreamingDetectionComponent.h>

#include "SubsenseUtils.h"

/*! \class SubsenseStreamingDetection
 *  \brief motion detection/tracking, based on the SubSENSE background
 *  subtraction algorithm, for streaming video jobs
 */
class SubsenseStreamingDetection : public MPF::COMPONENT::MPFStreamingDetectionComponent {
  public:
    explicit SubsenseStreamingDetection(const MPF::COMPONENT::MPFStreamingVideoJob &job);

    void BeginSegment(const MPF::COMPONENT::VideoSegmentInfo &segment_info) override;
    bool ProcessFrame(const cv::Mat &frame, int frame_number) override;
    std::vector<MPF::COMPONENT::MPFVideoTrack> EndSegment() override;

  private:
    SubsenseStreamingDetection(const MPF::COMPONENT::MPFStreamingVideoJob &job,
                               const SubsenseConfig &config);
    SubsenseConfig config_;
    std::string job_name_;
    std::string msg_prefix_;
    log4cxx::LoggerPtr motion_logger_;
    // Set to true when activity has been reported in a segment, so that this is done at most once.
    bool segment_activity_reported_ = false;
    int frame_width_ = -1;
    int frame_height_ = -1;
    int previous_segment_number_ = -1;
    int current_segment_number_ = -1;
    int downsample_count_ = 0;
    int tracker_id_ = 0;

    BackgroundSubtractorSuBSENSE bg_;
    bool bg_initialized_ = false;

    // Each entry in the following map is indexed by the segment frame
    // index, and contains the actual frame number. This map is used
    // in the EndSegment() method to set the track frame indices to
    // the frame numbers provided to the component in ProcessFrame().

    std::map<int, int> frame_number_map_;
    int segment_frame_index_ = -1;  // Counter used to index the frame number map.

    MPF::COMPONENT::MPFVideoTrack preprocessor_track_;
    std::vector<MPF::COMPONENT::MPFVideoTrack> tracks_;
    std::map<int, STRUCK> tracker_map_;
    std::map<int, MPF::COMPONENT::MPFVideoTrack> track_map_;
};


#endif //OPENMPF_CONTRIB_COMPONENTS_MOTION_SUBSENSE_STREAMING_DETECTION_H
