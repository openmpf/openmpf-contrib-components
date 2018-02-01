/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

#include <string>

#include <opencv2/opencv.hpp>

#include <QHash>
#include <QMap>
#include <QString>

#include <log4cxx/logger.h>

#include "SubSense/BackgroundSubtractorSuBSENSE.h"
#include "struck.h"

#include <MPFStreamingDetectionComponent.h>

/*! \class PPRStreamingDetection
 *  \brief face detection/tracking based on the Pittpatt 5 SDK for
 *         streaming video jobs
 */
class SubsenseStreamingDetection : public MPF::COMPONENT::MPFStreamingDetectionComponent {
  public:
    explicit SubsenseStreamingDetection(const MPF::COMPONENT::MPFStreamingVideoJob &job);
    ~SubsenseStreamingDetection() = default;

    virtual void BeginSegment(const MPF::COMPONENT::VideoSegmentInfo &segment_info) override;
    virtual bool ProcessFrame(const cv::Mat &frame, int frame_number) override;
    virtual std::vector<MPF::COMPONENT::MPFVideoTrack> EndSegment() override;

    std::string GetDetectionType() { return "MOTION"; }

  private:
    std::string job_name_;
    std::string msg_prefix_;
    log4cxx::LoggerPtr motion_logger_;
    unsigned int verbose_;
    bool segment_activity_detected_;   // Set to true when activity
                                        // has been detected in a
                                        // segment, so that activity
                                        // is only reported at most
                                        // once per segment.
    int frame_width_;
    int frame_height_;
    int previous_segment_number_;
    int current_segment_number_;
    int segment_frame_index_;
    int downsample_count_;
    int tracker_id_;

    BackgroundSubtractorSuBSENSE bg_;
    bool bg_initialized_;

    MPF::COMPONENT::MPFVideoTrack preprocessor_track_;
    std::vector<MPF::COMPONENT::MPFVideoTrack> tracks_;
    QMap<int, STRUCK> tracker_map_;
    QMap<int, MPF::COMPONENT::MPFVideoTrack> track_map_;

    QHash <QString, QString> parameters_;
};


#endif //OPENMPF_CONTRIB_COMPONENTS_MOTION_SUBSENSE_STREAMING_DETECTION_H
