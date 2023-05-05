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
 * This file is part of the Media Processing Framework (MPF)                  *
 * motion detection component software.                                       *
 *                                                                            *
 * The MPF motion detection component software is free software: you can      *
 * redistribute it and/or modify it under the terms of the GNU General        *
 * Public License as published by the Free Software Foundation, either        *
 * version 3 of the License, or (at your option) any later version.           *
 *                                                                            *
 * The MPF motion detection component software is distributed in the hope     *
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the         *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.   *
 * See the (GNU General Public License for more details.                      *
 *                                                                            *
 * You should have received a copy of the GNU General Public License          *
 * along with the MPF motion detection component software. If not, see        *
 * <http://www.gnu.org/licenses/>.                                            *
 ******************************************************************************/


#ifndef OPENMPF_CONTRIB_COMPONENTS_MOTIONDETECTION_MOG2_H
#define OPENMPF_CONTRIB_COMPONENTS_MOTIONDETECTION_MOG2_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <log4cxx/logger.h>

#include <MPFDetectionComponent.h>
#include <MPFDetectionObjects.h>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <MPFVideoCapture.h>


struct MogConfig;

class MotionDetection_MOG2 : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

public:
    bool Init() override;
    bool Close() override;

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job) override;

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetections(
            const MPF::COMPONENT::MPFImageJob &job) override;

    std::string GetDetectionType() override;

private:
    log4cxx::LoggerPtr motion_logger_;

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetectionsFromVideoCapture(
            const MPF::COMPONENT::MPFVideoJob &job,
            const MogConfig &mog_config,
            MPF::COMPONENT::MPFVideoCapture &video_capture);

    static cv::Rect Upscale(const cv::Rect &rect, const cv::Mat &orig_frame, int downsample_count);
};



#endif //OPENMPF_CONTRIB_COMPONENTS_MOTIONDETECTION_MOG2_H
