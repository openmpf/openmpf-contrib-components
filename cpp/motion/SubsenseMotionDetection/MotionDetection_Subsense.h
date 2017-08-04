/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2017 The MITRE Corporation. All Rights Reserved.                 *
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


#ifndef OPENMPF_CONTRIB_COMPONENTS_MOTIONDETECTION_SUBSENSE_H
#define OPENMPF_CONTRIB_COMPONENTS_MOTIONDETECTION_SUBSENSE_H


#include <string>
#include <vector>

#include <QHash>
#include <QString>
#include <log4cxx/logmanager.h>
#include <log4cxx/fileappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/consoleappender.h>

#include <MPFDetectionComponent.h>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <MPFVideoCapture.h>

class MotionDetection_Subsense : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

    log4cxx::LoggerPtr motion_logger;
    QHash<QString, QString> parameters;

public:
    MotionDetection_Subsense();
    ~MotionDetection_Subsense();
    bool Init();
    bool Close();

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks) override;

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFImageJob &job,
            std::vector<MPF::COMPONENT::MPFImageLocation> &locations) override;

    std::string GetDetectionType();

private:
    MPF::COMPONENT::MPFDetectionError GetDetectionsFromVideoCapture(
            const MPF::COMPONENT::MPFVideoJob &job,
            const int frame_skip,
            MPF::COMPONENT::MPFVideoCapture &video_capture,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);

    void GetPropertySettings(const std::map<std::string, std::string> &algorithm_properties);

    static cv::Rect Upscale(const cv::Rect &rect, const cv::Mat &orig_frame, int downsample_count);
};


#endif //OPENMPF_CONTRIB_COMPONENTS_MOTIONDETECTION_SUBSENSE_H
