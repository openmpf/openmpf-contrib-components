/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * This file is part of the Media Processing Framework (MPF)                  *
 * person detection component software.                                       *
 *                                                                            *
 * The MPF person detection component software is free software: you can      *
 * redistribute it and/or modify it under the terms of the GNU General        *
 * Public License as published by the Free Software Foundation, either        *
 * version 3 of the License, or (at your option) any later version.           *
 *                                                                            *
 * The MPF person detection component software is distributed in the hope     *
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the         *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.   *
 * See the (GNU General Public License for more details.                      *
 *                                                                            *
 * You should have received a copy of the GNU General Public License          *
 * along with the MPF person detection component software. If not, see        *
 * <http://www.gnu.org/licenses/>.                                            *
 ******************************************************************************/


#ifndef OPENMPF_CONTRIB_COMPONENTS_PERSONDETECTION_H
#define OPENMPF_CONTRIB_COMPONENTS_PERSONDETECTION_H


#include <QHash>
#include <QString>

#include <log4cxx/logger.h>

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <MPFDetectionComponent.h>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <MPFVideoCapture.h>

class PersonDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {
public:
    PersonDetection() : initialized(0), verbose(0) { }

    bool Init();
    bool Close();

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(const MPF::COMPONENT::MPFVideoJob &job) override;

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetections(const MPF::COMPONENT::MPFImageJob &job) override;

    std::string GetDetectionType();

private:
    bool initialized;
    bool verbose;
    QHash <QString, QString> parameters;

    log4cxx::LoggerPtr personLogger;
    void logPerson(const MPF::COMPONENT::MPFImageLocation& person, const std::string& job_name);
    void logTrack(const MPF::COMPONENT::MPFVideoTrack& track, const std::string& job_name);

    cv::Rect ImageLocationToCvRect(const MPF::COMPONENT::MPFImageLocation &detection);
    MPF::COMPONENT::MPFImageLocation CvRectToImageLocation(const cv::Rect &rect);

    void CloseAnyOpenTracks(int frame_index,
                            std::vector <MPF::COMPONENT::MPFVideoTrack> &tracks);
    void UpdateTracks(int frame_index,
                      std::vector <MPF::COMPONENT::MPFVideoTrack> &tracks);

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetectionsFromVideoCapture(
            const MPF::COMPONENT::MPFVideoJob &job,
            MPF::COMPONENT::MPFVideoCapture &video_capture);

    bool imshow_on;
};

#endif //OPENMPF_CONTRIB_COMPONENTS_PERSONDETECTION_H
