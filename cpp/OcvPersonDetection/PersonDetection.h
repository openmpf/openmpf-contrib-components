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

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks) override;

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFImageJob &job,
            std::vector<MPF::COMPONENT::MPFImageLocation> &locations) override;

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

    std::vector<cv::Rect>
    GetRectsAtFrameIndex(int frame_index,
                         const std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
                         std::vector <int> &track_indexes);

    std::vector <cv::Rect>
    GetRectsAtFrameIndex(int frame_index,
                         const std::vector <MPF::COMPONENT::MPFVideoTrack> &tracks);

    void writeDetectionToVideo(const int start_frame,
                               const int stop_frame,
                               const std::string &data_uri,
                               const int frame_count,
                               const int frame_interval,
                               const std::string& job_name,
                               std::vector <MPF::COMPONENT::MPFVideoTrack> &detections);

    MPF::COMPONENT::MPFDetectionError GetDetectionsFromVideoCapture(
            const MPF::COMPONENT::MPFVideoJob &job,
            const int frame_skip,
            MPF::COMPONENT::MPFVideoCapture &video_capture,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);

    bool imshow_on;
    bool output_video;
    bool output_image;
    std::string output_base_path;
};

#endif //OPENMPF_CONTRIB_COMPONENTS_PERSONDETECTION_H
