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

#ifndef OPENMPF_CONTRIB_COMPONENTS_MOTION_UTILS_H
#define OPENMPF_CONTRIB_COMPONENTS_MOTION_UTILS_H

#include <map>

#include <opencv2/core.hpp>

#include <QHash>
#include <QString>
#include <QMap>

#include <log4cxx/logger.h>

#include <MPFDetectionComponent.h>
#include <struck.h>

void GetPropertySettings(const std::map<std::string, std::string> &algorithm_properties,
                         QHash<QString, QString> &parameters);


void SetPreprocessorTrack(const cv::Mat fore, int frame_index,
                          int frame_cols, int frame_rows,
                          MPF::COMPONENT::MPFVideoTrack &track,
                          std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);

std::vector<cv::Rect> GetResizedRects(const std::string &job_name,
                                      const log4cxx::LoggerPtr &logger,
                                      const QHash<QString, QString> &parameters,
                                      cv::Mat &fore,
                                      int frame_cols,
                                      int frame_rows,
                                      int downsample_count);

void ProcessMotionTracks(const QHash<QString, QString> &parameters,
                         const std::vector<cv::Rect> &resized_rects,
                         const cv::Mat &orig_frame,
                         int frame_index, int &tracker_id,
                         QMap<int, STRUCK> &tracker_map,
                         QMap<int, MPF::COMPONENT::MPFVideoTrack> &track_map,
                         std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);

cv::Rect Upscale(const cv::Rect &rect,
                 int frame_cols, int frame_rows,
                 int downsample_count);
    
void AssignDetectionConfidence(MPF::COMPONENT::MPFVideoTrack &track,
                               float distance_factor,
                               float size_factor);

#endif   // OPENMPF_CONTRIB_COMPONENTS_MOTION_UTILS_H
