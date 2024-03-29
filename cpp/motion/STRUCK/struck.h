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

#ifndef STRUCK_H
#define STRUCK_H

#include <list>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "Tracker.h"
#include "Config.h"
//#include "vot.hpp"

class STRUCK
{
    Tracker* tracker;
    FloatRect init_BB;
    Config conf;
    double min_overlap;


public:
    STRUCK(){tracker = 0;}
    ~STRUCK();
    void initialize(cv::Mat first_frame, cv::Rect detection, double threshold, double min_overlap);
    cv::Rect nextFrame(cv::Mat frame, std::vector<cv::Rect> detections);
};

#endif // STRUCK_H
