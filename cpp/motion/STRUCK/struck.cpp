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

#include "struck.h"

#include <iostream>
#include <fstream>




using namespace std;
using namespace cv;


static const int kLiveBoxWidth = 80;
static const int kLiveBoxHeight = 80;


STRUCK::~STRUCK() {
    if (tracker) {
        delete tracker;
    }
}

void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
    IntRect r(rRect);
    rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}

void STRUCK::initialize(Mat first_frame, cv::Rect detection, double threshold, double min_overlap) {
    // Initialize config variable

    Config::FeatureKernelPair pair;
    conf.svmC = 100.0f;
    conf.svmBudgetSize = 100;
    conf.frameWidth = first_frame.rows;
    conf.frameHeight = first_frame.cols;
    pair.feature = Config::kFeatureTypeHaar;
    pair.kernel = Config::kKernelTypeGaussian;
    pair.params.push_back(0.2);
    conf.features.push_back(pair);
    this->min_overlap = min_overlap;

    tracker = new ::Tracker(conf, threshold);
    init_BB = FloatRect(detection.x, detection.y, detection.width, detection.height);
    tracker->Initialise(first_frame, init_BB);

}

cv::Rect STRUCK::nextFrame(Mat frame, vector<cv::Rect> detections) {
    vector<FloatRect> rects;
    cv::Rect init_BB_opencv(init_BB.XMin(), init_BB.YMin(), init_BB.Width(), init_BB.Height());
    for (cv::Rect detection : detections) {
        if ((detection & init_BB_opencv).area() > (init_BB_opencv.area() * min_overlap)) {
            rects.push_back(FloatRect(detection.x, detection.y, detection.width, detection.height));
        }
    }




    if (tracker->IsInitialised() && rects.size() > 0) {
        tracker->Track(frame, rects);

        init_BB = tracker->GetBB();
        return cv::Rect(init_BB.XMin(), init_BB.YMin(), init_BB.Width(), init_BB.Height());
    } else {
        return cv::Rect(0, 0, 1, 1);
    }
}

