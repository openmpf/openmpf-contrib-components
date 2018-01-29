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

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "SubsenseMotionDetectionUtils.h"

using namespace MPF;
using namespace COMPONENT;

void GetPropertySettings(const std::map<std::string, std::string> &algorithm_properties,
                         QHash<QString, QString> &parameters) {
    std::string property;
    std::string str_value;

    for (std::map<std::string,std::string>::const_iterator imap = algorithm_properties.begin(); imap != algorithm_properties.end(); imap++) {
        property = imap->first;
        str_value = imap->second;
        parameters.insert(QString::fromStdString(property), QString::fromStdString(str_value));
    }
}
