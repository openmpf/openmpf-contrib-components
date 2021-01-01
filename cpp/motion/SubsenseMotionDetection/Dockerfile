#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

ARG BUILD_REGISTRY
ARG BUILD_TAG=latest
FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

COPY SubsenseMotionDetection SubsenseMotionDetection
COPY STRUCK STRUCK
COPY Utils Utils

RUN SRC_DIR=$SRC_DIR/SubsenseMotionDetection build-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./SubsenseMotionDetectionTest; fi


FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

ENV COMPONENT_LOG_NAME subsense-motion-detection.log

COPY --from=build_component $BUILD_DIR/plugin/SubsenseMotionDetection $PLUGINS_DIR/SubsenseMotionDetection

COPY --from=build_component $BUILD_DIR/libmpfSubsenseMotionDetection.so \
                            $PLUGINS_DIR/SubsenseMotionDetection/lib/


LABEL org.label-schema.license="GPL 2.0" \
      org.label-schema.name="OpenMPF SuBSENSE Motion Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-contrib-components" \
      org.label-schema.vendor="MITRE"
