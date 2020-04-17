/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2019 The MITRE Corporation                                       *
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


#include <string>
#include <utility>
#include <vector>
#include <chrono> 

#include <gtest/gtest.h>
#include <MPFDetectionComponent.h>
#include <QDir>
#include <MPFSimpleConfigLoader.h>
#include <Utils.h>
#include <ReadDetectionsFromFile.h>
//#include <DetectionComparison.h>
#include <VideoGeneration.h>
#include <WriteDetectionsToFile.h>
#include <ImageGeneration.h>

#define private public
#include "OcvSsdFaceDetection.h"
#include "DetectionComparisonA.h"

using namespace std;
using namespace MPF::COMPONENT;

//-----------------------------------------------------------------------------
// global variable to hold the file name parameters
//-----------------------------------------------------------------------------
 QHash<QString, QString> GetTestParameters(){

  QString current_path = QDir::currentPath();
  QHash<QString, QString> parameters;
  string config_path(current_path.toStdString() + "/config/test_ocv_ssd_face_config.ini");
  std::cout << "config path: " << config_path << std::endl;
  int rc = LoadConfig(config_path, parameters);
  if(rc == 0){
    std::cout << "config file loaded" << std::endl;
  }else{
    parameters.clear();
    std::cout << "config file failed to load with error:"<< rc << std::endl;
  }
  return parameters;
}

//-----------------------------------------------------------------------------
// get current working directory with minimal error checking
//-----------------------------------------------------------------------------
static string GetCurrentWorkingDirectory() {
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
      std::cout << "Current working dir: " << cwd << std::endl;
      return string(cwd);
  }else{
      std::cout << "getcwd() error";
      return "";
  }
}

//-----------------------------------------------------------------------------
// Test initializing component 
//-----------------------------------------------------------------------------
TEST(OcvSsdFaceDetection, Init) {

  string current_working_dir = GetCurrentWorkingDirectory();

  OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
  ASSERT_TRUE(NULL != ocv_ssd_face_detection);

  string dir_input(current_working_dir + "/../plugin");
  ocv_ssd_face_detection->SetRunDirectory(dir_input);
  string rundir = ocv_ssd_face_detection->GetRunDirectory();
  EXPECT_EQ(dir_input, rundir);

  ASSERT_TRUE(ocv_ssd_face_detection->Init());

  MPFComponentType comp_type = ocv_ssd_face_detection->GetComponentType();
  ASSERT_TRUE(MPF_DETECTION_COMPONENT == comp_type);

  EXPECT_TRUE(ocv_ssd_face_detection->Close());
  delete ocv_ssd_face_detection;
}

//-----------------------------------------------------------------------------
// This test checks the confidence of faces detected by the OpenCV low-level 
// detection facility, which is used by the OpenCV face detection component.
//-----------------------------------------------------------------------------
TEST(OcvSsdFaceDetection, DISABLED_VerifyQuality) {

    string current_working_dir = GetCurrentWorkingDirectory();
    string plugins_dir         = current_working_dir + "/../plugin/OcvSsdFaceDetection";

    QHash<QString, QString> parameters = GetTestParameters();
    ASSERT_TRUE(parameters.count() > 0);


    // 	Create an OCV face detection object.
    std::cout << "\tCreating OCV SSD Face Detection" << std::endl;
    OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ocv_ssd_face_detection);
    ocv_ssd_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_ssd_face_detection->Init());

    //  Load test image
    string test_image_path = parameters["OCV_FACE_1_FILE"].toStdString();
    if(test_image_path.find_first_of('.') == 0) {
        test_image_path = current_working_dir + "/" + test_image_path;
    }
    cv::Mat image = cv::imread(test_image_path, CV_LOAD_IMAGE_IGNORE_ORIENTATION + CV_LOAD_IMAGE_COLOR);
    ASSERT_TRUE(!image.empty());

    // Detect detections and check conf levels
    DetectionLocationVec detections;
    JobConfig cfg;
    cfg.bgrFrame = image;
    ocv_ssd_face_detection->_detect(cfg, detections);
    ASSERT_TRUE(detections.size() == 1);
    cout << "Detection: " << detections[0] << endl;
    ASSERT_TRUE(detections[0].confidence > .9);
    detections.clear();

    // Detect detections and check conf level
    cfg.minDetectionSize = 500;
    ocv_ssd_face_detection->_detect(cfg, detections);
    ASSERT_TRUE(detections.size() == 0);
    detections.clear();

    // Detect detections and check conf levels
    cfg.minDetectionSize = 48;
    cfg.confThresh = 1.1;
    ocv_ssd_face_detection->_detect(cfg, detections);
    ASSERT_TRUE(detections.size() == 0);
    detections.clear();

    delete ocv_ssd_face_detection;


}

//-----------------------------------------------------------------------------
//  Test face detection in images
//-----------------------------------------------------------------------------
TEST(OcvSsdFaceDetection, DISABLED_TestOnKnownImage) {
    string current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";

    std::cout << "Reading parameters for image test." << std::endl;

    string known_image_file          = parameters["OCV_FACE_IMAGE_FILE"       ].toStdString();
    string known_detections_file     = parameters["OCV_FACE_KNOWN_DETECTIONS" ].toStdString();
    string output_image_file         = parameters["OCV_FACE_IMAGE_OUTPUT_FILE"].toStdString();
    string output_detections_file    = parameters["OCV_FACE_FOUND_DETECTIONS" ].toStdString();
    float comparison_score_threshold = parameters["OCV_FACE_COMPARISON_SCORE_IMAGE"].toFloat();

    // 	Create an OCV face detection object.
    OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ocv_ssd_face_detection);

    ocv_ssd_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_ssd_face_detection->Init());

    std::cout << "Input Known Detections:\t"  << known_detections_file      << std::endl;
    std::cout << "Output Found Detections:\t" << output_detections_file     << std::endl;
    std::cout << "Input Image:\t"             << known_image_file           << std::endl;
    std::cout << "Output Image:\t"            << output_image_file          << std::endl;
    std::cout << "comparison threshold:\t"    << comparison_score_threshold << std::endl;

    // 	Load the known detections into memory.
    vector<MPFImageLocation> known_detections;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadImageLocations(known_detections_file, known_detections));

    vector<MPFImageLocation> found_detections;
    MPFImageJob image_job("Testing", known_image_file, { }, { });
    ASSERT_FALSE(ocv_ssd_face_detection->GetDetections(image_job, found_detections));
    EXPECT_FALSE(found_detections.empty());

    float comparison_score = DetectionComparisonA::CompareDetectionOutput(found_detections, known_detections);
    std::cout << "Detection comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(known_image_file,
                                               found_detections,
                                               test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                            found_detections);

    EXPECT_TRUE(ocv_ssd_face_detection->Close());
    delete ocv_ssd_face_detection;
}


//-----------------------------------------------------------------------------
//  Test face-recognition with thumbnail images
//----------------------------------------------------------------------------- 

TEST(OcvSsdFaceDetection, DISABLED_Thumbnails) {
    string current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";
    std::cout << "Reading parameters for thumbnail test." << endl;

    // Get test image filenames into vector
    string test_file_dir = parameters["OCV_FACE_THUMBNAIL_TEST_FILE_DIR"].toStdString();
    std::cout << "Input Image Dir: " << test_file_dir << endl;
    vector<string> img_file_names;
    size_t idx = 0;
    while(1){
      stringstream ss;
      ss << "OCV_FACE_THUMBNAIL_TEST_FILE_" << setfill('0') << setw(2) << idx;
      QString key = QString::fromStdString(ss.str());
      auto img_file_it = parameters.find(key);
      if(img_file_it == parameters.end()) break;
      img_file_names.push_back(img_file_it.value().toStdString());
      idx++;
    }
    std::cout << "Found " << img_file_names.size() << " test images" << endl;

    // 	Create an OCV face detection object.
    OcvSsdFaceDetection *ssd = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ssd);

    ssd->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ssd->Init());

    TrackList tracks;
    for(auto img_file_name:img_file_names){
      string img_file = test_file_dir + img_file_name;
      MPFImageLocationVec found_detections;
      MPFImageJob job("Testing", img_file, { }, { });

      JobConfig cfg(job);
      cfg.bgrFrame = cv::imread(img_file);
      EXPECT_TRUE(NULL != cfg.bgrFrame.data) << "Could not load:" << img_file;

      // find detections
      DetectionLocationVec detections;
      ssd->_detect(cfg, detections);
      EXPECT_FALSE(detections.empty());

      // get landmarks
      ssd->_findLandmarks(cfg, detections);
      for(auto &det:detections){
        EXPECT_FALSE(det.landmarks.empty());
      }

      // draw landmarks
      JobConfig cfg2(job);
      cfg2.bgrFrame = ssd->_drawLandmarks(cfg,detections);
      cv::imwrite(test_output_dir + "lm_" + img_file_name,cfg2.bgrFrame);

      // create thumbnails
      ssd->_createThumbnails(cfg,detections);


      // calculate feature vectors
      ssd->_calcFeatures(cfg, detections);
      for(auto &det:detections){
        EXPECT_FALSE(det.feature.empty());
      }

      if(tracks.size() == 0){
        for(auto &det:detections){
          //DetectionLocationVec newTrack= {det};
          tracks.push_back({det});
        }
      }else{
        cv::Mat ass = ssd->_calcAssignemntMatrix(cfg,tracks,detections);
        ssd->_assignDetections2Tracks(cfg,tracks,detections, ass);
      }
    }

    // write out thumbnail image tracks
    MPF::COMPONENT::TrackList::iterator tracksItr = tracks.begin();
    for(size_t t=0; t<tracks.size();t++){
      Track track = *tracksItr;
      for(size_t i=0; i<track.size(); i++){
        EXPECT_FALSE(track[i].thumbnail.empty());
        stringstream ss;
        ss << test_output_dir << "t" << t << "_i"<< i << ".png";
        string out_file = ss.str();
        cout << "Writing tumbnail: " << out_file << endl;
        cv::imwrite(out_file,track[i].thumbnail);
      }
      tracksItr++;
    }

    EXPECT_TRUE(ssd->Close());
    delete ssd;
}

//-----------------------------------------------------------------------------
//  Test face detection and tracking in videos
//-----------------------------------------------------------------------------
TEST(OcvSsdFaceDetection, TestOnKnownVideo) {

    string         current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";

    std::cout << "Reading parameters for video test." << std::endl;

    int start = parameters["OCV_FACE_START_FRAME"].toInt();
    int  stop = parameters["OCV_FACE_STOP_FRAME" ].toInt();
    int  rate = parameters["OCV_FACE_FRAME_RATE" ].toInt();
    string inTrackFile  = parameters["OCV_FACE_KNOWN_TRACKS"     ].toStdString();
    string inVideoFile  = parameters["OCV_FACE_VIDEO_FILE"       ].toStdString();
    string outTrackFile = parameters["OCV_FACE_FOUND_TRACKS"     ].toStdString();
    string outVideoFile = parameters["OCV_FACE_VIDEO_OUTPUT_FILE"].toStdString();
    float comparison_score_threshold = parameters["OCV_FACE_COMPARISON_SCORE_VIDEO"].toFloat();

    std::cout << "Start:\t"    << start << std::endl;
    std::cout << "Stop:\t"     << stop  << std::endl;
    std::cout << "Rate:\t"     << rate  << std::endl;
    std::cout << "inTrack:\t"  << inTrackFile  << std::endl;
    std::cout << "outTrack:\t" << outTrackFile << std::endl;
    std::cout << "inVideo:\t"  << inVideoFile  << std::endl;
    std::cout << "outVideo:\t" << outVideoFile << std::endl;
    std::cout << "comparison threshold:\t" << comparison_score_threshold << std::endl;

    // 	Create an OCV face detection object.
    std::cout << "\tCreating OCV Face Detection" << std::endl;
    OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ocv_ssd_face_detection);
    ocv_ssd_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_ssd_face_detection->Init());
    
    // 	Load the known tracks into memory.
    std::cout << "\tLoading the known tracks into memory: " << inTrackFile << std::endl;
    vector<MPFVideoTrack> known_tracks;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    // create output kown video to view ground truth
    // std::cout << "\tWriting ground truth video and test tracks to files." << std::endl;
    // VideoGeneration video_generation_gt;
    // video_generation_gt.WriteTrackOutputVideo(inVideoFile, known_tracks, (test_output_dir + "/ground_truth.avi"));
    // WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/ground_truth.txt"), known_tracks);

    // 	Evaluate the known video file to generate the test tracks.
    std::cout << "\tRunning the tracker on the video: " << inVideoFile << std::endl;
    vector<MPFVideoTrack> found_tracks;
    MPFVideoJob videoJob("Testing", inVideoFile, start, stop, { }, { });
    auto start_time = chrono::high_resolution_clock::now();
    ASSERT_FALSE(ocv_ssd_face_detection->GetDetections(videoJob, found_tracks));
    auto end_time = chrono::high_resolution_clock::now();
    EXPECT_FALSE(found_tracks.empty());
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
    time_taken = time_taken * 1e-9;
    std::cout << "\tVideoJob processing time: " << fixed << setprecision(5) << time_taken << "[sec]" << std::endl;

    // create output video to view performance
    std::cout << "\tWriting detected video and test tracks to files." << std::endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    // 	Compare the known and test track output.
    std::cout << "\tComparing the known and test tracks." << std::endl;
    float comparison_score = DetectionComparisonA::CompareDetectionOutput(found_tracks, known_tracks);
    std::cout << "Tracker comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > comparison_score_threshold);



    // don't forget
    std::cout << "\tClosing down detection." << std::endl;
    EXPECT_TRUE(ocv_ssd_face_detection->Close());
    delete ocv_ssd_face_detection;
}

//-----------------------------------------------------------------------------
int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
