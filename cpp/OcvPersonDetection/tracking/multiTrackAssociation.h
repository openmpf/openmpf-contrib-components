/*************************************************************
*	Implemetation of the multi-person tracking system described in paper
*	"Online Multi-person Tracking by Tracker Hierarchy", Jianming Zhang, 
*	Liliana Lo Presti, Stan Sclaroff, AVSS 2012
*	http://www.cs.bu.edu/groups/ivc/html/paper_view.php?id=268
*
*	Copyright (C) 2012 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: jmzhang@bu.edu
***************************************************************/


#ifndef MULTI_TRACK_ASSOCIATION
#define MULTI_TRACK_ASSOCIATION

#include <fstream>

#include "opencv2/opencv.hpp"

#include "MPFDetectionComponent.h"

#include "parameter.h"
#include "util.h"
#include "tracker.h"
#include "detector.h"

#define GOOD 0
#define NOTSURE 1
#define BAD 2

#define COUNT_NUM 1000.0
#define SLIDING_WIN_SIZE 7.2*11 

typedef struct Result2D
{
	int id;
	float xc, yc;  //center point 
	float w, h; //width, height
	double response;
	Result2D(int i,float x_,float y_,float w_,float h_,double res=1)
		:id(i),xc(x_),yc(y_),w(w_),h(h_),response(res){}
	Result2D(){}
}Result2D;

class WaitingList
{
	typedef struct Waiting
	{
            int accu;
            int life_count;
            cv::Rect currentWin;
            cv::Point center;
            Waiting(cv::Rect win)
			:accu(1),
			life_count(1),
			currentWin(win),
			center((int)(win.x+0.5*win.width),(int)(win.y+0.5*win.height))
            {
            }
	}Waiting;

    std::list<Waiting> w_list;
    int life_limit;

public:
    WaitingList(int life):life_limit(life){}
    void update();
    std::vector<cv::Rect>outputQualified(double thresh);
    void feed(cv::Rect bodysize_win,double response);
};

class Controller
{
public:
	WaitingList waitList;
	WaitingList waitList_suspicious;

	Controller(
		cv::Size sz,int r, int c,double vh=0.01,
		double lr=1/COUNT_NUM,
		double thresh_expert=0.5);
	void takeVoteForHeight(cv::Rect bodysize_win);	
	std::vector<int> filterDetection(std::vector<cv::Rect> detction_bodysize);	
	void takeVoteForAvgHittingRate(std::list<EnsembleTracker*> _tracker_list);	

	/*
	Tracker death control. For modifying termination conditions, change here.
	*/
    void deleteObsoleteTracker(std::list<EnsembleTracker*>& _tracker_list,
                               std::vector<MPF::COMPONENT::MPFVideoTrack>& tracks,
                               int frame_index);
	
	void calcSuspiciousArea(std::list<EnsembleTracker*>& _tracker_list);	
        inline std::vector<cv::Rect> getQualifiedCandidates()
	{
		/*
		For modifying the birth condition for trackers, change here.
		*/
		double l=_hit_record._getAvgHittingRate(_alpha_hitting_rate,_beta_hitting_rate);
		return waitList.outputQualified((l-sqrt(l)-1.0));		
	}
private:
	// a rotate array for keep record of the average hitting rate
	typedef struct HittingRecord
	{
		cv::Mat record;
		int idx;
		HittingRecord():idx(0)
		{
			record=cv::Mat::zeros(2,(int)(SLIDING_WIN_SIZE),CV_64FC1);
		}
		void recordVote(bool vote)
		{
			idx=idx-record.cols*(idx/record.cols);
			record.at<double>(0,idx)=vote ? 1.0:0;
			record.at<double>(1,idx)=1;
			idx++;
		}
		double _getAvgHittingRate(double _alpha_hitting_rate, double _beta_hitting_rate)
		{
			cv::Scalar s1=sum(record.row(0));
			cv::Scalar s2=sum(record.row(1));
			return (s1[0]*11+_alpha_hitting_rate)/(_beta_hitting_rate+s2[0]);
		}
	}HittingRecord;
	
	HittingRecord _hit_record;
	int _grid_rows;
	int _grid_cols;
	double _prior_height_variance;
	cv::Size _frame_size;
	double _bodyheight_learning_rate;
	double _alpha_hitting_rate;
	double _beta_hitting_rate;
	double _thresh_for_expert;
	
	std::vector<std::vector<double> > _bodyheight_map;
	std::vector<std::vector<double> > _bodyheight_map_count;
	std::vector<cv::Rect> _suspicious_rect_list;
};

class TrakerManager
{
public:
	TrakerManager(
		HogDetector* detctor,cv::Mat& frame,
		double thresh_promotion);
	~TrakerManager();	
	void doWork(cv::Mat& frame, int frame_index,
                    std::vector<MPF::COMPONENT::MPFVideoTrack>& tracks);

	void setKey(char c)
	{
		_my_char = c;
	}	
private:	
	void doHungarianAlg(const std::vector<cv::Rect>& detections);
	inline static bool compareTraGroup(EnsembleTracker* c1,EnsembleTracker* c2)
	{
		return c1->getTemplateNum()>c2->getTemplateNum() ? true:false;
	}
	HogDetector* _detector;
	char _my_char;
	int _frame_count;
	int _tracker_count;
	Controller _controller;
	cv::Mat* _frame_set;
	std::list<EnsembleTracker*> _tracker_list;	
	cv::Mat _occupancy_map;	
	//XMLBBoxWriter resultWriter;
	double _thresh_for_expert_;
};
	


#endif
