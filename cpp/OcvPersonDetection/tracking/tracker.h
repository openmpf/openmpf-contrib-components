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


#ifndef TRACKER_H
#define TRACKER_H

#include <list>

#include <opencv2/opencv.hpp>

#include "appTemplate.h"
#include "parameter.h"
#include "util.h"

class EnsembleTracker
{

public:
    EnsembleTracker(int id,cv::Size body_size,double phi1=0.5,double phi2=1.5,double phi_max=4.0);
	~EnsembleTracker();	

	//reference counting
	inline void refcAdd1(){	++_refc;	}
	inline void refcDec1(){--_refc;_refc=MAX(0,_refc);}
	inline size_t getRefc(){return _refc;}

	// memory management
	inline bool getIsDumped(){return _is_dumped;}
	void dump();
	static void emptyTrash();

	// major functions
	void updateNeighbors(
            std::list<EnsembleTracker*> tr_list,
            double dis_thresh_r=2.5, // ratio of distance thresh to '_match_radius'
            double scale_r1=1.2, double scale_r2=0.8,
            double hist_thresh=0.5);
	void addAppTemplate(const cv::Mat* frame_set,cv::Rect iniWin);
	void track(const cv::Mat* frame_set,cv::Mat& occ_map);
	void calcConfidenceMap(const cv::Mat* frame_set, cv::Mat& occ_map);//using kalman filter to decide the window
	void calcScore();//calculate each template's score
	void deletePoorTemplate(double threshold);
	void deletePoorestTemplate();		
	void demote();
	void promote();
	void registerTrackResult();//record result window and the filtered one

	//for auxiliary stable appearance
	void updateMatchHist(cv::Mat& frame);
	double compareHisto(cv::Mat& frame, cv::Rect win);
	
	inline double getVel()//get velocity
	{
		return (abs(_kf.statePost.at<float>(2,0))+abs(_kf.statePost.at<float>(3,0)))*25;
	}
	inline void setAddNew(bool b){_added_new=b;}
	inline bool getAddNew(){return _added_new;}
	inline double getHitFreq()
	{
		cv::Scalar s=sum(_recentHitRecord.row(0));
		return s[0]/MIN((double)_recentHitRecord.cols,_result_history.size());		
	}
	inline double getHitMeanScore()
	{
		return mean(_recentHitRecord.row(1))[0];
	}
	
	//drawing function
	inline void drawFilterWin(cv::Mat& frame)
	{
		//Rect win=_filter_result_history.back();
		//rectangle(frame,win,COLOR(_ID),1);
	}
	inline void drawResult(cv::Mat& frame,double scale)
	{
		scale-=1;
		cv::Rect win=_result_history.back();//_result_history.back();
		if (!getIsNovice())
			rectangle(frame,scaleWin(win,1/0.5),COLOR(_ID),2);
		else
			rectangle(frame,scaleWin(win,1/0.5),COLOR(_ID),1);
	}
	inline void drawAssRadius(cv::Mat& frame)
	{
		cv::Rect win=_result_temp;
		circle(frame,cv::Point((int)(win.x+0.5*win.width),(int)(win.y+0.5*win.height)),(int)MAX(_match_radius,0),COLOR(_ID),1);
	}

	//getting function
	inline double getAssRadius(){return _match_radius;}	
	inline int getTemplateNum(){return _template_list.size();}
	inline std::vector<cv::Rect>& getResultHistory(){return _result_history;}
	inline int getID(){return _ID;}
	inline double getDisToLast(cv::Rect win) // to last position of non-novice status
	{
		return sqrt(
			(win.x+0.5*win.width-_result_last_no_sus.x-0.5*_result_last_no_sus.width)*(win.x+0.5*win.width-_result_last_no_sus.x-0.5*_result_last_no_sus.width)+
			(win.y+0.5*win.height-_result_last_no_sus.y-0.5*_result_last_no_sus.height)*(win.y+0.5*win.height-_result_last_no_sus.y-0.5*_result_last_no_sus.height)
			);		
	}
	inline bool getIsNovice(){return _is_novice;	}
	inline int getSuspensionCount(){return _novice_status_count;}
	inline double getHistMatchScore(){return hist_match_score;}
	inline cv::Rect getResult(){return _result_temp;}
	inline cv::Rect getBodysizeResult(){return _result_bodysize_temp;}

	inline void updateKfCov(double body_width)
	{
            cv::Mat m_temp=(cv::Mat_<float>(4,4)<<0.025,0,0,0,0,0.025,0,0,0,0,0.25,0,0,0,0,0.25);
		_kf.processNoiseCov=m_temp*((float)body_width/25)*((float)body_width/25);
		setIdentity(_kf.measurementNoiseCov,cv::Scalar::all(1.0*(float)body_width*(float)body_width));
	}

private:
	inline void init_kf(cv::Rect win)
	{
            _kf.statePost=(cv::Mat_<float>(4,1)<<win.x+0.5*win.width,win.y+0.5*win.height,0,0);
	}
	inline void correct_kf(cv::KalmanFilter& kf, cv::Rect win)
	{
            cv::Mat m_temp=(cv::Mat_<float>(2,1)<<win.x+0.5*win.width,win.y+0.5*win.height);
            kf.correct(m_temp);
	}
	inline double compareHisto(cv::Mat& h)
	{
		return compareHist(hist,h,cv::HISTCMP_INTERSECT);
	}

	typedef struct TraResult
	{
		cv::Rect window;
		double likelihood;
		TraResult(cv::Rect win,double like){window=win;likelihood=like;}
	}TraResult;

	size_t _refc;
	bool _is_dumped;
	double _phi1_;
	double _phi2_;
	double _phi_max_;
	int _novice_status_count;//cout the consecutive times being a novice
	int _template_count;//for its member
	int _ID;
	cv::KalmanFilter _kf;
	bool _is_novice;
	double _match_radius;
	double hist_match_score;
	bool _added_new;
	int _record_idx;
	
	static std::list<EnsembleTracker*> _TRASH_LIST;
	std::list<AppTemplate*> _template_list;
	AppTemplate* _retained_template;//The last template when all others are eliminated
	std::vector<cv::Rect> _result_history;
	std::vector<cv::Rect> _filter_result_history;
	int histSize[3];
	float _hRang[3][2];
	const float* hRange[3];
	int channels[3];
	cv::MatND hist;// 3d histogram for stable appearance
	cv::Size2f _window_size;//
	cv::Mat _confidence_map;
	cv::Rect _cm_win;//roi for computing confidence map
	cv::Rect _result_temp;//to store the meanshift result
	cv::Rect _result_last_no_sus;// to store the last result when not _is_novice
	cv::Rect _result_bodysize_temp;//GT size reuslt
	std::list<EnsembleTracker*> _neighbors;
	cv::Mat _recentHitRecord;
};


#endif
