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


#ifndef UTIL_
#define  UTIL_

#include <opencv2/core/types_c.h>
#include <stdio.h>

// fast image data access wrapper
template<class T> class Img_
{
public:
    Img_(IplImage* img=0){imgp=img;}
	~Img_(){imgp=0;}
	void operator=(IplImage* img){imgp=img;}
	inline T* operator[](const int rowIndx)
	{
		return (T*)(imgp->imageData+rowIndx*imgp->widthStep);
	}
private:
	IplImage* imgp;
};
typedef struct  
{
	unsigned char b,g,r;
}RgbPixel;
typedef struct
{
	float b,g,r;
}RgbPixelFloat;
typedef Img_<RgbPixel> RgbImage_;
typedef Img_<RgbPixelFloat> RgbImageFloat_;
typedef Img_<unsigned char> BwImage_;
typedef Img_<float> BwImageFloat_;

template<class T> class Img
{
public:
	Img(cv::Mat& img=0){imgp=&img;}
	~Img(){imgp=0;}
	void operator=(const cv::Mat& img){imgp=&img;}
	inline T* operator[](const int rowIndx)
	{
		return (T*)(imgp->data+rowIndx*imgp->step);
	}
private:
	cv::Mat* imgp;
};
typedef Img<RgbPixel> RgbImage;
typedef Img<RgbPixelFloat> RgbImageFloat;
typedef Img<unsigned char> BwImage;
typedef Img<float> BwImageFloat;

/* ****** ****** */

#define COLOR(ID) cv::Scalar(255-(5*ID)%256,255-(57*ID)%256,255-(1035*ID)%256)

//distance of rectangle
#define OVERLAP 0
inline double getRectDist(cv::Rect r1,cv::Rect r2,int type)
{
	cv::Rect op=r1&r2;
	return 1-(double)op.area()/(double)(r1.area()+r2.area()-op.area());
}
inline int string2int(const char* s)
{
	int i;
	if(sscanf(s, "%d", &i) == EOF )
	{
            std::cout<<"error reading integer"<<std::endl;
	}
	return i;
}

inline float string2float(const char* s)
{
	return (float)atof(s);
}

inline double _string2double(const std::string s)
{
	return atof(s.c_str());
}

inline std::string _double2string(double d)
{
	std::ostringstream s;
	s << d;
	return s.str();
}

inline int _char_p2int(const char* s)
{
	return atoi(s);
}

inline cv::Rect scaleWin(cv::Rect win, double scale)
{
	return cv::Rect(
		(int)(win.x+0.5*(1-scale)*win.width),	// CAUTION: could be rounded better (+0.5)
		(int)(win.y+0.5*(1-scale)*win.height),	 
		(int)(win.width*scale),
		(int)(win.height*scale));
}

#endif
