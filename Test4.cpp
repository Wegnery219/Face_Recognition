#include <iostream>
#include<sstream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include<dlib/opencv/cv_image.h>
#include<string.h>
#include<cstring>
#include <opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include"opencv2\core\core_c.h"
using namespace cv;
using namespace std;
using namespace dlib;

std::vector <dlib::rectangle> dets;

bool get_face(Mat frame) {
	try
	{
		frontal_face_detector detector = get_frontal_face_detector();
		
		/*
		array2d< bgr_pixel> img(frame.rows, frame.cols);
		for (int i = 0; i < frame.rows; i++)
		{
			for (int j = 0; j < frame.cols; j++)
			{
				img[i][j].blue = frame.at< cv::Vec3b>(i, j)[0];
				img[i][j].green = frame.at< cv::Vec3b>(i, j)[1];
				img[i][j].red =frame.at< cv::Vec3b>(i, j)[2];
			}
		}
		//·Å´ó
		pyramid_up(img);
		*/
		 dets = detector(frame);
		 cout << dets.size();
		 if (dets.size() == 1) return true;
		 else return false;
		//win.add_overlay(dets, rgb_pixel(255, 0, 0));
	}
	catch (exception& e)
	{
		return false;
	}
}
void opencamera() {
	VideoCapture capture(0);

	while (true)
	{
		Mat frame;
		capture >> frame;
		imshow("view", frame);
		if (get_face(frame)) {
			cv::rectangle(frame, Point(int(dets[0].left()), int(dets[0].top())), Point(int(dets[0].right()), int(dets[0].bottom())), (255, 0, 0), 2, 8, 0);
			String Img_name = "E:\\shixun\\frcode\\face\\build\\trainpic\\";
			imwrite(Img_name + "test.jpg", frame);
			break;
			
			}

			int c = waitKey(30);
			if ((char)c == 'c') break;
		}
		return;
	}




