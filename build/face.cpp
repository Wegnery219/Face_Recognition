#include<dlib/dnn.h>
//#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include<dlib/matrix.h>
//#include <dlib/image_io.h>
//#include <dlib/image_processing/frontal_face_detector.h>
#include<dlib/pixel.h>
#include<dlib/image_processing/scan_fhog_pyramid.h>
#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include<dlib/image_io.h>
#include<io.h>
#include<vector>
#include<fstream>
#include<iostream>
#define RATIO 1  
#define SKIP_FRAMES 2 

using namespace dlib;
using namespace std;
/*
导入使用多线程 6min
图片格式：label.jpg 向量文件格式：label.txt
不能在数据库中添加人脸图片，只能通过导入图片的方式加入，否则会导致打开系统每次都要训练一次

*/
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;
std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
);

frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;


bool flag;//全局变量，返回是否获取到了图像
//获取文件
void getFiles(string path, std::vector<string> & files)
{
	//文件句柄  
	long  long hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		try {
			do
			{
				//如果是目录,迭代之  
				//如果不是,加入列表  
				if ((fileinfo.attrib &  _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
						getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
				else
				{
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					files.push_back(fileinfo.name);
				}
			} while (_findnext(hFile, &fileinfo) == 0);
		}
		catch (exception &e) {
			cout << e.what() << endl;
		}
		_findclose(hFile);
	}
	return;
}


//打开摄像头，返回含有人脸的mat图像
cv::Mat open_video() {
	cv::Mat falsereturn;
	try
	{
		cv::VideoCapture cap("E:\\shixun\\frcode\\face\\build\\test1.mkv");
		image_window win;
		//cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);  
		//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  
		// Load face detection and pose estimation models.  
		
		
		int count = 0;
		std::vector<rectangle> faces;
		// Grab and process frames until the main window is closed by the user.  
		int times = 0;
		
		while (!win.is_closed())
		{
			
			if (times > 10) { flag = false; break; }
			// Grab a frame  
			cv::Mat img, img_small;
			cap >> img;
			
//			cv::imshow("test", img);
			cv::resize(img, img_small, cv::Size(), 1.0 / RATIO, 1.0 / RATIO);
			dlib::cv_image<bgr_pixel>cimg(img);
			dlib::cv_image<bgr_pixel>cimg_small(img_small);
			win.clear_overlay();
			win.set_image(cimg);
			// Detect faces   
			if (count++ % SKIP_FRAMES == 0) {
				faces = detector(cimg_small);
			}
			// Find the pose of each face.  
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i) {
				rectangle r(
					(long)(faces[i].left() * RATIO),
					(long)(faces[i].top() * RATIO),
					(long)(faces[i].right() * RATIO),
					(long)(faces[i].bottom() * RATIO)
				);
				shapes.push_back(sp(cimg, r));
				for (int k = 0; k < 68; ++k) {
					circle(img, cvPoint(shapes[i].part(k).x(), shapes[i].part(k).y()), 3, cv::Scalar(0, 0, 255), -1);
				}
			}
//			std::cout << "count:" << count << std::endl;
			// Display it all on the screen  
			if (count == 1) {
				
				win.add_overlay(render_face_detections(shapes));
				flag = true;
				return img;
			}
			times++;
		}
		
	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
	flag = false;
	system("pause");
	return falsereturn;
}

//匹配，将open_video获取的图像与对应二维码扫出来的string对应的图片进行match（欧式距离比较）
bool match(string label) {
	//先找有没有对应的人脸信息
	bool judge=false;
	char *filePath = "E:\\shixun\\frcode\\face\\build\\trainpic";
	std::vector<string> files;
	getFiles(filePath, files);
	for (int i = 0; i < files.size(); i++)
	{
		if (files[i] == label + ".jpg") {//有对应的图就说明有txt文件
			judge = true;
			break;
		}
	}
	if (judge == false) {
		cout << "二维码对应的人脸信息不存在" << endl;
		return false;
	}
	cv::Mat img = open_video();

}

//导入人脸图像，给定二维码对应的string，即保存的文件名，首先检查是否有重复，没有则打开摄像头，获取并保存
//保存之后将人脸特征向量保存到文件label.jpg
//数据库中图片格式都是jpg格式
void get_in(string label) {
	char *filePath = "E:\\shixun\\frcode\\face\\build\\trainpic";
	std::vector<string> files;
	getFiles(filePath, files);
	for (int i = 0; i < files.size(); i++)
	{
		if (files[i] == label + ".jpg") {
			cout << "人脸信息已存在" << endl;
			return;
		}
	}
	cv::Mat pic = open_video();
	if (flag == false) cout << "未识别到人脸，请重试"<<endl;
	else {
		//保存
		cv::imwrite("E:\\shixun\\frcode\\face\\build\\trainpic\\" + label + ".jpg", pic);
//		frontal_face_detector detector = get_frontal_face_detector();
//		shape_predictor sp;
//		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
		dlib::cv_image<bgr_pixel>cpic(pic);
		std::vector<matrix<rgb_pixel>> faces;
		std::vector<rectangle> face = detector(cpic);
//		{
//		std::vector<full_object_detection> shape;
//		shape.push_back(sp(cpic, face));
//		matrix<rgb_pixel> face_chip;
//		extract_image_chip(cpic, get_face_chip_details(shape, 150, 0.25), face_chip);
//		faces.push_back(move(face_chip));
//		}
		for (auto face : detector(cpic))
		{
			auto shape = sp(cpic, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(cpic, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
			// Also put some boxes on the faces so we can see that the detector is finding
			// them.
//			win.add_overlay(face);
		}
		std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
		ofstream location_out;
		location_out.open("E:\\shixun\\frcode\\face\\build\\vectorfile\\" + label + ".txt", std::ios::out | std::ios::app);
		if (!location_out.is_open()) {
			cout << "文件夹打不开" << endl;
			return;
		}
		location_out << trans(face_descriptors[0])<<endl;
		system("pause");
	}
}
//只通过导入人脸的方式就不需要train函数
//train函数，程序启动之前必须train一次，检查库中所有人脸是否都有对应的人脸特征向量文件
//void train() {
//	return;
//}
int main()
{
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	string str;
	cin >> str;
	get_in(str);
}