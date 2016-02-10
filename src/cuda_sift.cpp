#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#if FINDOBJECT_NONFREE == 0
#include <opencv2/nonfree/gpu.hpp>
#endif
#include <opencv2/gpu/gpu.hpp>
#include <chrono>
#include "sift.h"




using namespace cv;


void attendi(){
std::cout<<std::endl<<std::endl<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl<<std::endl;
while(!getchar()){};
std::cout<<std::endl<<std::endl;
};


class ObjectFind
{
private:
  cv::Mat frame; /// Image provided by the RGBD camera
  cv::Mat image; /// template image
  //ros::Subscriber cam_info_sub;
  cv::Mat outImage;
  
  image_transport::Publisher image_pub;
  image_transport::Publisher debug_pub;
  
  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber image_sub;
  std::string image_path;

  std::vector<cv::KeyPoint>  keypoints_image;
  std::vector<float>  descriptors_image;
  std::vector<cv::KeyPoint>  keypoints_frame;
  std::vector<float>  descriptors_frame;
  SiftGPUWrapper * sift ;


  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;

 //SIFT detector;
  
  double max_dist = 0;
  double min_dist = 1000;
  std::vector< DMatch > good_matches;
  Mat img_matches;
    
  int good_points = 10;

  double dist;

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  Mat H;
  std::vector<Point2f> obj_corners;
  std::vector<Point2f> scene_corners;


public:
  ObjectFind()
  :   nh("~"),
      it(nh)
  {
    image_sub = it.subscribe("/image", 1, &ObjectFind::image_callback, this);
    //cam_info_sub_ = nh_.subscribe("/camera_info", 1, &ObjectFind::cam_info_callback, this);

    image_pub = it.advertise("result", 1);
    debug_pub = it.advertise("debug", 1);

    nh.param<std::string>("image_path", image_path, "");

    std::cout << "image_path " << image_path << std::endl;

    
    image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE );

    if(!image.data )
    { 
      ROS_ERROR_STREAM(" --(!) Error reading images ");
    }

      //Detect the keypoints using SURF Detector
    int minHessian = 500;


    obj_corners.resize(4);
    scene_corners.resize(4);;

   // namedWindow("image_template", WINDOW_NORMAL);
    //std::cout<< image<<std::endl;
    //cv::imshow( "image_template", image );
    //cv::waitKey(100);
  	
    attendi();
    sift = SiftGPUWrapper::getInstance();

    //Calculate descriptors (feature vectors)

    sift->detect(image, keypoints_image, descriptors_image );
	std::cout << "got sift descriptors for image" << std::endl;

    namedWindow("Good Matches", WINDOW_NORMAL);

    //video_stream.setUndistortParameters(argv[2]);
    
    //-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners[0] = cvPoint(0,0); 
    obj_corners[1] = cvPoint( image.cols, 0 );
    obj_corners[2] = cvPoint( image.cols, image.rows ); 
    obj_corners[3] = cvPoint( 0, image.rows );
    attendi();

  }
  ////////////////////////////////////////////////////////////////////////////////////////////////

  void image_callback(const sensor_msgs::ImageConstPtr& msg)
  {	

  	using namespace std::chrono;
  	auto tnow = high_resolution_clock::now();
		
    ros::Time curr_stamp(ros::Time::now());
    cv_bridge::CvImagePtr cv_ptr;
      
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    frame = cv_ptr->image;


    cvtColor(frame, frame, CV_BGR2GRAY);

		sift->detect(frame, keypoints_frame, descriptors_frame );
		sift->match(descriptors_image, 128, descriptors_frame, 128, &good_matches);

		auto tpost = high_resolution_clock::now();
		std::cout << "time " << duration_cast<duration<double>>(tpost-tnow).count()*1000 << std::endl;

		/*
		for(int k = 0; k < cv::min(descriptors_image.rows-1,(int) matches.size()); k++)   
		{  
			if((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int) matches[k].size()<=2 && (int) matches[k].size()>0))  
			{  
				good_matches.push_back(matches[k][0]);  
			}  
		}  */

		std::cout << "keypoints image " << keypoints_image.size() << " keypoints frame " << keypoints_frame.size() << " matches " << good_matches.size() << std::endl;
  		
  		cv::drawMatches( image, keypoints_image, frame, keypoints_frame, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
				     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		drawKeypoints(frame, keypoints_frame, outImage);

		if (good_matches.size() >= 10){
			for( int i = 0; i < good_matches.size(); i++ ){
				obj.push_back( keypoints_image[ good_matches[i].queryIdx ].pt );
				scene.push_back( keypoints_frame[ good_matches[i].trainIdx ].pt );
			}

		H = cv::findHomography( obj, scene, CV_RANSAC );

		if(H.rows == 3 && H.cols == 3)
    {
				cv::perspectiveTransform( obj_corners, scene_corners, H);
				cv::line( img_matches, scene_corners[0] + cv::Point2f( image.cols, 0), scene_corners[1] + cv::Point2f( image.cols, 0), cv::Scalar(0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[1] + cv::Point2f( image.cols, 0), scene_corners[2] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[2] + cv::Point2f( image.cols, 0), scene_corners[3] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[3] + cv::Point2f( image.cols, 0), scene_corners[0] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
			}
		}
		cv::imshow ("key", outImage);
		cv::imshow( "Good Matches", img_matches );
		good_matches.clear();
		obj.clear();
		scene.clear();
		scene_corners.resize(4);
		char key = cv::waitKey(30);
  }
//////// END OF CLASS
};




int main(int argc,char **argv)
{
  ros::init(argc, argv, "object_find_sift_cuda");

  ObjectFind node;

  ros::spin();
}



