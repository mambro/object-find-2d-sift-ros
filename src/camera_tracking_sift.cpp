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
  
  image_transport::Publisher image_pub;
  image_transport::Publisher debug_pub;
  
  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber image_sub;
  std::string image_path;

  std::vector<KeyPoint> keypoints_image, keypoints_frame;
  Mat descriptors_image, descriptors_frame;

  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;

  SIFT detector;
  
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
    scene_corners.resize(4);
  
    //Calculate descriptors (feature vectors)

    detector( image, Mat(),  keypoints_image, descriptors_image);

    namedWindow("Good Matches", WINDOW_NORMAL);

    //video_stream.setUndistortParameters(argv[2]);
    
    //-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners[0] = cvPoint(0,0); 
    obj_corners[1] = cvPoint( image.cols, 0 );
    obj_corners[2] = cvPoint( image.cols, image.rows ); 
    obj_corners[3] = cvPoint( 0, image.rows );

  }
  ////////////////////////////////////////////////////////////////////////////////////////////////

  void image_callback(const sensor_msgs::ImageConstPtr& msg)
  {
      ros::Time curr_stamp(ros::Time::now());
      cv_bridge::CvImagePtr cv_ptr;
      
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        frame = cv_ptr->image;


        cvtColor(frame, frame, CV_BGR2GRAY);
  
        detector( frame, Mat(),  keypoints_frame, descriptors_frame);
    
        //Matching descriptor vectors using FLANN matcher
  
        matcher.match( descriptors_image, descriptors_frame, matches);

            //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_image.rows; i++ )
        {
          dist = matches[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
        }
      
        for( int i = 0; i < descriptors_image.rows; i++ )
        {
          if( matches[i].distance <= max(2*min_dist, 0.1))
            good_matches.push_back( matches[i]); 
        }
        //-- Draw only "good" matches
        drawMatches( image, keypoints_image, frame, keypoints_frame, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                   std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        std::cout << "good_points: " << good_points << std::endl;


        if (good_matches.size() >= good_points){     
          good_points += 2; 

          for( int i = 0; i < good_matches.size(); i++ ){
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_image[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_frame[ good_matches[i].trainIdx ].pt );
          }

          H = findHomography( obj, scene, CV_RANSAC );

          std::cout << "H rows " << H.rows << " H columns" << H.cols << std::endl;
          //attendi();

          if(H.rows == 3 && H.cols == 3)
          {
            perspectiveTransform( obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line( img_matches, scene_corners[0] + Point2f( image.cols, 0), scene_corners[1] + Point2f( image.cols, 0), Scalar(0, 255, 0), 4 );
            line( img_matches, scene_corners[1] + Point2f( image.cols, 0), scene_corners[2] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[2] + Point2f( image.cols, 0), scene_corners[3] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[3] + Point2f( image.cols, 0), scene_corners[0] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
          }
        }
        else if (good_points > 6)
          good_points--;


        
        //-- Show detected matches
        imshow( "Good Matches", img_matches );
        matches.clear();
        good_matches.clear();
        obj.clear();
        scene.clear();
        scene_corners.resize(4);
        char key = waitKey(10);

      
     
  }
//////// END OF CLASS
};




int main(int argc,char **argv)
{
  ros::init(argc, argv, "object_find");

  ObjectFind node;

  ros::spin();
}
