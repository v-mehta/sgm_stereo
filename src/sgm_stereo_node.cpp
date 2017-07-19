/*
    Copyright (C) 2017  Vaibhav Mehta

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


// C++ Includes
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <stack>
#include <ctime>

// ROS Includes
#include <ros/ros.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/stereo_camera_model.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <stereo_msgs/DisparityImage.h>

// OpenCV Includes
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <png++/png.hpp>

#include "sgm_stereo/SGMStereo.h"

using std::vector;

// Just a timer utility
std::stack<clock_t> tictoc_stack;
void tic()
{
  tictoc_stack.push(clock());
}
void toc(std::string s)
{
  ROS_INFO_STREAM("Time taken by routine : "<<s<< " "
                  <<((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC);
  tictoc_stack.pop();
}
//

namespace stereo_sgm
{

class StereoSGMNode
{
public:
  StereoSGMNode();

  void callback(const sensor_msgs::ImageConstPtr& left,
                const sensor_msgs::ImageConstPtr& right,
                const sensor_msgs::CameraInfoConstPtr& left_info,
                const sensor_msgs::CameraInfoConstPtr& right_info);

  void getOpenCVImage(const sensor_msgs::ImageConstPtr& ros_img, cv::Mat& opencv_img);

  void convertCvMatToPngImage(const cv::Mat& cvmat_image,
                            png::image<png::rgb_pixel>& png_image);

  void convertOpenCV2ROSImage(const cv::Mat& opencv_img,
                              const std::string& image_encoding,
                              sensor_msgs::Image& ros_img);

protected:
  ros::NodeHandle nh_;
  std::string image_encoding_;

  // Sync Policy for Images
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image,
    sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ImageSyncPolicy;

  typedef message_filters::Synchronizer<ImageSyncPolicy> Synchronizer;
  boost::shared_ptr<Synchronizer> sync_;

  message_filters::Subscriber<sensor_msgs::Image> left_image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> right_image_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> right_info_sub_;

  image_geometry::StereoCameraModel model_;

  SGMStereo sgm;
  ros::Publisher pub_disparity_;
};

StereoSGMNode::StereoSGMNode() :
  left_image_sub_(nh_, "left/image_rect", 10),
  right_image_sub_(nh_, "right/image_rect", 10),
  left_info_sub_(nh_, "left/camera_info", 10),
  right_info_sub_(nh_, "right/camera_info", 10)
{
  sync_.reset(new Synchronizer(ImageSyncPolicy(10), left_image_sub_, right_image_sub_, left_info_sub_, right_info_sub_)),
  sync_->registerCallback(boost::bind(&StereoSGMNode::callback, this, _1, _2, _3, _4));

  pub_disparity_ = nh_.advertise<stereo_msgs::DisparityImage>("disparity", 1);
}

void StereoSGMNode::getOpenCVImage(const sensor_msgs::ImageConstPtr& ros_img, cv::Mat& opencv_img)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(ros_img, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  opencv_img = cv_ptr->image;
}

void StereoSGMNode::convertOpenCV2ROSImage( const cv::Mat& opencv_img,
                                            const std::string& image_encoding,
                                            sensor_msgs::Image& ros_img)
{
  cv_bridge::CvImage cv_bridge_img;
  cv_bridge_img.encoding = image_encoding;
  cv_bridge_img.image = opencv_img;

  cv_bridge_img.toImageMsg(ros_img);
}

void StereoSGMNode::convertCvMatToPngImage(const cv::Mat& cvmat_image,
                            png::image<png::rgb_pixel>& png_image)
{
  for (size_t y = 0; y < png_image.get_height(); ++y)
  {
    for (size_t x = 0; x < png_image.get_width(); ++x)
    {
      png_image[y][x] = png::rgb_pixel( cvmat_image.at<cv::Vec3b>(y,x)[2],
                                        cvmat_image.at<cv::Vec3b>(y,x)[1],
                                        cvmat_image.at<cv::Vec3b>(y,x)[0]);
    }
  }
}

void StereoSGMNode::callback(const sensor_msgs::ImageConstPtr& left_ros_img,
                              const sensor_msgs::ImageConstPtr& right_ros_img,
                              const sensor_msgs::CameraInfoConstPtr& left_info,
                              const sensor_msgs::CameraInfoConstPtr& right_info)
{
  // update the camera info. Do it once as it's only needed once -
  model_.fromCameraInfo(left_info, right_info);

  cv::Mat left_opencv_img_color_, right_opencv_img_color_;
  getOpenCVImage(left_ros_img, left_opencv_img_color_);
//  cv::imshow("left_rect_image", left_opencv_img_color_);
  getOpenCVImage(right_ros_img, right_opencv_img_color_);
//  cv::imshow("right_rect_image", right_opencv_img_color_);

  png::image<png::rgb_pixel> left_png_img(left_opencv_img_color_.cols, left_opencv_img_color_.rows);
  convertCvMatToPngImage(left_opencv_img_color_, left_png_img);

  png::image<png::rgb_pixel> right_png_img(right_opencv_img_color_.cols, right_opencv_img_color_.rows);
  convertCvMatToPngImage(right_opencv_img_color_, right_png_img);

  float disparityImage[left_opencv_img_color_.cols * left_opencv_img_color_.rows];

  tic();
  sgm.compute(left_png_img, right_png_img, disparityImage);
  toc("sgm.compute");
  cv::Mat disparity(left_opencv_img_color_.rows, left_opencv_img_color_.cols, CV_32F, disparityImage);

  cv::Mat disparity_uc;
  disparity.convertTo(disparity_uc, CV_8UC1);
//  cv::imshow("disparity", disparity_uc);
//  cv::waitKey(1);

  // Allocate new disparity image message
  stereo_msgs::DisparityImagePtr disp_msg  = boost::make_shared<stereo_msgs::DisparityImage>();
  disp_msg->header            = left_ros_img->header;

  disp_msg->f                 = model_.left().fx();
  disp_msg->T                 = model_.baseline();
  disp_msg->min_disparity     = 0.0;
  disp_msg->max_disparity     = 127.0;
  disp_msg->delta_d           = 0.0625;

  disp_msg->image.header      = left_ros_img->header;
  disp_msg->image.height      = left_opencv_img_color_.rows;
  disp_msg->image.width       = left_opencv_img_color_.cols;

  convertOpenCV2ROSImage(disparity, sensor_msgs::image_encodings::TYPE_32FC1, disp_msg->image);

  pub_disparity_.publish(disp_msg);
}

}  // namespace fdcm

int main(int argc, char** argv)
{
  ros::init(argc, argv, "StereoSGMNode");
  stereo_sgm::StereoSGMNode node;

  ros::spin();
  return EXIT_SUCCESS;
}
