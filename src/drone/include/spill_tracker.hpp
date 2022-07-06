#pragma once

#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <geometry_msgs/Point.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

#include "tracker.hpp"


class SpillTracker
{
public:
    SpillTracker(ros::NodeHandle& control_node);
    ~SpillTracker();

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
    Tracker tracker;
    
    cv::Mat curr_image;
    ros::Subscriber camera_image_sub;
    ros::Publisher centroid_pos_pub;
    ros::Publisher pub_img;

    /**
     * @brief Get the oil mask of the given image
     * 
     * @param image 
     * @return cv::Mat 
     */
    cv::Mat getMask(cv::Mat image);
    /**
     * @brief Draws the contours of the given mask and its centroid
     * 
     * @param src 
     * @param contours 
     * @param hierarchy 
     * @param mc 
     * @return cv::Mat 
     */
    cv::Mat drawContours(cv::Mat src, 
        std::vector<std::vector<cv::Point>> contours,
        std::vector<cv::Vec4i> hierarchy, std::vector<cv::Point2f> mc);
    /**
     * @brief Helper to publish an image
     * 
     * @param image 
     */
    void publishImage(cv::Mat image);
};


SpillTracker::SpillTracker(ros::NodeHandle& control_node)
{
    this->camera_image_sub = control_node.subscribe(
            "/webcam/image_raw", 1, &SpillTracker::imageCallback, this);
    
    this->centroid_pos_pub = control_node.advertise<geometry_msgs::Point>(
        "/centroid_pred", 1);
    this->pub_img = control_node.advertise<sensor_msgs::Image>("/mask", 1);
}


SpillTracker::~SpillTracker() {  }


cv::Mat SpillTracker::getMask(cv::Mat image)
{
    cv::Mat res;
    // get pink spill. res will be WxHx3
    cv::inRange(this->curr_image, cv::Scalar(100, 0, 100),
        cv::Scalar(255, 255, 255), res);

    return res;
}


cv::Mat SpillTracker::drawContours(cv::Mat src, 
        std::vector<std::vector<cv::Point>> contours,
        std::vector<cv::Vec4i> hierarchy, std::vector<cv::Point2f> mc)
{
    // draw contours
    cv::Mat drawing(src.size(), CV_8UC3, cv::Scalar(255,255,255));
    for(int i = 0; i<contours.size(); i++)
    {
        auto color = tracker.get_color(i);
        cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
        cv::circle( drawing, mc[i], 4, color, -1, 8, 0 );
    }

    return drawing;
}


void SpillTracker::publishImage(cv::Mat image)
{
    static int counter=0;
    sensor_msgs::Image img_msg; // >> message to be sent
    std_msgs::Header header; // empty header
    header.seq = ++counter; // user defined counter
    header.stamp = ros::Time::now(); // time

    auto img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, image);
    img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image
    pub_img.publish(img_msg); 
}


void SpillTracker::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    
    // image conversion
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    this->curr_image = cv_ptr->image;

    cv::Mat mask = this->getMask(this->curr_image);

    // if there is no oil, then just return
    if (cv::sum(mask)[0] == 0) return;

    // make mask a 3 channel image
    std::vector<cv::Mat> vChannels;
    for (unsigned int c = 0; c < 3; c++)
    {
        vChannels.push_back(mask);
    }
    cv::Mat mask3channels;
    cv::merge(vChannels, mask3channels);

    this->tracker.update(mask3channels);
    std::vector<cv::Point2f> mc = tracker.get_centers();

    if (mc.empty()) return;
    
    // publish point
    geometry_msgs::Point p;
    p.x = mc[0].x;
    p.y = mc[0].y;
    p.z = 0;
    this->centroid_pos_pub.publish(p);

    // publish contour image
    auto contours = tracker.get_countours();
    auto hierarchy = tracker.get_hierarchy();
    cv::Mat contour_image = this->drawContours(mask3channels, contours, hierarchy, mc);

    this->publishImage(contour_image);
}