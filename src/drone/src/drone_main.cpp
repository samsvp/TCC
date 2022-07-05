#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "../include/drone.hpp"


/*
 * Main class for controlling the drone within the simulation 
 */
class DroneMain
{
public:
    DroneMain(float battery_duration, ros::NodeHandle& control_node,
            std::vector<types::waypoint> main_trajectory, float h) : 
            drone(battery_duration, control_node, h),
            main_trajectory(main_trajectory), altitude(h)
    {
        this->go_to_pos_sub = control_node.subscribe(
            "/go_to_pos", 1, &DroneMain::goToPosition, this);
        
        this->camera_image_sub = control_node.subscribe(
            "/webcam/image_raw", 1, &DroneMain::imageCallback, this);

        this->oil_detected = false;
        this->break_trajectory = false;

        // start main loop
        this->waypointLoop();

    }
    ~DroneMain() { }


    /*
     * Goes to the given position
     */
    void goToPosition(const geometry_msgs::Pose::ConstPtr& pose)
    {
        if (this->break_trajectory) return;

        this->break_trajectory = true;
        
        // pause trajectory to go to new target
        this->drone.pauseTrajectory();
        float psi = this->quaternionToHeading(pose->orientation);
        // we ignore the altitude and fly on the default altitude
        // this is useful for measurement
        std::vector<types::waypoint> waypoints = {
            { pose->position.x, pose->position.y, this->altitude, psi }
        };
        this->drone.setTrajectory(waypoints);
        // resume trajectory should be based on state machine
        this->drone.resumeTrajectory();

        this->break_trajectory = false;
    }


    /**
     * @brief Callback which receives the messages from the drone camera
     * 
     * @param msg 
     */
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        static bool first_image = false;

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

        if (!first_image) 
        {
            this->calibrateCamera();
            first_image = true;
        }
    }


private:
    Drone drone; 
    ros::Subscriber go_to_pos_sub;
    ros::Subscriber camera_image_sub;
    cv::Mat curr_image;

    // how many meters equals 1 pixel
    float pixel_size;
    // the altitude of the drone
    float altitude;

    // hold if oil has been detected on the given frame
    bool oil_detected;
    // holds if the drone should break the main trajectory
    bool break_trajectory;
    // holds the main waypoints that the drone should patrol
    std::vector<types::waypoint> main_trajectory;


    /**
     * @brief Makes the drone follow the main trajectory loop
     */
    void waypointLoop()
    {
        while (ros::ok())
        {
            if (!this->break_trajectory)
                this->drone.setTrajectory(this->main_trajectory);
            else
                ros::Duration(0.01).sleep();
        }
    }


    void calibrateCamera()
    {
        cv::Mat res;
        // get white box. res will be WxHx3
        cv::inRange(this->curr_image, cv::Scalar(200, 200, 200),
            cv::Scalar(255, 255, 255), res);

        // sums over all white pixels. divide by 255 as cv::inRange
        // sets as 255 everything in range
        int area = cv::sum(res)[0] / 255;
        // the pixel size will be the square root of the area of the square
        float size = std::sqrt(area);
        // the square has a real world length of 1x1, so 1m/pixel_size = size/1px
        this->pixel_size = 1.0 / size;
        
        ROS_INFO("1px equals to %f m", this->pixel_size);
    }


    /**
     * Transforms the given quaternion into the drone heading
     * @param q 
     * @return float 
     */
    float quaternionToHeading(geometry_msgs::Quaternion q)
    {
        return std::atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(std::pow(q.y,2) + std::pow(q.z,2))
        );
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "gnc_node");
	ros::NodeHandle nh;

    float h = 15;

    // main patrol
    std::vector<types::waypoint> waypoints = {
		{0, 0, h, 0},
		{5, 0, h, -90},
		{5, 5, h, 0},
		{0, 5, h, 90},
		{0, 0, h, 180},
		{0, 0, h, 0}
	};


    DroneMain drone_main(10000, nh, waypoints, h);
    ros::spin();

    return 0;
}