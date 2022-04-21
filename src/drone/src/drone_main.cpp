#include <vector>

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>


#include "../include/drone.hpp"


/*
 * Main class for controlling the drone within the simulation 
 */
class DroneMain
{
public:
    DroneMain(float battery_duration, ros::NodeHandle& control_node,
            std::vector<types::waypoint> main_trajectory) : drone(battery_duration, control_node),
            main_trajectory(main_trajectory)
    {
        this->go_to_pos_sub = control_node.subscribe(
            "/go_to_pos", 1, &DroneMain::goToPosition, this);

        this->break_trajectory = false;
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
        
        this->drone.pauseTrajectory();
        float psi = this->quaternionToHeading(pose->orientation);
        std::vector<types::waypoint> waypoints = {
            { pose->position.x, pose->position.y, pose->position.z, psi }
        };
        this->drone.setTrajectory(waypoints);
        this->drone.resumeTrajectory();

        this->break_trajectory = false;
    }


private:
    Drone drone; 
    ros::Subscriber go_to_pos_sub;

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

    std::vector<types::waypoint> waypoints = {
		{0, 0, 3, 0},
		{5, 0, 3, -90},
		{5, 5, 3, 0},
		{0, 5, 3, 90},
		{0, 0, 3, 180},
		{0, 0, 3, 0}
	};


    DroneMain drone_main(10000, nh, waypoints);
    ros::spin();

    return 0;
}