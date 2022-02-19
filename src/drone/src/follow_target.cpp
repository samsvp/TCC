#include "../include/drone.hpp"
#include <algorithm>    // std::find
#include "gazebo_msgs/ModelStates.h"

namespace target
{
types::waypoint lastWaypoint = {0, 0, 0, 0};
}


void targetCallback(const gazebo_msgs::ModelStates::ConstPtr& models, Drone* drone)
{
    auto it = std::find(models->name.begin(), models->name.end(), "will");
    int index = it != models->name.end() ? it - models->name.begin() : -1;
    
    geometry_msgs::Point target_position = models->pose.at(index).position;
    
    types::waypoint waypoint = {-target_position.y, target_position.x, 
        target_position.z, 0.0};

    if (target::lastWaypoint != waypoint)
    {
        drone->setDestination(waypoint);
        target::lastWaypoint = waypoint;
    }
    
    
}


int main(int argc, char** argv)
{
    //initialize ros 
	ros::init(argc, argv, "follow_node");
	ros::NodeHandle nh("~");
	
    float duration = 100000;
    Drone drone(duration, nh);

    ros::Subscriber sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, 
        boost::bind(&targetCallback, _1, &drone));
    

    ros::spin();

    return 0;
}