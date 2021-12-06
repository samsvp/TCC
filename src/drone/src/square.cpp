#include "../include/drone.hpp"
#include <vector>


int main(int argc, char** argv)
{
	//initialize ros 
	ros::init(argc, argv, "gnc_node");
	ros::NodeHandle gnc_node("~");
	
    float duration = 100000;

    Drone drone(duration, gnc_node);

	//specify some waypoints 
	std::vector<types::waypoint> waypoints = {
		{0, 0, 3, 0},
		{5, 0, 3, -90},
		{5, 5, 3, 0},
		{0, 5, 3, 90},
		{0, 0, 3, 180},
		{0, 0, 3, 0}
	};


	//specify control loop rate. We recommend a low frequency to not over load the FCU with messages. Too many messages will cause the drone to be sluggish
	ros::Rate rate(2.0);
	int counter = 0;
	
	while(ros::ok())
	{
		ros::spinOnce();
		rate.sleep();

		drone.setTrajectory(waypoints);
		drone.land();
	}

	return 0;
}