#include <gnc_functions.hpp>
//include API 

int main(int argc, char** argv)
{
	//initialize ros 
	ros::init(argc, argv, "gnc_node");
	ros::NodeHandle gnc_node("~");
	
	//initialize control publisher/subscribers
	gnc::init_publisher_subscriber(gnc_node);

  	// wait for FCU connection
	gnc::wait4connect();

	//wait for used to switch to mode GUIDED
	gnc::wait4start();

	//create local reference frame 
	gnc::initialize_local_frame();

	//request takeoff
	gnc::takeoff(3);

	//specify some waypoints 
	std::vector<gnc::types::waypoint> waypoints = {
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

		gnc::set_trajectory(waypoints);
		gnc::land();
	}

	return 0;
}