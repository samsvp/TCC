#include "../include/spill_tracker.hpp"


int main(int argc, char** argv)
{
    ros::init(argc, argv, "spill_node");
	ros::NodeHandle nh;

    SpillTracker spill_tracker(nh);
    ros::spin();

    return 0;
}