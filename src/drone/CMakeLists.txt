cmake_minimum_required(VERSION 2.8.3)
project(drone)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  mavros
  std_msgs
  geometry_msgs
  mavros_msgs
  rospy
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES iq_gnc
#  CATKIN_DEPENDS mavros std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


add_executable(drone src/drone_main.cpp)
target_link_libraries(drone ${catkin_LIBRARIES})
add_executable(square src/square.cpp)
target_link_libraries(square ${catkin_LIBRARIES})
add_executable(target src/follow_target.cpp)
target_link_libraries(target ${catkin_LIBRARIES})