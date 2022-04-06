# TCC

## Proposal
This project aims to create an autonomous Drone for oil spill detection and tracking.
The drone must be able to do the following:
 - Oil Spill Detection
 - Oil Spill Tracking
 - Oil Spill surface area calculation
 - Oil Spill volume estimation

The project will have a simulation with the following features:
 - Drone's trajectory system
 - Drone patrol's state machine
 - Tracking algorithm testing

## To Do
- Oil segmentation tracking
- Add oil to simulation
- Research termal sensors with less than 1 degree celcius resolution

## Running
```shell
. devel/setup.bash
roslaunch iq_sim droneOnly.launch
# New Terminal
. devel/setup.bash
src/iq_sim/scripts/startsitl.sh # MAVProxy terminal
# New Terminal
. devel/setup.bash
roslaunch iq_sim apm.launch
# New Terminal 
. devel/setup.bash
rosrun drone square
```

In the MAVProxy terminal run
```shell
mode guided 
```

## Dependencies
[Ardupilot](https://github.com/ArduPilot/ardupilot) (follow 
    [this link](https://github.com/Intelligent-Quads/iq_tutorials/blob/master/docs/Installing_Ardupilot_20_04.md) for instructions.)

[QGroundControl](https://github.com/Intelligent-Quads/iq_tutorials/blob/master/docs/installing_qgc.md)
    (Optional)

[Gazebo](http://www.gazebosim.org/tutorials?tut=install_ubuntu) (follow [this link](https://github.com/Intelligent-Quads/iq_tutorials/blob/master/docs/installing_gazebo_arduplugin.md) to get drone  and world models.)

[ROS](http://wiki.ros.org/noetic/Installation/Ubuntu)

[MavROS](http://wiki.ros.org/mavros) (follow [this link](https://github.com/Intelligent-Quads/iq_tutorials/blob/master/docs/installing_ros_20_04.md) for better instructions.)

[asv wave sim vrx](https://github.com/srmainwaring/asv_wave_sim) wave models (switch to development branch).
```shell
wstool init src

rosinstall_generator --upstream mavros | tee /tmp/mavros.rosinstall
rosinstall_generator mavlink | tee -a /tmp/mavros.rosinstall
wstool merge -t src /tmp/mavros.rosinstall
wstool update -t src

rosdep install --from-paths src --ignore-src --rosdistro `echo $ROS_DISTRO` -y
```

[Inteligent Quads GNC ROS](https://github.com/Intelligent-Quads/iq_gnc)

## Resources
Simulation and Drone Control:
[IQ Tutorials](https://github.com/Intelligent-Quads/iq_tutorials)

[IQ Sim](https://github.com/Intelligent-Quads/iq_sim)
