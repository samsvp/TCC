#!/bin/bash
# goes to the oil position

rostopic pub -1 /go_to_pos geometry_msgs/Pose "position:
  x: -10.0
  y: 20.0
  z: 0.0
orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 0.0" 


