#!/bin/bash
# goes to the oil position

# when published the drone position is:
# x: -9.098373 y: 20.425953 z: 15.298266
# so we have an offset to account for
rostopic pub -1 /go_to_pos geometry_msgs/Pose "position:
  x: -10.0
  y: 20.0
  z: 0.0
orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 0.0" 


