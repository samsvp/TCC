#%%
import cv2
import numpy as np

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from typing import Tuple


def calibrate(image: np.ndarray) -> float:
    """
    Returns how many meters 1 pixel is
    """
    # creates mask with the white square
    mask = cv2.inRange(image, np.array([0, 0, 0]),
        np.array([150, 150, 150]))

    # the square has sides of 1m
    p_size = np.sqrt(mask.sum() / 255)
    pixel_size = 1 / p_size
    return pixel_size


class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        
        self.cx_pub = rospy.Publisher(
            "/truth/centroid_x", Float32, queue_size=10)
        self.cy_pub = rospy.Publisher(
            "/truth/centroid_y", Float32, queue_size=10)
        self.area_pub = rospy.Publisher(
            "/truth/spill_size", Float32, queue_size=10)
        
        self.image_sub = rospy.Subscriber("/sky_camera/sky_image",
            Image,self.callback)

        self.image: np.ndarray # the current image
        self.pixel_size: float # how many meters 1 pixel is 
        self.first_image = True # check if it is the first image received


    def get_spill_info(self, image: np.ndarray) -> \
            Tuple[float, Tuple[float, float]]:
        """
        Returns the area of the spill and its centroid,
        both in meters 
        """
        # get mask from spill
        mask = cv2.inRange(image, np.array([110, 100, 100]),
            np.array([255, 255, 255]))

        area_px = mask.sum() // 255
        area_meters = area_px * self.pixel_size

        (cx, cy) = self.calc_centroid(mask)
        # divide by 2 and by 4 because the origin is at 
        # the middle (i dunno where the 4 came from). 
        # we also have a little offset on the y axis
        centroid = (cx * self.pixel_size / 4, 
            cy * self.pixel_size / 2 - 2)

        return (area_meters, centroid)



    def calc_centroid(self, image_mask: np.ndarray) -> Tuple[int, int]:
        """
        Gives the coordinates of the image centroid in the world
        """
        m = cv2.moments(image_mask)

        # calculate x,y coordinate of center
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        return (cx, cy)


    def callback(self,data):
        try:
            self.image = np.fliplr(
                self.bridge.imgmsg_to_cv2(data, "bgr8")
            )
        except CvBridgeError as e:
            print(e)
            return

        if self.first_image:
            self.first_image = False
            self.pixel_size = calibrate(self.image)


        spill_area, centroid = self.get_spill_info(self.image)
        
        self.cx_pub.publish(Float32(centroid[0]))
        self.cy_pub.publish(Float32(centroid[1]))
        self.area_pub.publish(Float32(spill_area))
        
        print(f"Spill area = {spill_area}, centroid = {centroid}")


if __name__ == '__main__':
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# %%