import rospy
from one_robot_one_obs.msg import relDist
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import time
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

digit = 0
status = 0

velocity_publisher = rospy.Publisher('/nav/cmd_vel', Twist, queue_size=10)

w = 640
h = 480

kp = 0.001

prev_gray = np.zeros((w,h),dtype=np.uint8)
gray = np.zeros((w,h),dtype=np.uint8)

def set_vel(v,q):
    vel_msg = Twist()
    (vel_msg.linear.x, vel_msg.linear.y, vel_msg.linear.z) = (v, 0, 0)
    (vel_msg.angular.x, vel_msg.angular.y, vel_msg.angular.z) = (0, 0, q)
    velocity_publisher.publish(vel_msg)

def goTotarget(digit,obs_ids,centerpoints):
    global w
    global kp
    try:
        idx = obs_ids.index(digit)
        cp = centerpoints[idx]
        x = cp[0]
        y = cp[1]
        q = kp*(x-w/2.0)
        set_vel(0.1,-q)
    except TypeError:
        set_vel(0,0)

def findTargets(current_frame):
    global obs_ids
    global centerpoints
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    markers, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=parameters)
    obs_ids = []
    centerpoints = []
    if ids is not None:
        for i, corners in enumerate(markers):
            obs_ids.append(ids[i][0])
            centerpoint = np.average(np.matrix(corners[0]), axis=0)
            x = centerpoint.tolist()[0][0]
            y = centerpoint.tolist()[0][1] 
            centerpoints.append([x,y])
    else:
        set_vel(0,0)
        print("no target detected")

def callback(data):
    global obs_ids
    global centerpoints
    digit = int(data.xrel)
    status = int(data.yrel)
    vel_msg = Twist()
    if status == 0:
        print("no digit received")
        set_vel(0,0)
    else:
        [v,w] = goTotarget(digit,obs_ids,centerpoints)
        print("digit received :",digit)
        set_vel(0.1,0)

def image_callback(data):
    br = CvBridge()
    print("receiving video frame")
    try:
        current_frame = br.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    try:
        findTargets(current_frame)
    except KeyboardInterrupt:
        pass

def stopNow():
    vel_msg = Twist()
    (vel_msg.linear.x, vel_msg.linear.y, vel_msg.linear.z) = (0, 0, 0)
    (vel_msg.angular.x, vel_msg.angular.y, vel_msg.angular.z) = (0, 0, 0)
    velocity_publisher.publish(vel_msg)
    rospy.loginfo("node terminated")
    rospy.signal_shutdown("target reached")

if __name__ == "__main__":
    rospy.init_node('target_track',anonymous=True)
    rospy.Subscriber('/camera/color/image_raw/', Image, image_callback)
    rospy.Subscriber('target_num', relDist, callback)
    rospy.spin()


