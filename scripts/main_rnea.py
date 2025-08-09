from utils.ros_publish import RosPub
from utils.common_functions import *
from utils.inv_kinematics_pinocchio import robotKinematics
from utils.math_tools import Math

import conf as conf

from functions.dyn import DynamicSimulator

ros_pub = RosPub("giraffe")
robot = getRobotModel("giraffe")
data = robot.data
model = robot.model

kin = robotKinematics(robot, conf.frame_name)

# Initialize robot state
q = conf.q0.copy()
qd = conf.qd0.copy()
qdd = conf.qdd0.copy()

q_des = conf.q0.copy() 
qd_des = conf.qd0.copy()
qdd_des = conf.qdd0.copy()

math_utils = Math()

t = 0.

frame_id = model.getFrameId(conf.frame_name)

simulator = DynamicSimulator(robot, ros_pub)
logs = simulator.simulate(q, qd, qdd, q_des, qd_des, qdd_des)

ros_pub.deregister_node()
