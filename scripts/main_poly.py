#!/usr/bin/env python
#common stuff 
from __future__ import print_function
import time as tm

from utils.common_functions import *
from utils.ros_publish import RosPub
from inverse_kinematics.inv_kinematics_pinocchio import robotKinematics
import conf
from utils.math_tools import Math
from utils.kin_dyn_utils import numericalInverseKinematics as ik
from forward_kinematics import direct_kinematics
from utils.kin_dyn_utils import fifthOrderPolynomialTrajectory as coeffTraj

if __name__ == '__main__':
    os.system("killall rosmaster rviz")
    
    # Instantiate graphic utils
    ros_pub = RosPub("giraffe")
    robot = getRobotModel("giraffe")
    kin = robotKinematics(robot, conf.frame_name)

    math = Math()

    q = conf.q0
    qd = conf.qd0
    qdd = conf.qdd0

    time = 0.0

    # Desired task space position / rotation
    p_des = np.array([1, 0, -1, 2 * np.pi/3])

    q_f, log_err, log_grad = ik(p_des, conf.q0, line_search = False, wrap = True)

    # sanity check
    # compare solution with values obtained through direct kinematics
    T_01, T_02, T_03, T_04, T_05, T_0e = direct_kinematics(q_f)
    rpy = math.rot2eul(T_0e[:3,:3])
    task_diff = p_des - np.hstack((T_0e[:3,3],rpy[0]))

    print("Desired End effector \n", p_des)
    print("Point obtained with IK solution \n", np.hstack((T_0e[:3, 3], rpy[0])))
    print("Norm of error at the end-effector position: \n", np.linalg.norm(task_diff))
    print("Final joint positions\n", q_f)

    ros_pub.publish(robot, conf.q0)
    tm.sleep(2.)
    while np.count_nonzero(q - q_f) :
        # Polynomial trajectory
        for i in range(4):
            a = coeffTraj(3.0,conf.q0[i],q_f[i])
            q[i] = a[0] + a[1]*time + a[2]*time**2 + a[3]*time**3 + a[4]*time**4 + a[5]*time**5
            qd[i] = a[1] + 2 * a[2] * time + 3 * a[3] * time ** 2 + 4 * a[4] * time ** 3 + 5 * a[5] * time ** 4
            qdd[i] = 2 * a[2] + 6 * a[3] * time + 12 * a[4] * time ** 2 + 20 * a[5] * time ** 3

        # update time
        time = time + conf.dt

        # publish joint variables
        ros_pub.publish(robot, q, qd)
        ros_pub.add_marker(p_des)
        ros.sleep(conf.dt*conf.SLOW_FACTOR)

        # stops the while loop if  you prematurely hit CTRL+C
        if ros_pub.isShuttingDown():
            print ("Shutting Down")
            break


    ros_pub.deregister_node()






