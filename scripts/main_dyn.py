#!/usr/bin/env python
import numpy as np
import pinocchio as pin
from pinocchio.utils import *
import time as tm
import os
from utils.common_functions import *
from utils.ros_publish import RosPub
import conf

# Main execution
if __name__ == '__main__':
    os.system("killall rosmaster rviz")
    ros_pub = RosPub("giraffe")
    robot = getRobotModel("giraffe")
    frame_id = robot.model.getFrameId(conf.frame_name)
    
    # Init variables
    zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    time = 0.0

    q = np.array([0.0, 0.0, -2., 0., 0.])
    qd = conf.qd0
    qdd = conf.qdd0

    q_des = zero
    qd_des = zero
    qdd_des = zero        # joint reference acceleration

    # get the ID corresponding to the frame we want to control
    assert(robot.model.existFrame(conf.frame_name))
    frame_ee = robot.model.getFrameId(conf.frame_name)

    error = np.array([1, 1, 1, 1, 1])

    # Main loop to simulate dynamics
    while (not ros.is_shutdown()) or any(i >= 0.01 for i in np.abs(error)):
        
        # initialize Pinocchio variables
        robot.computeAllTerms(q, qd)
        
        # vector of gravity acceleration
        g0 = np.array([0.0, 0.0, -9.81])
        
        # type of joints
        joint_types = np.array(['revolute', 'revolute', 'prismatic', 'revolute', 'revolute'])

        # gravity terms
        gp = robot.gravity(q)

        # joint space inertia with Pinocchio
        # using native function
        Mp = robot.mass(q, False)

        
        # Pinocchio bias terms (c+g)
        hp = robot.nle(q, qd, False)

        # add a damping term
        # viscous friction to stop the motion
        damping =  -20*qd

        end_stop_tau = np.zeros(5)
        
        # compute accelerations
        # Pinocchio
        qdd = np.linalg.inv(Mp).dot(end_stop_tau + damping-hp)

        # Forward Euler Integration
        qd = qd + qdd * conf.dt
        q = q + conf.dt * qd  + 0.5 * pow(conf.dt,2) * qdd

        # update time
        time = time + conf.dt
                    
        #publish joint variables
        ros_pub.publish(robot, q, qd)
        tm.sleep(conf.dt*conf.SLOW_FACTOR)
        
        # stops the while loop if  you prematurely hit CTRL+C
        if ros_pub.isShuttingDown():
            print ("Shutting Down")
            break
                
    ros_pub.deregister_node()