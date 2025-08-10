from utils.ros_publish import RosPub
from utils.common_functions import *
from utils.inv_kinematics_pinocchio import robotKinematics
from utils.math_tools import Math

import conf as conf

from functions.dynamic_simulator import DynamicSimulator

if __name__ == "__main__":
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

    print("\nGenerating plots...")

    plt.figure(1)
    plotJoint('position', logs['time'], logs['q'], logs['q_des'], 
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'])

    plt.figure(2)
    plotJoint('velocity', logs['time'], logs['q'], logs['q_des'], 
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'])
    plt.figure(3)
    plotJoint('acceleration', logs['time'], logs['q'], logs['q_des'], 
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'])

    plt.figure(4)
    plotJoint('torque', logs['time'], logs['q'], logs['q_des'], 
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'])

    plt.show()
    input("Press Enter to continue...")
    [plt.close(fig) for fig in plt.get_fignums()]  # Close all figures
