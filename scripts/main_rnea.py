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

    # Initialize robot state with extended configuration
    # [base_rot, shoulder_tilt, prismatic_extension, wrist_rot, wrist_tilt]
    q = np.array([0.5, -0.3, 1.0, 0.4, -0.2])     # Extended configuration (1m prismatic)
    qd = conf.qd0.copy()
    qdd = conf.qdd0.copy()

    # Target: home configuration [0, 0, 0, 0, 0]
    q_des = np.zeros(5)    # Home position
    qd_des = conf.qd0.copy()    # Zero velocity at target
    qdd_des = conf.qdd0.copy()  # Zero acceleration at target

    math_utils = Math()

    t = 0.

    frame_id = model.getFrameId(conf.frame_name)

    simulator = DynamicSimulator(robot, ros_pub)
    logs = simulator.simulate(q, qd, qdd, q_des, qd_des, qdd_des)

    ros_pub.deregister_node()

    print("\nGenerating plots...")

    # Create plots directory in parent folder if it doesn't exist
    import os
    os.makedirs('../plots', exist_ok=True)

    # Close any existing figures first
    plt.close('all')

    # Create plots
    joint_names = ['Base Rot', 'Shoulder', 'Prismatic', 'Wrist1', 'Wrist2']

    plotJoint('position', logs['time'], logs['q'], logs['q_des'],
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'],
            joint_names=joint_names, title='Joint Positions - Gravity Compensation Control')
    plt.gcf().set_size_inches(14, 10)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.92)
    plt.savefig('../plots/gravity_compensation_positions.png', dpi=300, bbox_inches='tight')

    plotJoint('velocity', logs['time'], logs['q'], logs['q_des'],
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'],
            joint_names=joint_names, title='Joint Velocities - Gravity Compensation Control')
    plt.gcf().set_size_inches(14, 10)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.92)
    plt.savefig('../plots/gravity_compensation_velocities.png', dpi=300, bbox_inches='tight')

    plotJoint('acceleration', logs['time'], logs['q'], logs['q_des'],
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'],
            joint_names=joint_names, title='Joint Accelerations - Gravity Compensation Control')
    plt.gcf().set_size_inches(14, 10)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.92)
    plt.savefig('../plots/gravity_compensation_accelerations.png', dpi=300, bbox_inches='tight')

    plotJoint('torque', logs['time'], logs['q'], logs['q_des'],
            logs['qd'], logs['qd_des'], logs['qdd'], logs['qdd_des'], logs['tau'],
            joint_names=joint_names, title='Joint Torques - Gravity Compensation Control')
    plt.gcf().set_size_inches(14, 10)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.92)
    plt.savefig('../plots/gravity_compensation_torques.png', dpi=300, bbox_inches='tight')

    print("Plots saved to ../plots/ folder")

    plt.show(block=False)

    print("\nGravity compensation simulation completed!")
    print(f"Started from: {q} (extended configuration)")
    print(f"Ended at: {logs['q'][:, -1]} (final configuration)")
    print(f"Target was: {q_des} (home configuration)")

    input("Press Enter to close all figures...")

    [plt.close(fig) for fig in plt.get_fignums()]  # Close all figures
