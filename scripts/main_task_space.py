from utils.ros_publish import RosPub
from utils.common_functions import *
from utils.inv_kinematics_pinocchio import robotKinematics
from utils.math_tools import Math
import pinocchio as pin

import conf as conf

from functions.task_space_controller import TaskSpaceController

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

    q_home = conf.q0.copy()
    qd_sim = np.zeros(robot.nv)
    pitch_des = np.radians(conf.pitch_des)
    
    controller = TaskSpaceController(robot, model, data, ros_pub)
    q_final, qd_final, logs = controller.simulate(pitch_des_final=pitch_des)  # desired pitch in radians

    # Update forward kinematiccs 
    pin.forwardKinematics(model, data, q_final, qd_final)
    pin.updateFramePlacement(model, data, frame_id)

    # Print final pose
    pos_final = data.oMf[frame_id].translation
    rpy_final = pin.rpy.matrixToRpy(data.oMf[frame_id].rotation)

    # Calculate pitch using same method as controller
    R_world = data.oMf[frame_id].rotation
    ee_z_axis = R_world[:, 2]
    pitch_final_cartesian = np.arctan2(-ee_z_axis[2], np.sqrt(ee_z_axis[0]**2 + ee_z_axis[1]**2))

    print("\nFinal Position of the End Effector (m):", pos_final)
    print("Final Orientation of the End-Effector (RPY - deg):", np.degrees(rpy_final))
    print("Pitch final (deg):", np.degrees(pitch_final_cartesian))
    print("Pitch desired (deg):", np.degrees(pitch_des))

    ros_pub.deregister_node()

    print("\nGenerating plots...")

    plt.figure(figsize=(12, 10))
    plt.suptitle("Task Space Control Performance")
    
    # Position plots
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(4, 1, i+1)
        plt.plot(logs['time'], logs['p'][i], label=f'Actual {labels[i]}')
        plt.plot(logs['time'], logs['p_des'][i], '--', label=f'Desired {labels[i]}')
        plt.ylabel(f'Position {labels[i]} [m]')
        plt.legend()
        plt.grid(True)

    # Pitch plot
    plt.subplot(4, 1, 4)
    plt.plot(logs['time'], np.degrees(logs['pitch']), label='Actual')
    plt.plot(logs['time'], np.degrees(logs['pitch_des']), '--', label='Desired')
    plt.ylabel('Pitch [deg]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('task_space_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'task_space_results.png'")
    # plt.show()  # Commented out for automated runs
    [plt.close(fig) for fig in plt.get_fignums()]  # Close all figures
