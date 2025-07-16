import numpy as np
import math
from utils.math_tools import Math


def setRobotParameters():
    """
    Define and return all key geometric and inertial parameters of the giraffe robot.

    Returns:
      lengths: np.array of link offsets [l1..l6]
      inertia_tensors: np.array of 7 (7 links) 3×3 inertia matrices around each COM
      link_masses: np.array of masses for links 0..6
      coms: np.array of COM positions in each link's local frame
    """
    # Link offsets along the robot's vertical (-Z) axis (meters):
    l1 = 0.1    # from ceiling mount (base_link) to shoulder joint
    l2 = 0.075  # from shoulder joint to upper_arm joint
    l3 = 0.15   # from upper_arm to start of telescopic extension
    l4 = 0.25   # full stroke of the telescopic link
    l5 = 0.04   # between wrist_1 and wrist_2 joints
    l6 = 0.03   # from wrist_2 to end-effector
    lengths = np.array([l1, l2, l3, l4, l5, l6])

    # Link masses (kg): base (0) through end-effector (6)
    link_masses = np.array([5.0, 3.0, 4.0, 2.0, 1.0, 0.5, 0.2])

    # Center-of-mass positions in link frames (all at origin here for simplicity)
    coms = np.zeros((7, 3))

    # Diagonal inertia tensors (kg·m²) about each COM
    inertia_tensors = np.array([
        np.diag([0.02,   0.02,   0.025]),   # base_link
        np.diag([0.015,  0.015,  0.0096]),   # shoulder_link
        np.diag([0.03,   0.03,   0.0096]),   # upper_arm_link
        np.diag([0.042,  0.042,  0.0016]),   # telescopic_link
        np.diag([0.0015, 0.0015, 0.0006]),   # wrist_1_link
        np.diag([0.0006, 0.0006, 0.0002]),   # wrist_2_link
        np.diag([0.0002, 0.0002, 0.00004])   # ee_link
    ])

    return lengths, inertia_tensors, link_masses, coms


def direct_kinematics(q):
    """
    Compute the chain of homogeneous transforms up to the end-effector.

    Args:
      q (length-5 array): [q1, q2, d3, q4, q5]
        q1: base rotation about Z
        q2: shoulder tilt about Y
        d3: telescopic extension along -Z
        q4: wrist_1 rotation about Z
        q5: wrist_2 rotation about Y

    Returns:
      list of 6 np.ndarray (4x4): T01, T02, T03, T04, T05, T0e
    """
    # Load link offsets
    lengths, _, _, _ = setRobotParameters()
    l1, l2, l3, l4, l5, l6 = lengths
    q1, q2, d3, q4, q5 = q

    # 1) Base_link -> Joint1 (rotation q1 about Z)
    #    a) translate down by l1 along robot Z
    T01_trans = np.array([[1,0,0, 0],
                          [0,1,0, 0],
                          [0,0,1,-l1],
                          [0,0,0, 1]])
    #    b) rotate about Z by q1
    Rz1 = np.array([[ math.cos(q1), -math.sin(q1), 0, 0],
                    [ math.sin(q1),  math.cos(q1), 0, 0],
                    [             0,              0, 1, 0],
                    [             0,              0, 0, 1]])
    T01 = T01_trans.dot(Rz1)

    # 2) Joint1 -> Joint2 (rotation q2 about Y)
    T12_trans = np.array([[1,0,0,   0],
                           [0,1,0,   0],
                           [0,0,1,-l2],
                           [0,0,0,   1]])
    Ry2 = np.array([[ math.cos(q2), 0, math.sin(q2), 0],
                    [             0, 1,            0, 0],
                    [-math.sin(q2), 0, math.cos(q2), 0],
                    [             0, 0,            0, 1]])
    T12 = T12_trans.dot(Ry2)
    T02 = T01.dot(T12)

    # 3) Joint2 -> Joint3 (prismatic d3 along -Z)
    T23_base = np.array([[1,0,0,   0],
                          [0,1,0,   0],
                          [0,0,1,-l3],
                          [0,0,0,   1]])
    T_prism = np.array([[1,0,0,    0],
                        [0,1,0,    0],
                        [0,0,1, -d3],
                        [0,0,0,    1]])
    T23 = T23_base.dot(T_prism)
    T03 = T02.dot(T23)

    # 4) Joint3 -> Joint4 (rotation q4 about Z)
    T34_trans = np.array([[1,0,0,   0],
                           [0,1,0,   0],
                           [0,0,1,-l4],
                           [0,0,0,   1]])
    Rz4 = np.array([[ math.cos(q4), -math.sin(q4), 0, 0],
                    [ math.sin(q4),  math.cos(q4), 0, 0],
                    [             0,              0, 1, 0],
                    [             0,              0, 0, 1]])
    T34 = T34_trans.dot(Rz4)
    T04 = T03.dot(T34)

    # 5) Joint4 -> Joint5 (rotation q5 about Y)
    T45_trans = np.array([[1,0,0,   0],
                           [0,1,0,   0],
                           [0,0,1,-l5],
                           [0,0,0,   1]])
    Ry5 = np.array([[ math.cos(q5), 0, math.sin(q5), 0],
                    [             0, 1,            0, 0],
                    [-math.sin(q5), 0, math.cos(q5), 0],
                    [             0, 0,            0, 1]])
    T45 = T45_trans.dot(Ry5)
    T05 = T04.dot(T45)

    # 6) Joint5 -> End-effector (fixed)
    T5e = np.array([[1,0,0,   0],
                    [0,1,0,   0],
                    [0,0,1,-l6],
                    [0,0,0,   1]])
    T0e = T05.dot(T5e)

    return [T01, T02, T03, T04, T05, T0e]


# -- Differential Kinematics / Jacobian --
def compute_jacobian(q):
    """
    Compute the 6x5 geometric Jacobian mapping joint velocities to end-effector spatial velocity.

    Args:
      q: same ordering as direct_kinematics
    Returns:
      J (6x5): [Jv; Jw], where Jv relates to linear velocity, Jw to angular.
    """
    # Fetch all transforms to each joint and ee
    T01, T02, T03, T04, T05, T0e = direct_kinematics(q)

    # Extract positions of each joint frame in base coords
    p = [T[:3,3] for T in (T01, T02, T03, T04, T05)]
    pe = T0e[:3,3]

    # Joint axes in base frame:
    z1 = T01[:3,2]     # joint1 axis (Z)
    y2 = T02[:3,1]     # joint2 axis (Y)
    z3 = T03[:3,2]     # prismatic axis is -Z, so linear = -z3
    z4 = T04[:3,2]     # joint4 axis (Z)
    y5 = T05[:3,1]     # joint5 axis (Y)

    # Build linear part Jv:
    Jv1 = np.cross(z1, pe - p[0])
    Jv2 = np.cross(y2, pe - p[1])
    Jv3 = -z3
    Jv4 = np.cross(z4, pe - p[3])
    Jv5 = np.cross(y5, pe - p[4])

    # Build angular part Jw:
    Jw1, Jw2 = z1, y2
    Jw3 = np.zeros(3)
    Jw4, Jw5 = z4, y5

    # Stack into 6×5 matrix
    J = np.zeros((6,5))
    cols = list(zip([Jv1,Jv2,Jv3,Jv4,Jv5],[Jw1,Jw2,Jw3,Jw4,Jw5]))
    for i,(jv,jw) in enumerate(cols):
        J[:3,i], J[3:,i] = jv, jw
    return J


def geometric2analyticJacobian(J,T_0e):
    R_0e = T_0e[:3,:3]
    math_utils = Math()
    rpy_ee = math_utils.rot2eul(R_0e)
    roll = rpy_ee[0]
    pitch = rpy_ee[1]
    yaw = rpy_ee[2]

    # compute the mapping between euler rates and angular velocity
    T_w = np.array([[math.cos(yaw)*math.cos(pitch),  -math.sin(yaw), 0],
                    [math.sin(yaw)*math.cos(pitch),   math.cos(yaw), 0],
                    [             -math.sin(pitch),               0, 1]])

    T_a = np.array([np.vstack((np.hstack((np.identity(3), np.zeros((3,3)))),
                                          np.hstack((np.zeros((3,3)),np.linalg.inv(T_w)))))])


    J_a = np.dot(T_a, J)

    return J_a[0]