import numpy as np
import math
from utils.math_tools import Math
import pinocchio as pin
import os


def setRobotParameters():
    """
    Define and return all key geometric and inertial parameters of the giraffe robot.

    Returns:
      lengths: np.array of link offsets [l1..l6]
      inertia_tensors: np.array of 7 (7 links) 3x3 inertia matrices around each COM
      link_masses: np.array of masses for links 0..6
      coms: np.array of COM positions in each link's local frame
    """
    # Link lengths based on URDF xacro properties
    lengths = np.array([
        0.1,    # l1: base_length/2 = 0.2/2
        0.075,  # l2: shoulder_length/2 = 0.15/2
        0.4,    # l3: upper_arm_size_z/2 = 0.8/2
        0.5,    # l4: telescopic_length/2 = 1.0/2
        0.04,   # l5: wrist1_length/2 = 0.08/2
        0.08    # l6: wrist2_length/2 + ee_length/2 = 0.06/2 + 0.1/2
    ])
    
    # Link masses based on URDF xacro properties
    link_masses = np.array([
        5.0,  # base_mass
        3.0,  # shoulder_mass
        4.0,  # upper_arm_mass
        2.0,  # telescopic_mass
        1.0,  # wrist1_mass
        0.5,  # wrist2_mass
        0.2   # ee_mass
    ])
    
    # Link dimensions for inertia calculations
    base_r, base_h = 0.1, 0.2
    shoulder_r, shoulder_h = 0.08, 0.15
    upper_arm_x, upper_arm_y, upper_arm_z = 0.12, 0.12, 0.8
    telescopic_r, telescopic_h = 0.06, 1.0
    wrist1_r, wrist1_h = 0.035, 0.08
    wrist2_r, wrist2_h = 0.03, 0.06
    ee_r, ee_h = 0.02, 0.1
    
    # Inertia tensors based on URDF geometry
    inertia_tensors = np.array([
        # Cylinder inertia: Ixx=Iyy=m*(3*r²+h²)/12, Izz=m*r²/2
        # Base link (cylinder)
        np.diag([link_masses[0]*(3*base_r**2 + base_h**2)/12, 
                link_masses[0]*(3*base_r**2 + base_h**2)/12,
                link_masses[0]*base_r**2/2]),
        # Shoulder link (cylinder)
        np.diag([link_masses[1]*(3*shoulder_r**2 + shoulder_h**2)/12,
                link_masses[1]*(3*shoulder_r**2 + shoulder_h**2)/12,
                link_masses[1]*shoulder_r**2/2]),
        # Upper arm link (box)
        # Box inertia: Ixx=m*(y²+z²)/12, Iyy=m*(x²+z²)/12, Izz=m*(x²+y²)/12
        np.diag([link_masses[2]*(upper_arm_y**2 + upper_arm_z**2)/12,
                link_masses[2]*(upper_arm_x**2 + upper_arm_z**2)/12,
                link_masses[2]*(upper_arm_x**2 + upper_arm_y**2)/12]),
        # Telescopic link (cylinder)
        np.diag([link_masses[3]*(3*telescopic_r**2 + telescopic_h**2)/12,
                link_masses[3]*(3*telescopic_r**2 + telescopic_h**2)/12,
                link_masses[3]*telescopic_r**2/2]),
        # Wrist1 link (cylinder)
        np.diag([link_masses[4]*(3*wrist1_r**2 + wrist1_h**2)/12,
                link_masses[4]*(3*wrist1_r**2 + wrist1_h**2)/12,
                link_masses[4]*wrist1_r**2/2]),
        # Wrist2 link (cylinder)
        np.diag([link_masses[5]*(3*wrist2_r**2 + wrist2_h**2)/12,
                link_masses[5]*(3*wrist2_r**2 + wrist2_h**2)/12,
                link_masses[5]*wrist2_r**2/2]),
        # End-effector link (cylinder)
        np.diag([link_masses[6]*(3*ee_r**2 + ee_h**2)/12,
                link_masses[6]*(3*ee_r**2 + ee_h**2)/12,
                link_masses[6]*ee_r**2/2])
    ])
    
    # Hardcoded centers of mass (assume centered for simplicity)
    coms = np.zeros((7, 3))
    
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
    T01_trans = np.array([[1,0,0, 0],
                          [0,1,0, 0],
                          [0,0,1,-l1],
                          [0,0,0, 1]])
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

    # 6) Joint5 -> End-effector (fixed) - now using l6 from setRobotParameters
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