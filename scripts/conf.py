import numpy as np

dt = 1e-3                    # controller time step
SLOW_FACTOR = 4              # to slow down simulation
frame_name = 'ee_link'       # name of the frame to control (end-effector) in the URDF

# Initial Conditions
# Positions
q0 = np.array([
    0.0,   # base_rotation_z: neutral
    0.0,   # shoulder_tilt_y: neutral
    0.0,   # telescopic_joint: retracted
    0.0,   # wrist_1_joint: neutral
    0.0    # wrist_2_joint: neutral
])

qd0 =  np.zeros(5, dtype=np.float)  # velocity
qdd0 = np.zeros(5, dtype=np.float)  # accelerations

p_des = np.array([1., 2., 1.])   # desired end-effector position in world frame
pitch_des = -30.    # desired end-effector pitch in degrees

exp_dyn_duration = 3.0  # duration of the dynamic experiment in seconds



