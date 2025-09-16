import numpy as np

dt = 1e-3                    # controller time step
SLOW_FACTOR = 1              # to slow down simulation
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

qd0 =  np.zeros(5, dtype=float)  # velocity
qdd0 = np.zeros(5, dtype=float)  # accelerations

p_des = np.array([1., 2., 1.])   # desired end-effector position in world frame
pitch_des = 30    # desired end-effector pitch in degrees

dyn_sim_duration = 3.   # duration of the dynamic simulation in seconds
sim_duration = 7.       # total duration of the simulation in seconds
traj_duration = 4.      # duration of the trajectory in seconds



