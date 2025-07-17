# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: Jonathan Fin
"""

import numpy as np

dt = 0.001                   # controller time step
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

qd0 =  np.zeros(5, dtype=np.float)  # velocity
qdd0 = np.zeros(5, dtype=np.float)  # accelerations





