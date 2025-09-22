import pinocchio as pin
from pinocchio.utils import *
import numpy as np
import time as tm
from utils.common_functions import *
import conf as conf

class DynamicSimulator:
    def __init__(self, robot, ros_pub):
        self.robot = robot
        self.ros_pub = ros_pub
        self.zero = np.zeros(5)
        self.error = np.ones(4)
        
        # Joint limits
        self.jl_K = 5       # Joint limit stiffness
        self.jl_D = 0.5     # Joint limit damping

        self.q_max = np.array([
            np.pi,        # base_rotation_limit (PI)
            np.pi/2,      # shoulder_tilt_limit (PI_2)
            6.5,          # telescopic_limit (6.5)
            np.pi,        # wrist_rotation_limit (PI)
            np.pi*5/6     # wrist_tilt_limit (PI * 5/6)
        ])

        self.q_min = np.array([
            -np.pi,       # base_rotation_limit (-PI)
            -np.pi/2,     # shoulder_tilt_limit (-PI_2)
            0.0,          # telescopic_limit (0.0 - fully retracted)
            -np.pi,       # wrist_rotation_limit (-PI)
            -np.pi*5/6    # wrist_tilt_limit (-PI * 5/6)
        ])
        
    def initialize_logs(self, time, q, qd, qdd, q_des, qd_des, qdd_des):
        """Initialize logging variables"""
        logs = {
            'time': [time],
            'q': [q.copy()],
            'qd': [qd.copy()],
            'qdd': [qdd.copy()],
            'q_des': [q_des.copy()],
            'qd_des': [qd_des.copy()],
            'qdd_des': [qdd_des.copy()],
            'tau': [self.zero.copy()]
        }
        return logs
    
    def compute_dynamic_terms(self, q, qd):
        """Compute all dynamic terms using RNEA"""
        # Compute gravity vector using RNEA
        g = pin.rnea(self.robot.model, self.robot.data, q, self.zero, self.zero)

        # Compute mass matrix using RNEA column by column
        n = len(q)
        M = np.zeros((n, n))
        for i in range(n):
            ei = self.zero.copy()
            ei[i] = 1.0
            # M[:, i] = RNEA(q, 0, ei) - g
            tau_col = pin.rnea(self.robot.model, self.robot.data, q, self.zero, ei)
            M[:, i] = tau_col - g

        # Compute Coriolis and centrifugal terms using RNEA
        C = pin.rnea(self.robot.model, self.robot.data, q, qd, self.zero) - g

        # Bias term h = C + g
        h = C + g

        return M, h, g
    
    def compute_joint_limits_torque(self, q, qd):
        """Compute joint limit avoidance torque"""
        return ((q > self.q_max) * (self.jl_K * (self.q_max - q) + self.jl_D * (-qd)) + 
                (q < self.q_min) * (self.jl_K * (self.q_min - q) + self.jl_D * (-qd)))
    
    def compute_gravity_compensation_torque(self, q, qd, q_des, qd_des):
        """Compute gravity compensation torque with PD control to desired configuration"""
        # Compute gravity vector using RNEA
        g = pin.rnea(self.robot.model, self.robot.data, q, self.zero, self.zero)

        # PD gains for smooth motion to home position
        Kp = np.array([8.0, 10.0, 5.0, 6.0, 6.0])   # Position gains
        Kd = np.array([2.0, 2.5, 1.5, 2.0, 2.0])    # Derivative gains

        # PD control torque
        tau_pd = Kp * (q_des - q) + Kd * (qd_des - qd)

        # Joint limit forces (very soft)
        joint_limits_tau = self.compute_joint_limits_torque(q, qd)

        # Total torque = PD control + gravity compensation + joint limits
        return tau_pd + g + joint_limits_tau

    def compute_total_torque(self, q, qd, q_des=None, qd_des=None):
        """Compute total joint torque input for simulation"""
        if q_des is None:
            q_des = np.zeros(5)  # Home position
        if qd_des is None:
            qd_des = np.zeros(5)  # Desired velocity

        # Use gravity compensation control
        return self.compute_gravity_compensation_torque(q, qd, q_des, qd_des)
    
    def simulate(self, q_init, qd_init, qdd_init, q_des, qd_des, qdd_des):
        """Main simulation loop implementing forward dynamics (Task 3)"""
        time = 0.0
        q, qd, qdd = q_init.copy(), qd_init.copy(), qdd_init.copy()
        logs = self.initialize_logs(time, q, qd, qdd, q_des, qd_des, qdd_des)

        while (not ros.is_shutdown()) and (time < conf.dyn_sim_duration):
            # Compute dynamic terms using RNEA
            M, h, _ = self.compute_dynamic_terms(q, qd)

            # Compute total torque input with gravity compensation
            tau = self.compute_total_torque(q, qd, q_des, qd_des)

            # TASK 3: Forward dynamics simulation using RNEA
            # As explained in the labs: compute forward dynamics to get joint accelerations
            # that will be used to calculate velocities and then positions
            try:
                # Add small regularization to avoid singularities
                M_reg = M + 1e-6 * np.eye(5)
                M_inv = np.linalg.inv(M_reg)
                # Forward dynamics: qdd = M^-1 * (tau - h)
                qdd = M_inv.dot(tau - h)

                # Clamp accelerations to prevent numerical explosion
                qdd = np.clip(qdd, -100.0, 100.0)

            except np.linalg.LinAlgError:
                print("Warning: Singular mass matrix, using small accelerations")
                qdd = 0.01 * np.ones(5)

            # Forward Euler Integration
            # First update velocities from accelerations
            qd = qd + qdd * conf.dt
            # Then update positions from velocities and accelerations
            q = q + conf.dt * qd + 0.5 * conf.dt * conf.dt * qdd

            # Clamp velocities and positions to prevent numerical issues
            qd = np.clip(qd, -10.0, 10.0)

            # Clamp positions to reasonable joint limits
            q = np.clip(q, self.q_min * 1.1, self.q_max * 1.1)

            # Check for NaN values and reset if found
            if np.any(np.isnan(q)) or np.any(np.isnan(qd)) or np.any(np.isnan(qdd)):
                print("Warning: NaN detected, resetting to safe values")
                q = np.zeros(5)
                qd = np.zeros(5)
                qdd = np.zeros(5)
            
            # Update time
            time += conf.dt
            
            # Log data
            logs['time'].append(time)
            logs['q'].append(q.copy())
            logs['qd'].append(qd.copy())
            logs['qdd'].append(qdd.copy())
            logs['q_des'].append(q_des.copy())
            logs['qd_des'].append(qd_des.copy())
            logs['qdd_des'].append(qdd_des.copy())
            logs['tau'].append(tau.copy())
            
            # Publish joint variables
            self.ros_pub.publish(self.robot, q, qd)
            tm.sleep(conf.dt * conf.SLOW_FACTOR)
        
        # Convert logs to numpy arrays
        for key in logs:
            logs[key] = np.array(logs[key]).T if key != 'time' else np.array(logs[key])
            
        return logs
