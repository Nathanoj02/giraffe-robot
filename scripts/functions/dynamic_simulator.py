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
        self.jl_K = 10000  # Joint limit stiffness
        self.jl_D = 10     # Joint limit damping

        self.q_max = np.array([
            np.pi,        # base_rotation_limit (PI)
            np.pi/2,      # shoulder_tilt_limit (PI_2)
            6.5,          # telescopic_limit (6.5)
            np.pi,        # wrist_rotation_limit (PI)
            np.pi/2       # wrist_tilt_limit (PI_2)
        ])

        self.q_min = np.array([
            -np.pi,       # base_rotation_limit (-PI)
            -np.pi/2,     # shoulder_tilt_limit (-PI_2)
            0.0,          # telescopic_limit (0.0 - fully retracted)
            -np.pi,       # wrist_rotation_limit (-PI)
            -np.pi/2      # wrist_tilt_limit (-PI_2)
        ])
        
        # Damping coefficient
        self.damping_coeff = -0.1
        
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
        """Compute all dynamic terms using Pinocchio"""
        self.robot.computeAllTerms(q, qd)
        
        # Compute gravity vector
        g = self.robot.gravity(q)
        
        # Compute joint space inertia matrix using RNEA
        M = np.zeros((5, 5))
        for i in range(5):
            ei = self.zero.copy()
            ei[i] = 1
            M[:, i] = pin.rnea(self.robot.model, self.robot.data, q, self.zero, ei) - g
        
        # Compute bias terms (Coriolis + gravity)
        h = self.robot.nle(q, qd, False)
        
        return M, h, g
    
    def compute_joint_limits_torque(self, q, qd):
        """Compute joint limit avoidance torque"""
        return ((q > self.q_max) * (self.jl_K * (self.q_max - q) + self.jl_D * (-qd)) + 
                (q < self.q_min) * (self.jl_K * (self.q_min - q) + self.jl_D * (-qd)))
    
    def compute_total_torque(self, q, qd):
        """Compute total joint torque input"""
        damping = self.damping_coeff * qd
        joint_limits_tau = self.compute_joint_limits_torque(q, qd)
        return joint_limits_tau + damping
    
    def simulate(self, q_init, qd_init, qdd_init, q_des, qd_des, qdd_des):
        """Main simulation loop"""
        time = 0.0
        q, qd, qdd = q_init.copy(), qd_init.copy(), qdd_init.copy()
        logs = self.initialize_logs(time, q, qd, qdd, q_des, qd_des, qdd_des)
        
        while (not ros.is_shutdown()) and (time < conf.dyn_sim_duration):
            # Compute dynamic terms
            M, h, _ = self.compute_dynamic_terms(q, qd)
            
            # Compute total torque
            total_tau = self.compute_total_torque(q, qd)
            
            # Compute joint accelerations using forward dynamics
            qdd = np.linalg.inv(M).dot(total_tau - h)
            
            # Forward Euler Integration
            qd += qdd * conf.dt
            q += conf.dt * qd + 0.5 * pow(conf.dt, 2) * qdd
            
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
            logs['tau'].append(total_tau.copy())
            
            # Publish joint variables
            self.ros_pub.publish(self.robot, q, qd)
            tm.sleep(conf.dt * conf.SLOW_FACTOR)
        
        # Convert logs to numpy arrays
        for key in logs:
            logs[key] = np.array(logs[key]).T if key != 'time' else np.array(logs[key])
            
        return logs
