from __future__ import print_function

import pinocchio as pin
import numpy as np
import math
import time as tm

from utils.kin_dyn_utils import fifthOrderPolynomialTrajectory as coeffTraj
import conf as conf

class TaskSpaceController:
    def __init__(self, robot, model, data, ros_pub, q0, qd0, qdd0, p_des, pitch_des):
        self.robot = robot
        self.model = model
        self.data = data
        self.ros_pub = ros_pub
        self.frame_id = model.getFrameId(conf.frame_name)

        self.ts = conf.task_sim_duration  # settling time

        # Initial joint states
        self.q0 = q0
        self.qd0 = qd0
        self.qdd0 = qdd0

        # Desired end-effector position and orientation
        self.p_des = p_des
        self.pitch_des = pitch_des

        # Position gains
        self.kp_pos = 100.0
        self.kd_pos = 2 * np.sqrt(self.kp_pos)
        
        # Pitch gains
        self.kp_pitch = 50.0
        self.kd_pitch = 2 * np.sqrt(self.kp_pitch)
        
        # Null-space gains
        self.kp_ns = 5.0
        self.kd_ns = 2 * np.sqrt(self.kp_ns)
        

    def forward_kinematics(self, q):
        """Compute current end-effector position and orientation"""
        pin.framesForwardKinematics(self.model, self.data, q)
        p = self.data.oMf[self.frame_id].translation.copy()
        R = self.data.oMf[self.frame_id].rotation
        
        pitch = np.arcsin(-R[2, 2])
        
        return p, pitch
    

    def initialize_logs(self):
        """Initialize logs"""
        buffer_size = int(math.ceil(self.ts / conf.dt))
        return {
            'time': np.zeros(buffer_size),
            'q': np.zeros((self.model.nq, buffer_size)),
            'p': np.zeros((3, buffer_size)),
            'p_des': np.zeros((3, buffer_size)),
            'pitch': np.zeros(buffer_size),
            'pitch_des': np.zeros(buffer_size)
        }


    def generate_trajectory(self, pitch_des_final):
        """Generate 5th-order polynomial trajectory in task space"""
        p0, pitch0 = self.forward_kinematics(self.q0)

        # Create time array
        time_steps = int(math.ceil(self.ts / conf.dt))
        time_array = np.arange(0, self.ts, conf.dt)
        
        # Initialize trajectory arrays
        p_traj = np.zeros((3, time_steps))
        v_traj = np.zeros((3, time_steps))
        a_traj = np.zeros((3, time_steps))
        pitch_traj = np.zeros(time_steps)
        pitch_vel_traj = np.zeros(time_steps)
        pitch_acc_traj = np.zeros(time_steps)
        
        # Compute position coefficients
        pos_coeffs = [coeffTraj(self.ts, p0[i], self.p_des[i]) for i in range(3)]
        # Compute pitch coefficients
        pitch_coeffs = coeffTraj(self.ts, pitch0, pitch_des_final)
        
        for idx, t in enumerate(time_array):
            # Position trajectory
            for i in range(3):
                c = pos_coeffs[i]
                p_traj[i, idx] = sum(c[j] * t**j for j in range(6))
                v_traj[i, idx] = sum((j+1)*c[j+1] * t**j for j in range(5))
                a_traj[i, idx] = sum((j+1)*(j+2)*c[j+2] * t**j for j in range(4))
            
            # Pitch trajectory
            pitch_traj[idx] = sum(pitch_coeffs[j] * t**j for j in range(6))
            pitch_vel_traj[idx] = sum((j+1)*pitch_coeffs[j+1] * t**j for j in range(5))
            pitch_acc_traj[idx] = sum((j+1)*(j+2)*pitch_coeffs[j+2] * t**j for j in range(4))
        
        return {
            'time': time_array,
            'p': p_traj,
            'v': v_traj,
            'a': a_traj,
            'pitch': pitch_traj,
            'pitch_vel': pitch_vel_traj,
            'pitch_acc': pitch_acc_traj
        }


    def compute_control(self, q, qd, p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des, q0):
        """Compute control torques using inverse dynamics in task space with null-space projection"""
        # Update model with current state and compute dynamics
        pin.computeAllTerms(self.model, self.data, q, qd)
        pin.framesForwardKinematics(self.model, self.data, q)

        # Get the full 6D Jacobian for the end-effector frame
        J6 = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        R = self.data.oMf[self.frame_id].rotation

        ## Task Jacobian (position + pitch)
        J_pos = J6[:3, :]
        J_angular_world = J6[3:6, :]
        J_pitch = (R.T @ J_angular_world)[1, :].reshape(1, -1)
        J_task = np.vstack([J_pos, J_pitch])
        
        # Get current state
        p, pitch = self.forward_kinematics(q)
        v_linear = J_pos @ qd
        omega_world = J_angular_world @ qd
        pitch_vel_current = (R.T @ omega_world)[1]
        
        # Calculate errors
        p_error = p_des - p
        v_error = v_des - v_linear
        pitch_error = pitch_des - pitch
        pitch_vel_error = pitch_vel_des - pitch_vel_current

        # Desired task-space acceleration (PD control)
        acc_des_task = np.zeros(4)
        acc_des_task[:3] = a_des + self.kd_pos * v_error + self.kp_pos * p_error
        acc_des_task[3] = pitch_acc_des + self.kd_pitch * pitch_vel_error + self.kp_pitch * pitch_error

        # Time derivative of Jacobian (approximating pitch part as zero)
        Jdot6 = pin.getFrameJacobianTimeVariation(self.model, self.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jdot_pos = Jdot6[:3, :]
        Jdot_pitch = np.zeros((1, self.model.nv))
        Jdot_task = np.vstack([Jdot_pos, Jdot_pitch])
        
        # Inverse Dynamics (without null-space)
        lambda_damping = 0.1
        I = np.eye(J_task.shape[0])
        J_task_pinv = J_task.T @ np.linalg.inv(J_task @ J_task.T + lambda_damping**2 * I)
        
        # Null-space projection for secondary task
        null_space_projector = np.eye(self.model.nv) - J_task_pinv @ J_task
        
        # Null-space desired acceleration
        q_ddot_null = self.kp_ns * (q0 - q) - self.kd_ns * qd
        
        # Final joint acceleration
        q_ddot = J_task_pinv @ (acc_des_task - Jdot_task[:4,:] @ qd) + null_space_projector @ q_ddot_null

        # Compute control torques
        tau = self.data.M @ q_ddot + self.data.nle

        return tau


    def simulate(self):
        """Main simulation loop"""
        logs = self.initialize_logs()
        q, qd = self.q0.copy(), np.zeros(self.model.nv)
        log_counter = 0

        # Generate trajectory
        trajectory = self.generate_trajectory(self.pitch_des)

        # Simulation loop
        for idx in range(len(trajectory['time'])):
            # Current time and desired states from trajectory
            t = trajectory['time'][idx]
            p_des = trajectory['p'][:, idx]
            v_des = trajectory['v'][:, idx]
            a_des = trajectory['a'][:, idx]
            pitch_des = trajectory['pitch'][idx]
            pitch_vel_des = trajectory['pitch_vel'][idx]
            pitch_acc_des = trajectory['pitch_acc'][idx]

            # Compute control
            tau = self.compute_control(q, qd, p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des, self.q0)
                
            # Forward dynamics integration
            q = pin.integrate(self.model, q, qd * conf.dt)
            a = pin.aba(self.model, self.data, q, qd, tau)
            qd += a * conf.dt

            # Log data
            if log_counter < len(logs['time']):
                p, pitch = self.forward_kinematics(q)
                logs['time'][log_counter] = t
                logs['q'][:, log_counter] = q
                logs['p'][:, log_counter] = p
                logs['p_des'][:, log_counter] = p_des
                logs['pitch'][log_counter] = pitch
                logs['pitch_des'][log_counter] = pitch_des
                log_counter += 1
                
            # Publish to ROS
            self.ros_pub.publish(self.robot, q, qd, tau)
            tm.sleep(conf.dt * conf.SLOW_FACTOR)
        
        trimmed_logs = {k: v[:log_counter] for k, v in logs.items()}
        
        return q, qd, trimmed_logs