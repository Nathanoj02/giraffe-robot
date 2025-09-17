from __future__ import print_function

import pinocchio as pin
import numpy as np
import math
import time as tm

from utils.kin_dyn_utils import fifthOrderPolynomialTrajectory as coeffTraj
import conf as conf

class TaskSpaceController:
    def __init__(self, robot, model, data, ros_pub):
        self.robot = robot
        self.model = model
        self.data = data
        self.ros_pub = ros_pub
        self.frame_id = model.getFrameId(conf.frame_name)
        
        # Further increased damping to eliminate overshoot
        ts = 7.0  # settling time
        damping_ratio = 1.3  # More overdamped (1.3 instead of 1.1)
        
        # Position gains
        omega_n_pos = 4.0 / (ts * damping_ratio)
        self.Kp_pos = np.eye(3) * omega_n_pos**2
        self.Kd_pos = np.eye(3) * 2 * damping_ratio * omega_n_pos
        
        # Pitch gains
        omega_n_pitch = 4.0 / (ts * damping_ratio)
        self.Kp_pitch = omega_n_pitch**2
        self.Kd_pitch = 2 * damping_ratio * omega_n_pitch
        
        # Integral gains (small to prevent windup)
        self.Ki_pos = np.eye(3) * omega_n_pos**2 * 0.1
        self.Ki_pitch = omega_n_pitch**2 * 0.1
        
        # Error integrals
        self.pos_error_integral = np.zeros(3)
        self.pitch_error_integral = 0.0
        
        # Further reduced null-space gains
        self.q0_postural = conf.q0.copy()
        self.Kp_postural = 2.0  # Even lower
        self.Kd_postural = 2.0 * np.sqrt(self.Kp_postural)
        
        # Filter initialization
        self.vel_prev = np.zeros(3)
        self.pitch_vel_prev = 0.0
        self.alpha = 0.9  # Stronger filtering
        
        # Deadband and anti-windup
        self.deadband = 0.002
        self.integral_limit = 0.5  # Limit for integral term to prevent windup
        
        # Error-based gain scheduling
        self.error_threshold = 0.01  # Reduce gains when error is below this

    def forward_kinematics(self, q):
        """Compute current end-effector position and orientation"""
        pin.framesForwardKinematics(self.model, self.data, q)
        p = self.data.oMf[self.frame_id].translation.copy()

        # Get rotation matrix and extract pitch in cartesian space
        R = self.data.oMf[self.frame_id].rotation
        
        # For microphone: pitch is rotation around world Y-axis
        pitch = np.arcsin(-R[2, 2])
        
        return p, pitch

    def initialize_logs(self):
        """Initialize logs"""
        buffer_size = int(math.ceil(conf.sim_duration / conf.dt))
        return {
            'time': np.zeros(buffer_size),
            'q': np.zeros((self.model.nq, buffer_size)),
            'p': np.zeros((3, buffer_size)),
            'p_des': np.zeros((3, buffer_size)),
            'pitch': np.zeros(buffer_size),
            'pitch_des': np.zeros(buffer_size)
        }

    def compute_postural_target(self, pitch_des_final):
        """Compute q0_calibrated using IK for desired end-effector pose"""
        eps = 1e-4
        IT_MAX = 5000
        damp = 1e-5
        dt_ik = 0.01
        q_ik = conf.q0.copy()

        for i in range(IT_MAX):
            pin.forwardKinematics(self.model, self.data, q_ik)
            pin.updateFramePlacement(self.model, self.data, self.frame_id)

            p_ik = self.data.oMf[self.frame_id].translation
            R_ik = self.data.oMf[self.frame_id].rotation
            pitch_ik = np.arcsin(-R_ik[2, 2])
            
            err_4d = np.hstack([p_ik - conf.p_des, pitch_ik - pitch_des_final])

            if np.linalg.norm(err_4d) < eps:
                print(f"IK converged in {i} iterations")
                break

            J = pin.computeFrameJacobian(self.model, self.data, q_ik, self.frame_id,
                                       pin.LOCAL_WORLD_ALIGNED)
            
            # Proper pitch Jacobian in cartesian space
            J_pos = J[:3, :]
            
            # For pitch around world Y-axis, we need the Jacobian for rotation around Y
            J_angular = J[3:6, :]
            
            # The pitch Jacobian is the Y-component of the angular velocity in world frame
            J_pitch = (R_ik.T @ J_angular)[1, :].reshape(1, -1)
            
            J_4d = np.vstack([J_pos, J_pitch])
            J_inv = J_4d.T @ np.linalg.inv(J_4d @ J_4d.T + damp * np.eye(4))
            q_ik = pin.integrate(self.model, q_ik, -J_inv @ err_4d * dt_ik)
        else:
            print("IK didn't converge")
            
        return q_ik

    def generate_trajectory(self, t, pitch_des_final):
        """Generate 5th-order polynomial trajectory in task space"""
        p0, pitch0 = self.forward_kinematics(conf.q0)
        
        if t > conf.traj_duration:
            return conf.p_des, np.zeros(3), np.zeros(3), pitch_des_final, 0.0, 0.0

        # Position trajectory
        p_des, v_des, a_des = np.zeros(3), np.zeros(3), np.zeros(3)
        for i in range(3):
            c = coeffTraj(conf.traj_duration, p0[i], conf.p_des[i])
            p_des[i] = sum(c[j] * t**j for j in range(6))
            v_des[i] = sum((j+1)*c[j+1] * t**j for j in range(5))
            a_des[i] = sum((j+1)*(j+2)*c[j+2] * t**j for j in range(4))

        # Pitch trajectory
        c = coeffTraj(conf.traj_duration, pitch0, pitch_des_final)
        pitch_des = sum(c[j] * t**j for j in range(6))
        pitch_vel_des = sum((j+1)*c[j+1] * t**j for j in range(5))
        pitch_acc_des = sum((j+1)*(j+2)*c[j+2] * t**j for j in range(4))

        return p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des

    def compute_control(self, q, qd, p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des, q0_calibrated):
        """Compute inverse dynamics control with task space linearization"""
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q, qd)
        pin.updateFramePlacement(self.model, self.data, self.frame_id)
        
        # Current state
        p = self.data.oMf[self.frame_id].translation
        R = self.data.oMf[self.frame_id].rotation
        pitch = np.arcsin(-R[2, 2])
        
        J = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        twist = J @ qd
        v, omega = twist[:3], twist[3:]
        
        # Compute pitch velocity (rotation around world Y-axis)
        J_angular = J[3:6, :]
        pitch_vel_current = (R.T @ J_angular @ qd)[1]
        
        # Apply low-pass filtering to velocity estimates
        v_filtered = self.alpha * self.vel_prev + (1 - self.alpha) * v
        pitch_vel_current_filtered = self.alpha * self.pitch_vel_prev + (1 - self.alpha) * pitch_vel_current
        
        # Update previous values for next iteration
        self.vel_prev = v_filtered
        self.pitch_vel_prev = pitch_vel_current_filtered
        
        # Task space errors
        pos_error = p_des - p
        vel_error = v_des - v_filtered
        
        # Pitch error calculation
        pitch_error = pitch_des - pitch
        pitch_vel_error = pitch_vel_des - pitch_vel_current_filtered
        
        # Error-based gain scheduling
        pos_error_norm = np.linalg.norm(pos_error)
        pitch_error_abs = abs(pitch_error)
        
        # Reduce gains when close to target to prevent overshoot
        if pos_error_norm < self.error_threshold:
            Kp_pos_effective = self.Kp_pos * (pos_error_norm / self.error_threshold)
            Kd_pos_effective = self.Kd_pos * (pos_error_norm / self.error_threshold)
        else:
            Kp_pos_effective = self.Kp_pos
            Kd_pos_effective = self.Kd_pos
            
        if pitch_error_abs < self.error_threshold:
            Kp_pitch_effective = self.Kp_pitch * (pitch_error_abs / self.error_threshold)
            Kd_pitch_effective = self.Kd_pitch * (pitch_error_abs / self.error_threshold)
        else:
            Kp_pitch_effective = self.Kp_pitch
            Kd_pitch_effective = self.Kd_pitch
        
        # Update integral terms with anti-windup
        self.pos_error_integral += pos_error * conf.dt
        self.pitch_error_integral += pitch_error * conf.dt
        
        # Limit integral terms to prevent windup
        self.pos_error_integral = np.clip(self.pos_error_integral, -self.integral_limit, self.integral_limit)
        self.pitch_error_integral = np.clip(self.pitch_error_integral, -self.integral_limit, self.integral_limit)
        
        # PD control with computed gains and integral term
        a_pos_des = a_des + Kd_pos_effective @ vel_error + Kp_pos_effective @ pos_error + self.Ki_pos @ self.pos_error_integral
        a_pitch_des = pitch_acc_des + Kd_pitch_effective * pitch_vel_error + Kp_pitch_effective * pitch_error + self.Ki_pitch * self.pitch_error_integral
        
        # Task Jacobian (position + pitch)
        J_pos = J[:3, :]
        
        # Pitch Jacobian (rotation around world Y-axis)
        J_pitch = (R.T @ J_angular)[1, :].reshape(1, -1)
        J_task = np.vstack([J_pos, J_pitch])

        # Time derivative of Jacobian
        Jdot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        Jdot_pos = Jdot[:3, :]
        
        # Approximate Jdot for pitch
        Jdot_pitch = np.zeros((1, self.model.nv))
        Jdot_task = np.vstack([Jdot_pos, Jdot_pitch])
        
        # Inverse dynamics
        M = pin.crba(self.model, self.data, q)
        h = pin.nonLinearEffects(self.model, self.data, q, qd)
        Lambda_task = np.linalg.inv(J_task @ np.linalg.inv(M) @ J_task.T + 1e-6*np.eye(4))
        
        # Null-space projection
        N_task = np.eye(self.model.nv) - J_task.T @ np.linalg.pinv(J_task.T, 1e-4)
        q_postural = self.Kp_postural * (q0_calibrated - q) - self.Kd_postural * qd
        
        # Combined control
        a_task = np.hstack([a_pos_des, a_pitch_des])
        qdd_des = np.linalg.inv(M) @ J_task.T @ Lambda_task @ (a_task - Jdot_task @ qd) + N_task @ q_postural
        tau = M @ qdd_des + h
        
        # Apply deadband to prevent small oscillations
        tau_norm = np.linalg.norm(tau)
        if tau_norm < self.deadband:
            tau = np.zeros_like(tau)

        return tau

    def simulate(self, pitch_des_final):
        """Main simulation loop"""
        logs = self.initialize_logs()
        q, qd = conf.q0.copy(), np.zeros(self.model.nv)
        q0_calibrated = self.compute_postural_target(pitch_des_final)
        log_counter = 0
        
        # Reset integrals at start of simulation
        self.pos_error_integral = np.zeros(3)
        self.pitch_error_integral = 0.0
        
        for t in np.arange(0, conf.sim_duration, conf.dt):
            if log_counter >= len(logs['time']):
                break
                
            # Generate trajectory
            p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des = \
                self.generate_trajectory(t, pitch_des_final)
            
            # Compute control
            tau = self.compute_control(q, qd, p_des, v_des, a_des, 
                                     pitch_des, pitch_vel_des, pitch_acc_des, q0_calibrated)
            
            # Forward dynamics integration
            Minv = np.linalg.inv(pin.crba(self.model, self.data, q))
            h = pin.nonLinearEffects(self.model, self.data, q, qd)
            qdd = Minv @ (tau - h)
            qd_next = qd + qdd * conf.dt
            q_next = pin.integrate(self.model, q, (qd + qd_next) * 0.5 * conf.dt)
            
            # Update state
            q, qd = q_next, qd_next
            
            # Log data
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