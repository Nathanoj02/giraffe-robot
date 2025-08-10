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
        
        # Task 6: PD gains for 7s settling time without overshoot (critically damped)
        # For second-order system: Kp = (4/ts)^2, Kd = 2*sqrt(Kp)
        ts = 7.0  # settling time
        self.Kp_pos = np.eye(3) * (4/ts)**2
        self.Kd_pos = np.eye(3) * 2*np.sqrt((4/ts)**2)
        self.Kp_pitch = (4/ts)**2
        self.Kd_pitch = 2*np.sqrt(self.Kp_pitch)
        
        # Task 7: Null-space configuration and gains
        self.q0_postural = conf.q0.copy()
        self.Kp_postural = 1.0
        self.Kd_postural = 2.0

    def forward_kinematics(self, q):
        """Compute current end-effector position and orientation"""
        pin.framesForwardKinematics(self.model, self.data, q)
        p = self.data.oMf[self.frame_id].translation.copy()
        rpy = pin.rpy.matrixToRpy(self.data.oMf[self.frame_id].rotation)
        return p, rpy, rpy[1]  # position, full rpy, pitch

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
        """Task 4: Compute q0_calibrated using IK for desired end-effector pose"""
        eps = 1e-4
        IT_MAX = 5000
        damp = 1e-5
        dt_ik = 0.01
        q_ik = conf.q0.copy()

        for i in range(IT_MAX):
            pin.forwardKinematics(self.model, self.data, q_ik)
            pin.updateFramePlacement(self.model, self.data, self.frame_id)

            p_ik = self.data.oMf[self.frame_id].translation
            pitch_ik = pin.rpy.matrixToRpy(self.data.oMf[self.frame_id].rotation)[1]
            err_4d = np.hstack([p_ik - conf.p_des, pitch_ik - pitch_des_final])

            if np.linalg.norm(err_4d) < eps:
                print(f"IK converged in {i} iterations")
                break

            J = pin.computeFrameJacobian(self.model, self.data, q_ik, self.frame_id, 
                                       pin.LOCAL_WORLD_ALIGNED)
            J_4d = np.vstack([J[:3, :], J[4, :]])
            J_inv = J_4d.T @ np.linalg.inv(J_4d @ J_4d.T + damp * np.eye(4))
            q_ik = pin.integrate(self.model, q_ik, -J_inv @ err_4d * dt_ik)
        else:
            print("IK didn't converge")
            
        return q_ik

    def generate_trajectory(self, t, pitch_des_final):
        """Task 4: Generate 5th-order polynomial trajectory in task space"""
        p0, _, pitch0 = self.forward_kinematics(conf.q0)
        
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
        """Tasks 5,6,7: Compute inverse dynamics control with task space linearization and null-space projection"""
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q, qd)
        pin.updateFramePlacement(self.model, self.data, self.frame_id)
        
        # Current state
        p = self.data.oMf[self.frame_id].translation
        pitch = pin.rpy.matrixToRpy(self.data.oMf[self.frame_id].rotation)[1]
        J = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        twist = J @ qd
        v, omega = twist[:3], twist[3:]
        
        # Task 5: Task space errors
        pos_error = p_des - p
        vel_error = v_des - v
        pitch_error = pitch_des - pitch
        pitch_vel_error = pitch_vel_des - omega[1]
        
        # Task 6: PD control with computed gains
        a_pos_des = a_des + self.Kd_pos @ vel_error + self.Kp_pos @ pos_error
        a_pitch_des = pitch_acc_des + self.Kd_pitch * pitch_vel_error + self.Kp_pitch * pitch_error
        
        # Task Jacobian (position + pitch)
        J_task = np.vstack([J[:3, :], J[4, :]])
        Jdot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        Jdot_task = np.vstack([Jdot[:3, :], Jdot[4, :]])
        
        # Task 5: Inverse dynamics
        M = pin.crba(self.model, self.data, q)
        h = pin.nonLinearEffects(self.model, self.data, q, qd)
        Lambda_task = np.linalg.inv(J_task @ np.linalg.inv(M) @ J_task.T + 1e-6*np.eye(4))
        
        # Task 7: Null-space projection
        N_task = np.eye(self.model.nv) - J_task.T @ np.linalg.pinv(J_task.T, 1e-4)
        q_postural = self.Kp_postural * (q0_calibrated - q) - self.Kd_postural * qd
        
        # Combined control
        a_task = np.hstack([a_pos_des, a_pitch_des])
        qdd_des = np.linalg.inv(M) @ J_task.T @ Lambda_task @ (a_task - Jdot_task @ qd) + N_task @ q_postural
        tau = M @ qdd_des + h
        
        return tau

    def simulate(self, pitch_des_final):
        """Main simulation loop"""
        logs = self.initialize_logs()
        q, qd = conf.q0.copy(), np.zeros(self.model.nv)
        q0_calibrated = self.compute_postural_target(pitch_des_final)
        log_counter = 0
        
        for t in np.arange(0, conf.sim_duration, conf.dt):
            if log_counter >= len(logs['time']):
                break
                
            # Task 4: Generate trajectory
            p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des = \
                self.generate_trajectory(t, pitch_des_final)
            
            # Tasks 5,6,7: Compute control
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
            p, _, pitch = self.forward_kinematics(q)
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
