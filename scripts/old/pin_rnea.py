import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import time as tm
import numpy as np

class GiraffeRobotSimulator:
    def __init__(self, urdf_path):
        """
        Initialize the robot simulator using Pinocchio RNEA
        
        Args:
            urdf_path: Path to the URDF file
        """
        # Load robot model
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
        self.model = self.robot.model
        self.data = self.robot.data
        
        # Robot properties
        self.nq = self.model.nq  # Number of configuration variables
        self.nv = self.model.nv  # Number of velocity variables
        
        # Initialize state
        self.q = pin.neutral(self.model)  # Joint positions
        self.dq = np.zeros(self.nv)       # Joint velocities
        self.ddq = np.zeros(self.nv)      # Joint accelerations
        
    def forward_dynamics(self, q, dq, tau):
        """
        Compute forward dynamics: ddq = M^-1(tau - h(q,dq))
        where h(q,dq) includes Coriolis, centrifugal and gravity forces
        
        Args:
            q: Joint positions [nq x 1]
            dq: Joint velocities [nv x 1] 
            tau: Joint torques [nv x 1]
            
        Returns:
            ddq: Joint accelerations [nv x 1]
        """
        # Compute the mass matrix M(q)
        M = pin.crba(self.model, self.data, q)
        
        # Compute bias forces h(q,dq) = C(q,dq)*dq + g(q)
        h = pin.rnea(self.model, self.data, q, dq, np.zeros(self.nv))
        
        # Solve: M*ddq = tau - h  =>  ddq = M^-1*(tau - h)
        ddq = pin.aba(self.model, self.data, q, dq, tau)
        
        return ddq
    
    def inverse_dynamics(self, q, dq, ddq):
        """
        Compute inverse dynamics using RNEA: tau = M*ddq + h(q,dq)
        
        Args:
            q: Joint positions [nq x 1]
            dq: Joint velocities [nv x 1]
            ddq: Joint accelerations [nv x 1]
            
        Returns:
            tau: Required joint torques [nv x 1]
        """
        tau = pin.rnea(self.model, self.data, q, dq, ddq)
        return tau
    
    def step_simulation(self, tau, dt=0.001):
        # Special velocity limits for prismatic joint
        max_vel = np.array([1.5, 1.5, 0.05, 1.0, 1.0])  # 5cm/s max for joint 3
        
        # Compute dynamics
        self.ddq = self.forward_dynamics(self.q, self.dq, tau)
        
        # Apply strict velocity limits
        new_dq = self.dq + self.ddq * dt
        self.dq = np.clip(new_dq, -max_vel, max_vel)
        
        # Extra protection for prismatic joint
        if abs(new_dq[2]) > max_vel[2]:
            print(f"Prismatic joint velocity capped: {new_dq[2]:.3f}m/s")

        prismatic_limit = 6.0  # Max extension in meters
        if (self.q[2] + self.dq[2]*dt) > prismatic_limit:
            self.dq[2] = min(0, self.dq[2])  # Only allow retraction
            print("Prismatic joint at maximum extension")
            
        self.q = pin.integrate(self.model, self.q, self.dq * dt)
        
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
    
    def simulate_with_ros(self, torque_function, ros_pub, robot, duration=10.0, dt=0.01, 
                         viz_rate=10, save_data=True):
        """
        Run simulation with ROS visualization using external ros_pub
        
        Args:
            torque_function: Function that returns tau given (t, q, dq)
            ros_pub: RosPub instance
            robot: robot model
            duration: Simulation time in seconds
            dt: Simulation time step
            viz_rate: Visualization update rate (Hz)
            save_data: Whether to save simulation data
            
        Returns:
            If save_data=True: (time_array, q_history, dq_history, ddq_history)
        """
        steps = int(duration / dt)
        viz_step = max(1, int(1.0 / (viz_rate * dt)))  # Steps between viz updates
        
        if save_data:
            t_history = np.zeros(steps)
            q_history = np.zeros((steps, self.nq))
            dq_history = np.zeros((steps, self.nv))  
            ddq_history = np.zeros((steps, self.nv))
        
        for i in range(steps):
            t = i * dt
            
            # Get torques from user-defined function
            tau = torque_function(t, self.q.copy(), self.dq.copy())
            
            # Store data before step
            if save_data:
                t_history[i] = t
                q_history[i] = self.q.copy()
                dq_history[i] = self.dq.copy()
                ddq_history[i] = self.ddq.copy()
            
            # Simulate one step
            self.step_simulation(tau, dt)
            
            # Update ROS visualization
            if i % viz_step == 0:
                ros_pub.publish(robot, self.q)
            
            # Progress display
            if i % (steps//10) == 0:
                print(f"Progress: {100*i//steps}%")
                
        print("Simulation completed!")
        
        if save_data:
            return t_history, q_history, dq_history, ddq_history


    def demonstrate_config_ros(self, torque_function, q, ros_pub, robot, duration=3.0, description="", 
                                dt=0.01, save_data=True):
        """
        Display a configuration using your existing ros_pub
        
        Args:
            q: Joint configuration
            ros_pub: Your RosPub instance  
            robot: Your robot model
            duration: Display duration
            description: Description to print
        """
        if description:
            print(f"Demonstrating: {description}")
            
        # Update internal state
        self.q = q.copy()
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get end-effector position and add marker
        ee_pos, _ = self.get_end_effector_pose(self.q)
        ros_pub.add_marker(ee_pos)
        
        # Publish and wait
        ros_pub.publish(robot, self.q)
        tm.sleep(duration)
        """
        Run a complete simulation
        
        Args:
            torque_function: Function that returns tau given (t, q, dq)
            duration: Simulation time in seconds
            dt: Time step
            save_data: Whether to save simulation data
            
        Returns:
            If save_data=True: (time_array, q_history, dq_history, ddq_history)
        """
        steps = int(duration / dt)
        
        if save_data:
            t_history = np.zeros(steps)
            q_history = np.zeros((steps, self.nq))
            dq_history = np.zeros((steps, self.nv))  
            ddq_history = np.zeros((steps, self.nv))
        
        print(f"Running simulation: {steps} steps, dt={dt}s")
        
        for i in range(steps):
            t = i * dt
            
            # Get torques from user-defined function
            tau = torque_function(t, self.q.copy(), self.dq.copy())
            
            # Store data before step
            if save_data:
                t_history[i] = t
                q_history[i] = self.q.copy()
                dq_history[i] = self.dq.copy()
                ddq_history[i] = self.ddq.copy()
            
            # Simulate one step
            self.step_simulation(tau, dt)
            
            if i % (steps//10) == 0:
                print(f"Progress: {100*i//steps}%")
        
        print("Simulation completed!")
        
        if save_data:
            return t_history, q_history, dq_history, ddq_history
    
    
    def get_end_effector_pose(self, q=None):
        """
        Get current end-effector position and orientation
        
        Args:
            q: Joint configuration (uses current if None)
            
        Returns:
            position: [x, y, z]
            orientation: rotation matrix 3x3
        """
        if q is not None:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
        
        # Get end-effector frame (assuming last frame is end-effector)
        ee_frame_id = self.model.getFrameId("ee_link")
        ee_pose = self.data.oMf[ee_frame_id]
        
        position = ee_pose.translation
        orientation = ee_pose.rotation
        
        return position, orientation
    
    def gravity_compensation_torques(self, q):
        """
        Compute gravity compensation torques
        
        Args:
            q: Joint positions
            
        Returns:
            tau_gravity: Torques to compensate gravity
        """
        return pin.rnea(self.model, self.data, q, np.zeros(self.nv), np.zeros(self.nv))