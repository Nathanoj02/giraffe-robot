import numpy as np
import pinocchio as pin
from utils.ros_publish import RosPub  # Assuming you have this utility

class SimplePinocchioSimulator:
    def __init__(self, urdf_path):
        """Initialize robot model and simulator state"""
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
        self.model = self.robot.model
        self.data = self.robot.data
        
        # Initialize state
        self.q = pin.neutral(self.model)  # Joint positions
        self.dq = np.zeros(self.model.nv)  # Joint velocities
        self.ddq = np.zeros(self.model.nv)  # Joint accelerations
        
        # Visualization setup
        self.ros_pub = RosPub("giraffe")
        
    def compute_gravity_compensation(self):
        """Compute torques to counteract gravity using RNEA"""
        return pin.rnea(self.model, self.data, self.q, 
                       np.zeros(self.model.nv),  # Zero velocity
                       np.zeros(self.model.nv))  # Zero acceleration

    def step(self, tau, dt=0.001):
        """Simulate one time step using RNEA for dynamics"""
        # Forward dynamics (compute acceleration)
        self.ddq = pin.aba(self.model, self.data, self.q, self.dq, tau)
        
        # Semi-implicit Euler integration
        self.dq += self.ddq * dt
        self.q = pin.integrate(self.model, self.q, self.dq * dt)
        
        # Update kinematics for visualization
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def run_simulation(self, control_func, duration=5.0, dt=0.01):
        """Main simulation loop"""
        steps = int(duration / dt)
        
        for i in range(steps):
            # Get control torques (user-provided function)
            tau = control_func(i*dt, self.q.copy(), self.dq.copy())
            
            # Simulation step
            self.step(tau, dt)
            
            # ROS visualization (publish at ~50Hz)
            if i % int(0.05/dt) == 0:  # Throttle to ~50Hz
                self.ros_pub.publish(self.robot, self.q)
                
        self.ros_pub.deregister_node()

if __name__ == "__main__":
    # Example usage
    urdf_path = "../urdf/giraffe.urdf"
    sim = SimplePinocchioSimulator(urdf_path)
    
    # Define a simple gravity-compensated position controller
    def simple_controller(t, q, dq):
        q_target = np.array([0.2, 0.1, 0.5, 0.1, -0.05])  # Example target
        Kp = 10.0  # Proportional gain
        return Kp * (q_target - q) + sim.compute_gravity_compensation()
    
    # Run simulation
    sim.run_simulation(simple_controller, duration=10.0)