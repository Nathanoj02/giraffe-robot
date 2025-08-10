# 🦒 Giraffe Robot

## 🚀 Setup
**Prerequisites**: ROS Noetic workspace already configured

```bash
cd ~/itr_ws/src
git clone git@github.com:Nathanoj02/giraffe-robot.git

cd ~/itr_ws
catkin build
```

## 📁 Repo organization
- 🤖 **`urdf/`** - Robot model
- 🐍 **`scripts/`** - Main simulations and implementations  
- 🔧 **`scripts/functions/`** - Core algorithms for project tasks
- 🛠️ **`scripts/utils/`** - Helper functions and ROS interface
- 📊 **`report/`** - Project report

## 🛠️ Useful Commands

### 👁️ Visualization
Visualize URDF in RViz
```bash
roslaunch giraffe-robot visualize.launch
```

### 🔧 URDF Management
Convert from xacro to urdf
```bash
rosrun xacro xacro -o giraffe.urdf giraffe.urdf.xacro
```

## 🎮 Running Simulations
```bash
cd ~/itr_ws/src/giraffe-robot/scripts
```

RNEA dynamics simulation
```bash
python3 main_rnea.py
```

Task space control simulation
```bash
python3 main_task_space.py
```

## 🤖 Robot Specifications
- **DOF**: 5 (base rotation, shoulder tilt, telescopic extension, 2x wrist joints)
- **Mounting**: 🏠 Ceiling-mounted configuration
- **Workspace**: 📏 ~6.5m radius with telescopic extension
- **Control**: 🎯 Position and orientation control

## 📋 Tasks Implemented
1. ➡️ **Forward Kinematics** - Compute end-effector pose
2. ⬅️ **Inverse Kinematics** - Solve for joint angles
3. ⚙️ **RNEA Dynamics** - Recursive Newton-Euler Algorithm
4. 🎯 **Task Space Control** - Cartesian trajectory following
5. 📈 **Trajectory Generation** - 5th-order polynomials
6. 🎛️ **PD Control** - Position/orientation control
7. 🔄 **Null-space Control** - Redundancy resolution

## ⚙️ Configuration
- `scripts/conf.py` - Main configuration file
- `urdf/giraffe.urdf.xacro` - Robot geometry and limits
