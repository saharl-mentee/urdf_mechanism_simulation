from pathlib import Path
import pybullet as p
import pybullet_data
import time
import numpy as np
import math

# --- 1. Initialize PyBullet ---
physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --- 2. Load the URDF ---
# Load the URDF. We use useFixedBase=True to ground the base_link.
path = Path('/mnt/c/Users/saharl/Documents/V3.5/closed_loop_test/URDF/urdf/robot.urdf')
robot_id = p.loadURDF(str(path), basePosition=[0, 0, 0], useFixedBase=True)

# --- 3. Identify Joint and Link Indices ---
# PyBullet assigns link indices that match their parent joint indices.
# Based on the URDF structure:
# Index 0: joint_1 (Drives link1)
# Index 1: joint_2 (Drives link_2)
# Index 2: joint_3 (Drives link_3)
# Index 3: joint_4 (Fixed to link_4)
# Index 4: joint_4_ (Fixed to link_4_)

driving_joint_index = 0  # joint_1
link_4_index = 3
link_4_underscore_index = 4

# --- 4. Close the Kinematic Loop ---
# We create a constraint binding link_4 and link_4_ together. 
# Since both links have dummy inertial origins at 0,0,0, we connect their local origins.
constraint_id = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=link_4_index,
    childBodyUniqueId=robot_id,
    childLinkIndex=link_4_underscore_index,
    jointType=p.JOINT_POINT2POINT,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0], 
    childFramePosition=[0, 0, 0]   
)

# Make the loop closure highly rigid
p.changeConstraint(constraint_id, maxForce=100000)

# --- 5. Setup Control ---
num_joints = p.getNumJoints(robot_id)

# Turn off default motors for the passive revolute joints (joint_2 and joint_3)
# so the constraint can freely pull them into the correct geometric positions.
for j in range(num_joints):
    if j != driving_joint_index:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

# # --- 6. Run the Simulation ---
# print("Starting simulation. Press Ctrl+C in the terminal to stop.")

# try:
#     while True:
#         # Oscillate joint_1 within its URDF limits (-1.57 to 1.57 rad)
#         # We use a smaller range (e.g., +/- 1.0 rad) to prevent snapping at mathematical singularities
#         target_angle = math.sin(time.time() * 2) * 1.0 
        
#         # Drive joint_1
#         p.setJointMotorControl2(
#             bodyUniqueId=robot_id,
#             jointIndex=driving_joint_index,
#             controlMode=p.POSITION_CONTROL,
#             targetPosition=target_angle,
#             force=1000 
#         )
        
#         # Step the physics engine
#         p.stepSimulation()
        
#         # Read the resulting positions of the active mechanism joints
#         joint_states = p.getJointStates(robot_id, [0, 1, 2])
#         j1_pos = joint_states[0][0]
#         j2_pos = joint_states[1][0]
#         j3_pos = joint_states[2][0]
        
#         # Print the kinematics
#         print(f"Driven j1: {j1_pos:+.2f} rad | Passive j2: {j2_pos:+.2f} rad | Passive j3: {j3_pos:+.2f} rad", end='\r')
        
#         # Real-time stepping
#         time.sleep(1./240.) 

# except KeyboardInterrupt:
#     print("\nSimulation stopped.")
#     p.disconnect()
    
# --- 6. Run the Simulation (Fixed Position) ---
print("Moving to fixed position. Press Ctrl+C in the terminal to stop.")

# Set your desired fixed angle in radians (e.g., 0.5 rad)
fixed_angle = np.deg2rad(20)

try:
    while True:
        # Command the driving joint to hold the fixed angle
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=driving_joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=fixed_angle,
            force=1000 # Make sure this is high enough to resist gravity and constraints
        )
        
        # Step the physics engine
        p.stepSimulation()
        
        # Read the resulting positions of the active mechanism joints
        joint_states = p.getJointStates(robot_id, [0, 1, 2])
        j1_pos = joint_states[0][0]
        j2_pos = joint_states[1][0]
        j3_pos = joint_states[2][0]
        
        # Print the kinematics
        print(f"Holding j1: {np.rad2deg(j1_pos):+.2f} deg | Passive j2: {np.rad2deg(j2_pos):+.2f} deg | Passive j3: {np.rad2deg(j3_pos):+.2f} deg", end='\r')
        
        
        link2_state = p.getLinkState(robot_id, 1)
        link2_quaternion = link2_state[5]
        link2_euler = p.getEulerFromQuaternion(link2_quaternion)
        joint2_global_angle = link2_euler[2]
        global_angle_degrees = math.degrees(joint2_global_angle)
        print(f"Joint 2 Global Angle: {global_angle_degrees:.1f}°")
        
        # Real-time stepping
        time.sleep(1./240.)

except KeyboardInterrupt:
    print("\nSimulation stopped.")
    p.disconnect()