from pathlib import Path

import pybullet as p
import pybullet_data
import time
import math

# --- 1. Initialize PyBullet ---
physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --- 2. Load the URDF ---
path = Path('/mnt/c/Users/saharl/Documents/V3.5/closed_loop_test/leg_urdf/robot/urdf/robot.urdf')
robot_id = p.loadURDF(str(path), basePosition=[0, 0, 0], useFixedBase=True)

# --- 3. Dynamically Map Joints and Links ---
# This dictionary will let us look up the PyBullet index using the URDF names
joint_dict = {}
link_dict = {}

num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    child_link_name = info[12].decode('utf-8')
    
    joint_dict[joint_name] = i
    link_dict[child_link_name] = i

# --- 4. Close the Kinematic Loops ---
# Loop 1: loc1 to loc1_
constraint_1 = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=link_dict['loc1'],
    childBodyUniqueId=robot_id,
    childLinkIndex=link_dict['loc1_'],
    jointType=p.JOINT_POINT2POINT,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0], 
    childFramePosition=[0, 0, 0]   
)

# Loop 2: loc2 to loc2_
constraint_2 = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=link_dict['loc2'],
    childBodyUniqueId=robot_id,
    childLinkIndex=link_dict['loc2_'],
    jointType=p.JOINT_POINT2POINT,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0], 
    childFramePosition=[0, 0, 0]   
)

# Make the closures rigid
p.changeConstraint(constraint_1, maxForce=100000)
p.changeConstraint(constraint_2, maxForce=100000)

# --- 5. Setup Control ---
# Identify the main driving motors
# "base_link_to_new_link" drives the down-crank, "motor_up_joint" drives the up-crank
drive_down_idx = joint_dict['base_link_to_new_link']
drive_up_idx = joint_dict['motor_up_joint']

# The central gimbal joints that will be passively pushed/pulled
pitch_idx = joint_dict['pitch_joint']
roll_idx = joint_dict['roll_joint']

# Turn off default motors for ALL joints first so they are fully passive
for j in range(num_joints):
    p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

# --- 6. Run the Simulation ---
print("Starting parallel mechanism simulation. Press Ctrl+C to stop.")

try:
    while True:
        # Generate two sine waves, slightly out of phase, to drive the two motors
        t = time.time()
        target_down = math.sin(t * 2) * 0.5 
        target_up = math.sin((t * 2) + 1.0) * 0.5 
        
        # Drive Motor Down
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=drive_down_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_down,
            force=2500 
        )
        
        # Drive Motor Up
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=drive_up_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_up,
            force=2500 
        )
        
        # Step the physics engine
        p.stepSimulation()
        
        # Read the resulting passive angles of the central Pitch and Roll joints
        pitch_state = p.getJointState(robot_id, pitch_idx)[0]
        roll_state = p.getJointState(robot_id, roll_idx)[0]
        
        # Print the kinematics
        print(f"Mot_Dn: {target_down:+.2f} | Mot_Up: {target_up:+.2f} || Result Pitch: {pitch_state:+.2f} | Result Roll: {roll_state:+.2f}", end='\r')
        
        # Real-time stepping
        time.sleep(1./240.) 

except KeyboardInterrupt:
    print("\nSimulation stopped.")
    p.disconnect()