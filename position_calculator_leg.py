from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt

# --- 1. Initialize PyBullet ---
physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --- 2. Load the URDF ---
path = Path('/mnt/c/Users/saharl/Documents/V3.5/closed_loop_test/leg_urdf/robot/urdf/robot.urdf')
robot_id = p.loadURDF(str(path), basePosition=[0, 0, 0], useFixedBase=True, globalScaling=10.0)

# --- 3. Dynamically Map Joints and Links ---
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
constraint_1 = p.createConstraint(
    parentBodyUniqueId=robot_id, parentLinkIndex=link_dict['loc1'],
    childBodyUniqueId=robot_id, childLinkIndex=link_dict['loc1_'],
    jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]   
)

constraint_2 = p.createConstraint(
    parentBodyUniqueId=robot_id, parentLinkIndex=link_dict['loc2'],
    childBodyUniqueId=robot_id, childLinkIndex=link_dict['loc2_'],
    jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]   
)

p.changeConstraint(constraint_1, maxForce=100000)
p.changeConstraint(constraint_2, maxForce=100000)

# --- 5. Setup Control ---
drive_down_idx = joint_dict['base_link_to_new_link']
drive_up_idx = joint_dict['motor_up_joint']
pitch_idx = joint_dict['pitch_joint']
roll_idx = joint_dict['roll_joint']

# Turn off default motors for ALL joints so they are fully passive
for j in range(num_joints):
    p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

# --- 6. Generate the Grid ---
# Define the range of motion for pitch and roll in degrees, then convert to radians
# Example: -15 to +15 degrees, taking 5 steps (e.g., -15, -7.5, 0, 7.5, 15)
pitch_angles = np.deg2rad(np.arange(-20, 50, 2))
roll_angles = np.deg2rad(np.arange(-20, 20, 2))

# --- 7. Run the Grid Simulation ---
print("Starting IK Grid Calculation...")
print("-" * 75)

# Optional list to store results if you want to plot or save them later
kinematic_data = []
motor_up_data = np.zeros((len(pitch_angles), len(roll_angles)))
motor_down_data = np.zeros((len(pitch_angles), len(roll_angles)))

try:
    for i, target_pitch in enumerate(pitch_angles):
        # ZIGZAG LOGIC: Reverse the roll angles every other pitch step
        if i % 2 == 0:
            current_roll_angles = roll_angles
        else:
            current_roll_angles = roll_angles[::-1] # Read the array backward
            
        for j, target_roll in enumerate(current_roll_angles):
            
            # 1. Drive Pitch and Roll to the target grid position
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=pitch_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pitch,
                force=5000 
            )
            
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=roll_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_roll,
                force=5000 
            )
            
            # 2. Step the simulation multiple times to let the constraints settle
            for _ in range(100):
                p.stepSimulation()
                # Uncomment the next line ONLY if you want to watch it move slowly in the GUI
                # time.sleep(1./2400.) 
            
            # 3. Read the resulting passive angles of the motors
            mot_down_state = p.getJointState(robot_id, drive_down_idx)[0]
            mot_up_state = p.getJointState(robot_id, drive_up_idx)[0]
            motor_up_data[i,j] = np.rad2deg(mot_up_state)
            motor_down_data[i,j] = np.rad2deg(mot_down_state)

            # 4. Save and Print the data
            kinematic_data.append({
                "pitch_deg": np.rad2deg(target_pitch),
                "roll_deg": np.rad2deg(target_roll),
                "mot_down_deg": np.rad2deg(mot_down_state),
                "mot_up_deg": np.rad2deg(mot_up_state)
            })
            
            print(f"Target Pitch: {np.rad2deg(target_pitch):+6.1f}° | Target Roll: {np.rad2deg(target_roll):+6.1f}° || "
                  f"Req Mot_Dn: {np.rad2deg(mot_down_state):+6.1f}° | Req Mot_Up: {np.rad2deg(mot_up_state):+6.1f}°")

    print("-" * 75)
    print(f"Grid calculation complete. Collected {len(kinematic_data)} data points.")
    
    i=0
    j=0
    # Keep window open at the end until user closes it
    while i < len(pitch_angles) and j < len(roll_angles):
        p.stepSimulation()
        time.sleep(1./240.)
        i +=1
        j +=1
    
    np.save("kinematic_results_up.npy", motor_up_data)
    np.save("kinematic_results_down.npy", motor_down_data)
    p.disconnect()
    print("Simulator closed.")
    
    x, y = np.meshgrid(pitch_angles, roll_angles, indexing='ij')  # Create a meshgrid for plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np.rad2deg(x), np.rad2deg(y), motor_up_data, 
                       cmap='coolwarm',    # Color map
                       linewidth=0,        # Removes grid lines on the surface
                       antialiased=True,   # Smooths the edges
                       shade=True,         # Enables lighting/shading
                       alpha=0.9)          # Slight transparency
    plt.title('motor up')
    plt.xlabel('pitch angle')
    plt.ylabel('roll angle')
    plt.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to show the scale
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()

except KeyboardInterrupt:
    print("\nSimulation stopped.")
    p.disconnect()