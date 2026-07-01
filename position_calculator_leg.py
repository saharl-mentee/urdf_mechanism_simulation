from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import pandas as pd


def pivot_df(df, index_col, columns_col, values_col):
    z_pivot = df.pivot(
    index=index_col,   # This becomes the X-axis
    columns=columns_col,   # This becomes the Y-axis
    values=values_col        # This becomes the Z-axis height
    )

    # --- 2. Create the Meshgrids ---
    # Extract the unique 1D coordinates from the pivot table
    x_1d = z_pivot.index.values
    y_1d = z_pivot.columns.values

    # Create the 2D X and Y grids. 
    # We use indexing='ij' so the shapes perfectly match our z_pivot matrix!
    X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')

    # Extract the 2D Z grid directly from the pivot table values
    Z = z_pivot.values
    return X, Y, Z

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
mot_down_angles = np.deg2rad(np.linspace(-30, 30, 10))
mot_up_angles = np.deg2rad(np.linspace(-30, 30, 10))

# --- 7. Run the Grid Simulation ---
print("Starting IK Grid Calculation...")
print("-" * 75)

# Optional list to store results if you want to plot or save them later
kinematic_data = []
pitch_data = np.zeros((len(mot_down_angles), len(mot_up_angles)))
roll_data = np.zeros((len(mot_down_angles), len(mot_up_angles)))

try:
    for i, target_mot_down in enumerate(mot_down_angles):
        # ZIGZAG LOGIC: Reverse the roll angles every other pitch step
        if i % 2 == 0:
            current_roll_angles = mot_up_angles
        else:
            current_roll_angles = mot_up_angles[::-1] # Read the array backward
            
        for j, target_mot_up in enumerate(current_roll_angles):
            
            # 1. Drive Pitch and Roll to the target grid position
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=drive_down_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_mot_down,
                force=5000 
            )
            
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=drive_up_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_mot_up,
                force=5000 
            )
            
            # 2. Step the simulation multiple times to let the constraints settle
            for _ in range(100):
                p.stepSimulation()
                # Uncomment the next line ONLY if you want to watch it move slowly in the GUI
                # time.sleep(1./2400.) 
            
            # 3. Read the resulting passive angles of the motors
            pitch_state = p.getJointState(robot_id, pitch_idx)[0]
            roll_state = p.getJointState(robot_id, roll_idx)[0]

            # 4. Save and Print the data
            kinematic_data.append([
                np.rad2deg(target_mot_down),
                np.rad2deg(target_mot_up),
                np.rad2deg(pitch_state),
                np.rad2deg(roll_state)
            ])
            
            print(f"Req Mot_Dn: {np.rad2deg(target_mot_down):+6.1f}° | Req Mot_Up: {np.rad2deg(target_mot_up):+6.1f}° || "
                  f"Result Pitch: {np.rad2deg(pitch_state):+6.1f}° | Result Roll: {np.rad2deg(roll_state):+6.1f}°")

    print("-" * 75)
    print(f"Grid calculation complete. Collected {len(kinematic_data)} data points.")
    
    p.disconnect()
    print("Simulator closed.")
    
    df = pd.DataFrame(
        kinematic_data, 
        columns=['Motor Down (deg)', 'Motor Up (deg)', 'Pitch (deg)', 'Roll (deg)']
    )
    
    X, Y, pitch = pivot_df(df, 'Motor Down (deg)', 'Motor Up (deg)', 'Pitch (deg)')
    X, Y, roll = pivot_df(df, 'Motor Down (deg)', 'Motor Up (deg)', 'Roll (deg)')


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np.rad2deg(X), np.rad2deg(Y), pitch, 
                       cmap='coolwarm',    # Color map
                       linewidth=0,        # Removes grid lines on the surface
                       antialiased=True,   # Smooths the edges
                       shade=True,         # Enables lighting/shading
                       alpha=0.9)          # Slight transparency
    plt.title('pitch [deg]')
    plt.xlabel('motor down angle [deg]')
    plt.ylabel('motor up angle [deg]')
    plt.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to show the scale
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np.rad2deg(X), np.rad2deg(Y), roll, 
                       cmap='coolwarm',    # Color map
                       linewidth=0,        # Removes grid lines on the surface
                       antialiased=True,   # Smooths the edges
                       shade=True,         # Enables lighting/shading
                       alpha=0.9)          # Slight transparency
    plt.title('roll [deg]')
    plt.xlabel('motor down angle [deg]')
    plt.ylabel('motor up angle [deg]')
    plt.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to show the scale
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()

except KeyboardInterrupt:
    print("\nSimulation stopped.")
    p.disconnect()