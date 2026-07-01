from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata


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


# =========================================================================
# STAGE 1: AUTO-FIND MOTOR BOUNDS USING INVERSE KINEMATICS
# =========================================================================
print("STAGE 1: Tracing Ellipse Perimeter to find Motor Bounds...")
PITCH_MAX_LIMIT = 20.0
ROLL_MAX_LIMIT = 20.0

# Generate points along the perimeter of the ellipse (0 to 2*PI)
theta = np.linspace(0, 2 * np.pi, 50)
pitch_perimeter = np.deg2rad(PITCH_MAX_LIMIT * np.cos(theta))
roll_perimeter = np.deg2rad(ROLL_MAX_LIMIT * np.sin(theta))

mot_down_perim = []
mot_up_perim = []

for p_target, r_target in zip(pitch_perimeter, roll_perimeter):
    p.setJointMotorControl2(robot_id, pitch_idx, p.POSITION_CONTROL, targetPosition=p_target, force=5000)
    p.setJointMotorControl2(robot_id, roll_idx, p.POSITION_CONTROL, targetPosition=r_target, force=5000)
    
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.) 
        
    mot_down_perim.append(p.getJointState(robot_id, drive_down_idx)[0])
    mot_up_perim.append(p.getJointState(robot_id, drive_up_idx)[0])

# Auto-calculate the bounding box for the motors
MD_MIN, MD_MAX = min(mot_down_perim), max(mot_down_perim)
MU_MIN, MU_MAX = min(mot_up_perim), max(mot_up_perim)

print(f"Auto-Bounds Found! Motor Down: [{np.rad2deg(MD_MIN):.1f}°, {np.rad2deg(MD_MAX):.1f}°]")
print(f"Auto-Bounds Found! Motor Up: [{np.rad2deg(MU_MIN):.1f}°, {np.rad2deg(MU_MAX):.1f}°]")

# Release the passive joints back to normal before Stage 2
p.setJointMotorControl2(robot_id, pitch_idx, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot_id, roll_idx, p.VELOCITY_CONTROL, force=0)

# --- 6. Generate the Grid ---
# Define the range of motion for pitch and roll in degrees, then convert to radians
# Example: -15 to +15 degrees, taking 5 steps (e.g., -15, -7.5, 0, 7.5, 15)
mot_down_angles = np.deg2rad(np.linspace(-150, 150, 100))
mot_up_angles = np.deg2rad(np.linspace(-150, 150, 100))

PITCH_MIN = -20.0
PITCH_MAX = 50.0
ROLL_MIN = -20.0
ROLL_MAX = 20.0

# --- 7. Run the Grid Simulation ---
print("Starting IK Grid Calculation...")
print("-" * 75)

# Optional list to store results if you want to plot or save them later
kinematic_data = []
pitch_data = np.zeros((len(mot_down_angles), len(mot_up_angles)))
roll_data = np.zeros((len(mot_down_angles), len(mot_up_angles)))
out_of_bounds_count = 0

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

            # Convert to degrees BEFORE appending
            pitch_deg = np.rad2deg(pitch_state)
            roll_deg = np.rad2deg(roll_state)
            mot_down_deg = np.rad2deg(target_mot_down)
            mot_up_deg = np.rad2deg(target_mot_up)
            
            # 4. BOUNDARY FILTER: Only save if within bounds
            # Notice we ONLY append here, and we append the degree variables!
            if (PITCH_MIN <= pitch_deg <= PITCH_MAX) and (ROLL_MIN <= roll_deg <= ROLL_MAX):
                kinematic_data.append([
                    mot_down_deg,
                    mot_up_deg,
                    pitch_deg,
                    roll_deg
                ])
                print(f"✅ SAVED | Mot_Dn: {mot_down_deg:+6.1f}° | Mot_Up: {mot_up_deg:+6.1f}° || "
                      f"Pitch: {pitch_deg:+6.1f}° | Roll: {roll_deg:+6.1f}°")
            else:
                out_of_bounds_count += 1
                print(f"❌ SKIP  | Mot_Dn: {mot_down_deg:+6.1f}° | Mot_Up: {mot_up_deg:+6.1f}° || "
                      f"Pitch: {pitch_deg:+6.1f}° | Roll: {roll_deg:+6.1f}°", end='\r')
            

    print("-" * 75)
    print(f"Grid calculation complete. Collected {len(kinematic_data)} data points.")
    
    p.disconnect()
    print("Simulator closed.")
    
    df = pd.DataFrame(
        kinematic_data, 
        columns=['Motor Down (deg)', 'Motor Up (deg)', 'Pitch (deg)', 'Roll (deg)']
    )
    
    # X, Y, pitch = pivot_df(df, 'Motor Down (deg)', 'Motor Up (deg)', 'Pitch (deg)')
    # X, Y, roll = pivot_df(df, 'Motor Down (deg)', 'Motor Up (deg)', 'Roll (deg)')
    
    X, Y = np.mgrid[df['Motor Down (deg)'].min():df['Motor Down (deg)'].max():200j, 
                    df['Motor Up (deg)'].min():df['Motor Up (deg)'].max():200j]
   
    pitch = griddata((df['Motor Down (deg)'], df['Motor Up (deg)']), df['Pitch (deg)'], (X, Y), method='linear')
    roll = griddata((df['Motor Down (deg)'], df['Motor Up (deg)']), df['Roll (deg)'], (X, Y), method='linear')


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, pitch, 
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
    surf = ax.plot_surface(X, Y, roll, 
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