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

# =========================================================================
# STAGE 0: INITIALIZATION & SNAPSHOT
# =========================================================================
if p.isConnected():
    p.disconnect()

# physicsClient = p.connect(p.GUI) # Fast background mode
physicsClient = p.connect(p.DIRECT) # Fast background mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

path = Path('/mnt/c/Users/saharl/Documents/V3.5/closed_loop_test/leg_urdf/robot/urdf/robot.urdf')
robot_id = p.loadURDF(str(path), basePosition=[0, 0, 0], useFixedBase=True, globalScaling=10.0)

joint_dict = {}
link_dict = {}
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    child_link_name = info[12].decode('utf-8')
    joint_dict[joint_name] = i
    link_dict[child_link_name] = i

constraint_1 = p.createConstraint(robot_id, link_dict['loc1'], robot_id, link_dict['loc1_'], p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [0, 0, 0])
constraint_2 = p.createConstraint(robot_id, link_dict['loc2'], robot_id, link_dict['loc2_'], p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [0, 0, 0])
p.changeConstraint(constraint_1, maxForce=100000)
p.changeConstraint(constraint_2, maxForce=100000)

for j in range(num_joints):
    p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)
    
drive_down_idx = joint_dict['base_link_to_new_link']
drive_up_idx = joint_dict['motor_up_joint']
pitch_idx = joint_dict['pitch_joint']
roll_idx = joint_dict['roll_joint']

# ⭐ THE MAGIC FIX: Save a perfect snapshot of the clean robot ⭐
clean_state_id = p.saveState()


# =========================================================================
# STAGE 1: AUTO-FIND MOTOR BOUNDS USING INVERSE KINEMATICS
# =========================================================================
print("STAGE 1: Tracing Ellipse Perimeter to find Motor Bounds...")

PITCH_MIN = 20.0
PITCH_MAX = -50.0
ROLL_MIN = -20.0
ROLL_MAX = 20.0

# Auto-calculate the center points and radii for the offset ellipse
PITCH_CENTER = (PITCH_MAX + PITCH_MIN) / 2.0  
PITCH_RADIUS = (PITCH_MAX - PITCH_MIN) / 2.0  

ROLL_CENTER = (ROLL_MAX + ROLL_MIN) / 2.0     
ROLL_RADIUS = (ROLL_MAX - ROLL_MIN) / 2.0     

# --- NEW: Calculate the exact starting angle where Pitch = 0 ---
# cos(theta) = (Target - Center) / Radius
theta_start = np.arccos((0.0 - PITCH_CENTER) / PITCH_RADIUS)

# Generate points starting from that angle and doing one full 360-degree sweep
theta = np.linspace(theta_start, theta_start + 2 * np.pi, 50)

# Calculate the perimeter targets
pitch_perimeter = np.deg2rad(PITCH_CENTER + PITCH_RADIUS * np.cos(theta))
roll_perimeter = np.deg2rad(ROLL_CENTER + ROLL_RADIUS * np.sin(theta))

mot_down_perim = []
mot_up_perim = []

# time.sleep(2.)
for p_target, r_target in zip(pitch_perimeter, roll_perimeter):
    p.setJointMotorControl2(robot_id, pitch_idx, p.POSITION_CONTROL, targetPosition=p_target, force=5000)
    p.setJointMotorControl2(robot_id, roll_idx, p.POSITION_CONTROL, targetPosition=r_target, force=5000)
    
    for _ in range(150):
        p.stepSimulation()
        # time.sleep(1./240.)
        
    mot_down_perim.append(p.getJointState(robot_id, drive_down_idx)[0])
    mot_up_perim.append(p.getJointState(robot_id, drive_up_idx)[0])

MD_MIN, MD_MAX = min(mot_down_perim), max(mot_down_perim)
MU_MIN, MU_MAX = min(mot_up_perim), max(mot_up_perim)

print(f"Auto-Bounds Found! Motor Down: [{np.rad2deg(MD_MIN):.1f}°, {np.rad2deg(MD_MAX):.1f}°]")
print(f"Auto-Bounds Found! Motor Up: [{np.rad2deg(MU_MIN):.1f}°, {np.rad2deg(MU_MAX):.1f}°]")
# time.sleep(2.)

# =========================================================================
# STAGE 2: FORWARD KINEMATICS SWEEP & ELLIPSE FILTER
# =========================================================================
print("\nSTAGE 2: Sweeping FK Grid within auto-bounds...")

# ⭐ THE MAGIC FIX: Rewind time back to the pristine snapshot ⭐
p.restoreState(clean_state_id)

# Release the passive joints back to normal (just in case)
p.setJointMotorControl2(robot_id, pitch_idx, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot_id, roll_idx, p.VELOCITY_CONTROL, force=0)

padding = 0.05 
mot_down_angles = np.linspace(MD_MIN - padding, MD_MAX + padding, 20)
mot_up_angles = np.linspace(MU_MIN - padding, MU_MAX + padding, 20)

kinematic_data = []

for i, target_mot_down in enumerate(mot_down_angles):
    current_mot_up_angles = mot_up_angles if i % 2 == 0 else mot_up_angles[::-1]
        
    for target_mot_up in current_mot_up_angles:
        p.setJointMotorControl2(robot_id, drive_down_idx, p.POSITION_CONTROL, targetPosition=target_mot_down, force=5000)
        p.setJointMotorControl2(robot_id, drive_up_idx, p.POSITION_CONTROL, targetPosition=target_mot_up, force=5000)
        
        for _ in range(150):
            p.stepSimulation()
            # time.sleep(1./2400.)
            
        pitch_deg = np.rad2deg(p.getJointState(robot_id, pitch_idx)[0])
        roll_deg = np.rad2deg(p.getJointState(robot_id, roll_idx)[0])
        mot_down_deg = np.rad2deg(target_mot_down)
        mot_up_deg = np.rad2deg(target_mot_up)
        
        # MATH FILTER: Does this point fall inside the OFFSET ellipse?
        ellipse_val = ((pitch_deg - PITCH_CENTER) / PITCH_RADIUS)**2 + ((roll_deg - ROLL_CENTER) / ROLL_RADIUS)**2
        
        if ellipse_val <= 1.05: # Allowing 5% tolerance at the edges
            kinematic_data.append([mot_down_deg, mot_up_deg, pitch_deg, roll_deg])

p.disconnect()
print(f"Sweep complete! Saved {len(kinematic_data)} valid points.")
    
# =========================================================================
# STAGE 3: PLOTTING THE ELLIPTICAL WORKSPACE
# =========================================================================
df_valid = np.array(kinematic_data)
MD = df_valid[:, 0]
MU = df_valid[:, 1]
PITCH = df_valid[:, 2]
ROLL = df_valid[:, 3]

# 1. Create a dense 2D meshgrid based on the min/max of the valid motors
md_grid = np.linspace(MD.min(), MD.max(), 100)
mu_grid = np.linspace(MU.min(), MU.max(), 100)
X, Y = np.meshgrid(md_grid, mu_grid)

# 2. Interpolate the valid Pitch and Roll data onto the square grid.
# The 'linear' method naturally assigns NaN to the missing corners outside your ellipse!
Z_PITCH = griddata((MD, MU), PITCH, (X, Y), method='linear')
Z_ROLL = griddata((MD, MU), ROLL, (X, Y), method='linear')

fig = plt.figure()
# Plot 1: The resulting Pitch surface based on Motor inputs
ax1 = fig.add_subplot(111, projection='3d')
scat1 = ax1.plot_surface(X, Y, Z_PITCH, cmap='coolwarm', alpha=1.0)
ax1.set_title('FK: Motor Inputs to Pitch')
ax1.set_xlabel('Motor Down (deg)')
ax1.set_ylabel('Motor Up (deg)')
ax1.set_zlabel('Pitch (deg)')
fig.colorbar(scat1, ax=ax1, shrink=0.5)

fig = plt.figure()
# Plot 2: The resulting Roll surface based on Motor inputs
ax1 = fig.add_subplot(111, projection='3d')
scat1 = ax1.plot_surface(X, Y, Z_ROLL, cmap='coolwarm', alpha=1.0)
ax1.set_title('FK: Motor Inputs to Roll')
ax1.set_xlabel('Motor Down (deg)')
ax1.set_ylabel('Motor Up (deg)')
ax1.set_zlabel('Roll (deg)')
fig.colorbar(scat1, ax=ax1, shrink=0.5)

# Plot 2: 2D Top-Down View proving it formed a perfect ellipse
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.scatter(Z_PITCH, Z_ROLL, c='purple', alpha=0.6)
ax2.set_title('Top-Down View of Gimbal Space')
ax2.set_xlabel('Pitch (deg)')
ax2.set_ylabel('Roll (deg)')
ax2.grid(True)
ax2.set_aspect('equal', 'box')

plt.tight_layout()
plt.show()