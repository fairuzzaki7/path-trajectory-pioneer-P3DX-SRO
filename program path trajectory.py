# %%
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# 1. CONNECT & SIMULATION SETUP (SYNCHRONOUS MODE)
client = RemoteAPIClient()
sim = client.require('sim')

# Mengaktifkan stepped mode agar Python dan CoppeliaSim berjalan sinkron
sim.setStepping(True) 
sim.startSimulation()
print("Simulation Started (Synchronous Mode)")

# TRANSFORMATION MATRIX FUNCTION
def transformMat(yaw, tx, ty, tz):
    T = np.array([
        [math.cos(yaw), -math.sin(yaw), 0, tx],
        [math.sin(yaw),  math.cos(yaw), 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

# 2. OBJECT & SENSOR HANDLES
robot = sim.getObject('/PioneerP3DX')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')
left_motor  = sim.getObject('/PioneerP3DX/leftMotor')
LH_Handle   = sim.getObject('/LH')
perp_Handle = sim.getObject('/Perp')

# SENSOR HANDLE (Index: 0, 3, 4, 7)
sensor_handles = []
sensor_index = [0, 3, 4, 7]
for idx in sensor_index:
    sensor = sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{idx}]')
    sensor_handles.append(sensor)

# PATH POINTS LOADING
path_points = []
for i in range(39):
    disc = sim.getObject(f'/Disc[{i}]')
    pos = sim.getObjectPosition(disc, sim.handle_world)
    path_points.append(np.array(pos).reshape((3, 1)))
    
print(f"Loaded {len(path_points)} path points")

# ROBOT CONFIGURATION PARAMETERS
rw = 0.195 / 2
rb = 0.318 / 2
LH_distance = 0.7

# DATA STORAGE FOR PLOTTING
traj_x = []
traj_y = []
traj_yaw = []
map_x = []
map_y = []

# 3. MAIN SIMULATION LOOP
try:
    # Menggunakan waktu simulasi internal CoppeliaSim agar lebih presisi
    while sim.getSimulationTime() < 60:

        # AMBIL ROBOT POSE
        robot_pos = sim.getObjectPosition(robot, sim.handle_world)
        robot_ori = sim.getObjectOrientation(robot, sim.handle_world)

        # Simpan data trajectory robot
        traj_x.append(robot_pos[0])
        traj_y.append(robot_pos[1])
        traj_yaw.append(robot_ori[2])

        # HITUNG LH POSITION
        T_world_robot = transformMat(
            robot_ori[2],
            robot_pos[0],
            robot_pos[1],
            robot_pos[2]
        )

        LH_world = T_world_robot @ np.array([
            [LH_distance],
            [0],
            [0],
            [1]
        ])

        LH_world = LH_world[:3, :]
        LH_xy = LH_world[:2, :]

        # CARI TITIK PERP TERDEKAT TERHADAP LH
        best_distance = float('inf')
        best_proj = None
        for i in range(len(path_points)):
            A = path_points[i][:2]
            B = path_points[(i + 1) % len(path_points)][:2]

            AB = B - A
            ALH = LH_xy - A

            t = np.dot(ALH.T, AB) / (np.linalg.norm(AB)**2)
            t = t[0][0]
            t = max(0.0, min(1.0, t)) # Clamp t antara 0 dan 1
            
            proj = A + (t * AB)
            dist = np.linalg.norm(LH_xy - proj)

            if dist < best_distance:
                best_distance = dist
                best_proj = proj

        # PERP POSITION WORLD
        perp_world = np.array([
            [best_proj[0][0]],
            [best_proj[1][0]],
            [robot_pos[2]]
        ])

        # UPDATE POSISI VISUAL DI COPPELIASIM
        sim.setObjectPosition(LH_Handle, sim.handle_world, LH_world.flatten().tolist())
        sim.setObjectPosition(perp_Handle, sim.handle_world, perp_world.flatten().tolist())

        # ROBOT KINEMATICS CONTROL
        perp_robot = np.linalg.inv(T_world_robot) @ np.array([
            [perp_world[0][0]],
            [perp_world[1][0]],
            [perp_world[2][0]],
            [1]
        ])
        
        error_x = perp_robot[0][0]
        error_y = perp_robot[1][0]
        heading_error = math.atan2(error_y, error_x)
        
        v = 0.4                  # Linear velocity (m/s)
        omega = 2.0 * heading_error  # Angular velocity (rad/s)
        
        # Hitung kecepatan sudut roda kanan dan kiri
        wr = (v + rb * omega) / rw
        wl = (v - rb * omega) / rw
        
        # Kirim perintah kecepatan roda ke simulator
        sim.setJointTargetVelocity(right_motor, wr)
        sim.setJointTargetVelocity(left_motor, wl)
        
        # ENVIRONMENT MAPPING MENGGUNAKAN SENSOR ULTRASONIK
        for sensor in sensor_handles:
            result, distance, detectedPoint, _, _ = sim.readProximitySensor(sensor)

            if result > 0:
                sensor_matrix = sim.getObjectMatrix(sensor, sim.handle_world)
                T_sensor_3x4 = np.array(sensor_matrix).reshape(3, 4)
                
                T_sensor = np.vstack((T_sensor_3x4, [0, 0, 0, 1]))

                point_sensor = np.array([
                    [detectedPoint[0]],
                    [detectedPoint[1]],
                    [detectedPoint[2]],
                    [1]
                ])
                
                point_world = T_sensor @ point_sensor
                map_x.append(point_world[0][0])
                map_y.append(point_world[1][0])

        sim.step()

# SAFE EXIT & CLEANUP
finally:
    sim.setJointTargetVelocity(right_motor, 0)
    sim.setJointTargetVelocity(left_motor, 0)
    sim.stopSimulation()
    print("\nSimulation Stopped Gracefully")

# 4. UPGRADED & AESTHETIC VISUALIZATION
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

path_x = [p[0][0] for p in path_points]
path_y = [p[1][0] for p in path_points]
path_x.append(path_points[0][0][0])
path_y.append(path_points[0][1][0])

# PLOT 1: Reference Path (Garis putus-putus abu-abu netral)
ax.plot(path_x, path_y, linestyle='--', color='#7f8c8d', linewidth=2.0, 
        label='Reference Path', zorder=2)

# PLOT 2: Robot Trajectory (Garis biru tua solid dan tebal)
ax.plot(traj_x, traj_y, color='#1b4f72', linewidth=3.0, 
        label='Robot Actual Trajectory', zorder=4)

# PLOT 3: Obstacle Mapping (Warna coral kemerahan dengan transparansi alpha)
if map_x and map_y:
    ax.scatter(map_x, map_y, color='#e74c3c', s=8, alpha=0.3, 
               edgecolors='none', label='Detected Obstacles (Sensor)', zorder=1)

# PLOT 4: Robot Heading Direction 
step = 20  
if len(traj_x) > 0:
    q_x = traj_x[::step]
    q_y = traj_y[::step]
    q_u = [math.cos(y) for y in traj_yaw[::step]]
    q_v = [math.sin(y) for y in traj_yaw[::step]]
    ax.quiver(q_x, q_y, q_u, q_v, color='#2e86c1', scale=25, 
              width=0.005, headwidth=4, headlength=5, zorder=5, label='Robot Heading')

# PLOT 5: Start & Finish Markers
if len(traj_x) > 0:
    # Titik awal (Start) berupa lingkaran hijau emerald tebal
    ax.scatter(traj_x[0], traj_y[0], color='#2ecc71', edgecolors='#27ae60', 
               s=150, marker='o', linewidths=2, label='Start Point', zorder=6)
    # Posisi akhir (Finish) berupa bintang emas/kuning cerah
    ax.scatter(traj_x[-1], traj_y[-1], color='#f1c40f', edgecolors='#f39c12', 
               s=200, marker='*', linewidths=1.5, label='Finish/Current Position', zorder=6)

# FORMATTING & GRAPH STYLING
ax.set_title('Robot Path Tracking & Environment Mapping', 
             fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
ax.set_xlabel('X Position (meters)', fontsize=11, labelpad=8)
ax.set_ylabel('Y Position (meters)', fontsize=11, labelpad=8)
ax.grid(True, linestyle=':', alpha=0.6, color='#b2babb')
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#b2babb')
ax.axis('equal')
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d5dbdb', 
          framealpha=0.9, shadow=False, fontsize=10)

plt.tight_layout()
plt.show()