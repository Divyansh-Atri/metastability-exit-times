
import sys
import os
import random
import time
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

from src.potentials import SymmetricDoubleWell
from src.sde_solvers import EulerMaruyama

# Suppress font warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)



# =============================================================================
# PROFESSIONAL AESTHETICS ENGINE
# =============================================================================
plt.style.use('dark_background')

# Professional Scientific Colormap
colors = [(0.05, 0.05, 0.1), (0.1, 0.3, 0.5), (0.0, 0.6, 0.7), (0.9, 0.7, 0.1), (1.0, 0.9, 0.8)]
n_bins = 256
cmap_name = 'scientific_magma'
pro_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

print("="*80)
print("  [INIT] INITIALIZING ENGINE: HIGH-FIDELITY STOCHASTIC VISUALIZER (PRO)")
print(f"  [INFO] Renderer: FFMpeg | Aesthetics: {cmap_name.upper()}")
print("="*80)

# --- Configuration ---
N_PARTICLES = 30
DIM = 2
EPSILON_HIGH = 0.6
DT = 0.01
MAX_STEPS = 600
K_FRAMES = 300 # Faster rendering, 10 seconds at 30fps
FPS = 30
OMEGA = 2.0

# --- Setup System ---
potential = SymmetricDoubleWell(dim=DIM, omega=OMEGA)
solver = EulerMaruyama(potential.grad_V, dim=DIM, epsilon=EPSILON_HIGH)

# --- Initialize Swarm ---
print(f"  [CORE] Spawning {N_PARTICLES} stochastic agents in metastable well A...")
x0_center = np.array([-1.0, 0.0])
swarm_pos = x0_center + np.random.normal(scale=0.15, size=(N_PARTICLES, DIM))

# --- Simulation Phase ---
print("  [COMPUTE] Integrating SDEs (Euler-Maruyama Scheme)...")
all_trajectories = np.zeros((K_FRAMES + 1, N_PARTICLES, DIM))
all_trajectories[0] = swarm_pos

current_pos = swarm_pos.copy()
rng = np.random.default_rng(seed=42)

for step in range(K_FRAMES):
    for i in range(N_PARTICLES):
        current_pos[i] = solver.step(current_pos[i], DT, rng)
    
    all_trajectories[step + 1] = current_pos
    if step % 50 == 0:
        print(f"    Simulating step {step}/{K_FRAMES}...")

print("  [SUCCESS] Simulation completed. Rendering high-res video...")

# =============================================================================
# RENDERING PIPELINE
# =============================================================================
# 1920x1080 resolution at 120 DPI
fig = plt.figure(figsize=(16, 9), dpi=120) 
ax = fig.add_subplot(111, projection='3d')

# Deep charcoal background
bg_color = '#0b0c10'
ax.set_facecolor(bg_color)
fig.patch.set_facecolor(bg_color)

# --- Generate Potential Surface ---
n_grid = 100
x_range = np.linspace(-2.5, 2.5, n_grid)
y_range = np.linspace(-2.0, 2.0, n_grid)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
for i in range(n_grid):
    for j in range(n_grid):
        Z[i, j] = potential.V(np.array([X[i, j], Y[i, j]]))

# Render limit
Z[Z > 3.5] = 3.5

# High-quality surface plot
surf = ax.plot_surface(X, Y, Z, cmap=pro_cmap, alpha=0.15, 
                       linewidth=0, antialiased=True, rstride=1, cstride=1, shade=True)

# Subtle wireframe
wire = ax.plot_wireframe(X, Y, Z, color='#1f2833', alpha=0.2, linewidth=0.5, rstride=5, cstride=5)

# --- HUD Elements ---
# Use default fonts to avoid warnings
title_font = {'weight': 'bold', 'size': 18}
meta_font = {'size': 10, 'color': '#66fcf1'}

title_text = ax.text2D(0.02, 0.95, "METASTABILITY // RARE EVENT DYNAMICS", transform=ax.transAxes, 
                       color='#c5c6c7', **title_font)

subtitle_text = ax.text2D(0.02, 0.92, "Stochastic Differential Equation Solver | Euler-Maruyama Method", 
                          transform=ax.transAxes, color='#45a29e', fontsize=12)

# Stats Block
stats_template = (
    "TIME:        {time:6.2f}\n"
    "TRANSITIONS: {trans}/{total}\n"
    "MEAN ENERGY: {energy:.4f}"
)
stats_text = ax.text2D(0.02, 0.82, "", transform=ax.transAxes, 
                       fontsize=10, color='#c5c6c7',
                       bbox=dict(facecolor='#1f2833', alpha=0.5, edgecolor='none', pad=5))

# Status Indicator (Subtle, Top Right)
status_indicator = ax.text2D(0.95, 0.95, "System stable", transform=ax.transAxes,
                             fontsize=12, weight='bold', color='#45a29e', ha='right',
                             bbox=dict(facecolor='#1f2833', alpha=0.8, edgecolor='#45a29e', boxstyle='round,pad=0.5'))



# --- Particles Setting ---
particles_scatter = ax.scatter([], [], [], s=120, c='white', edgecolors='white', alpha=1.0, depthshade=False)
particles_glow = ax.scatter([], [], [], s=350, c='#66fcf1', alpha=0.2, depthshade=False, linewidth=0)

# Trails
trails = []
for i in range(N_PARTICLES):
    t, = ax.plot([], [], [], '-', linewidth=1.5, alpha=0.4)
    trails.append(t)

# Limits & Camera
# Zoomed-in framing for animation (closer camera and tighter axis limits)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(0, 2.5)
# Move the 3D camera closer for a stronger zoom effect (smaller = closer)
try:
    ax.dist = 7
except Exception:
    pass
ax.set_axis_off()

def update(frame):
    # 1. Smooth Orbit Camera
    angle = 30 + (frame / K_FRAMES) * 60  # Rotate 60 degrees over the video
    elev = 40
    ax.view_init(elev=elev, azim=angle)
    
    # 2. Data Retrieval
    current_x = all_trajectories[frame, :, 0]
    current_y = all_trajectories[frame, :, 1]
    
    energies = []
    for p in all_trajectories[frame]:
        energies.append(potential.V(p))
    energies = np.array(energies)
    current_z = energies + 0.1
    
    # 3. Particle Updates
    particles_scatter._offsets3d = (current_x, current_y, current_z)
    
    # Color logic
    colors_arr = np.empty((N_PARTICLES, 4))
    is_transitioned = current_x > 0
    
    # Base color: Cyan
    colors_arr[:] = [0.4, 0.99, 0.94, 1.0] # #66fcf1
    # Transitioned: Gold
    colors_arr[is_transitioned] = [1.0, 0.65, 0.0, 1.0] # #ffa700
    
    particles_scatter.set_color(colors_arr)
    
    # Glow match
    particles_glow._offsets3d = (current_x, current_y, current_z)
    glow_colors = colors_arr.copy()
    glow_colors[:, 3] = 0.3 # Reduce alpha for glow
    particles_glow.set_color(glow_colors)
    
    # 4. Trail Logic
    tail_len = 50
    start = max(0, frame - tail_len)
    
    transition_count = np.sum(is_transitioned)
    
    for i in range(N_PARTICLES):
        hx = all_trajectories[start:frame+1, i, 0]
        hy = all_trajectories[start:frame+1, i, 1]
        hz = np.array([potential.V(p) for p in all_trajectories[start:frame+1, i]]) + 0.05
        
        trails[i].set_data(hx, hy)
        trails[i].set_3d_properties(hz)
        
        # Trail color matches particle
        if current_x[i] > 0:
            trails[i].set_color('#ffa700') # Gold
            trails[i].set_alpha(0.6)
        else:
            trails[i].set_color('#45a29e') # Muted Teal
            trails[i].set_alpha(0.3)

    # 5. HUD Updates
    stats_text.set_text(stats_template.format(
        time=frame*DT,
        trans=transition_count,
        total=N_PARTICLES,
        energy=np.mean(energies)
    ))
    
    # if transition_count > 0:
    #     status_indicator.set_text("Transition detected")
    #     status_indicator.set_color('#ffa700')
    #     status_indicator.set_bbox(dict(facecolor='#1f2833', alpha=0.9, edgecolor='#ffa700', boxstyle='round,pad=0.5'))
    # else:
    #     status_indicator.set_text("System stable")
    #     status_indicator.set_color('#45a29e')
    #     status_indicator.set_bbox(dict(facecolor='#1f2833', alpha=0.8, edgecolor='#45a29e', boxstyle='round,pad=0.5'))

    # return [particles_scatter, particles_glow, stats_text, status_indicator] + trails

print(f"  [RENDER] Rendering {K_FRAMES} frames @ {FPS} FPS to GIF...")
anim = FuncAnimation(fig, update, frames=K_FRAMES, interval=1000/FPS, blit=False)

# Save
output_dir = os.path.join(os.path.dirname(__file__), '../plots')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'custom_swarm_study_mode.gif')

# Use Pillow for GIF compatibility
print(f"  [RENDER] Rendering {K_FRAMES} frames @ {FPS} FPS to GIF...")
anim.save(output_path, writer='pillow', fps=FPS)

print(f"  [COMPLETE] Animation saved to: {output_path}")
print("="*80)
