a

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

from src.potentials import SymmetricDoubleWell
from src.sde_solvers import EulerMaruyama

print("="*80)
print("  3D RARE TRANSITION ANIMATION")
print("="*80)
print()

# Setup
print("Step 1: Creating 2D symmetric double-well potential...")
potential = SymmetricDoubleWell(dim=2, omega=2.0)
epsilon = 0.3  # Higher noise for faster transition
dt = 0.01

print(f"  Noise level: ε = {epsilon}")
print(f"  Time step: dt = {dt}")
print()

# Generate transition trajectory
print("Step 2: Simulating transition trajectory...")
solver = EulerMaruyama(potential.grad_V, dim=2, epsilon=epsilon)
x0 = np.array([-1.0, 0.0])

def exit_condition(x):
    return x[0] > 0.8

traj, exited = solver.simulate_until_exit(x0, exit_condition, dt, max_steps=20000, seed=42)

if exited:
    print(f"  [OK] Transition observed!")
    print(f"  Exit time: {traj.times[-1]:.2f}")
    print(f"  Number of steps: {len(traj)}")
else:
    print(f"  Using partial trajectory ({len(traj)} steps)")

print()

# Create 3D surface
print("Step 3: Creating 3D potential surface...")
n_points = 50
x = np.linspace(-2, 2, n_points)
y = np.linspace(-1.5, 1.5, n_points)
X, Y = np.meshgrid(x, y)

V = np.zeros_like(X)
for i in range(n_points):
    for j in range(n_points):
        V[i, j] = potential.V(np.array([X[i, j], Y[i, j]]))

print(f"  Grid resolution: {n_points}×{n_points}")
print()

# Create static 3D figure
print("Step 4: Creating static 3D visualization...")
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(X, Y, V, cmap='viridis', alpha=0.7, 
                       edgecolor='none', antialiased=True)

# Plot trajectory on surface
traj_positions = traj.positions[:, :2]
traj_V = np.array([potential.V(pos) for pos in traj_positions])

# Full trajectory path
ax.plot(traj_positions[:, 0], traj_positions[:, 1], traj_V, 
       'r-', linewidth=2, alpha=0.6, label='Transition path')

# Start and end points
ax.scatter([traj_positions[0, 0]], [traj_positions[0, 1]], [traj_V[0]], 
          color='green', s=200, marker='o', label='Start', zorder=10)
ax.scatter([traj_positions[-1, 0]], [traj_positions[-1, 1]], [traj_V[-1]], 
          color='red', s=200, marker='^', label='End', zorder=10)

# Labels and title
ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_zlabel('V(x₁, x₂)', fontsize=12)
ax.set_title('3D Rare Transition: Particle Escaping Metastable Well', fontsize=14)
ax.legend(fontsize=11)

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, label='Potential Energy')

plt.tight_layout()

# Save static figure
static_path = '../plots/3d_transition_static.png'
plt.savefig(static_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Static 3D figure saved: {static_path}")
print()

# Create animated GIF
print("Step 5: Creating animated GIF (this may take 30-60 seconds)...")

# Subsample trajectory for animation (every 10th point)
subsample = 10
anim_positions = traj_positions[::subsample]
anim_V = traj_V[::subsample]
n_frames = len(anim_positions)

print(f"  Animation frames: {n_frames}")

# Create animation figure
fig_anim = plt.figure(figsize=(12, 9))
ax_anim = fig_anim.add_subplot(111, projection='3d')

# Plot surface
surf_anim = ax_anim.plot_surface(X, Y, V, cmap='viridis', alpha=0.6, 
                                 edgecolor='none', antialiased=True)

# Initialize trajectory line and particle
line, = ax_anim.plot([], [], [], 'r-', linewidth=2, alpha=0.6)
particle, = ax_anim.plot([], [], [], 'ro', markersize=12, markeredgecolor='white', 
                        markeredgewidth=2)

# Labels
ax_anim.set_xlabel('x₁', fontsize=12)
ax_anim.set_ylabel('x₂', fontsize=12)
ax_anim.set_zlabel('V(x₁, x₂)', fontsize=12)
ax_anim.set_title('Animated Rare Transition', fontsize=14)
ax_anim.view_init(elev=25, azim=45)

# Set limits
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(-1.5, 1.5)
ax_anim.set_zlim(V.min(), V.max())

# Animation update function
def update(frame):
    # Update trajectory line
    line.set_data(anim_positions[:frame+1, 0], anim_positions[:frame+1, 1])
    line.set_3d_properties(anim_V[:frame+1])
    
    # Update particle position
    particle.set_data([anim_positions[frame, 0]], [anim_positions[frame, 1]])
    particle.set_3d_properties([anim_V[frame]])
    
    # Update title with time
    time = frame * subsample * dt
    ax_anim.set_title(f'Rare Transition (t = {time:.2f})', fontsize=14)
    
    return line, particle

# Create animation
print("  Creating animation...")
anim = FuncAnimation(fig_anim, update, frames=n_frames, 
                    interval=50, blit=False, repeat=True)

# Save as GIF
gif_path = '../plots/3d_transition_animation.gif'
writer = PillowWriter(fps=20)
anim.save(gif_path, writer=writer)

print(f"  [OK] Animated GIF saved: {gif_path}")
print()

# Create multiple viewing angles
print("Step 6: Creating multi-angle views...")

fig_multi, axes = plt.subplots(2, 2, figsize=(16, 14), 
                               subplot_kw={'projection': '3d'})
axes = axes.flatten()

angles = [(20, 30), (20, 120), (60, 45), (10, 60)]
titles = ['View 1: Front-Left', 'View 2: Front-Right', 
          'View 3: Top', 'View 4: Side']

for idx, (ax_view, (elev, azim), title) in enumerate(zip(axes, angles, titles)):
    # Plot surface
    ax_view.plot_surface(X, Y, V, cmap='viridis', alpha=0.6, 
                        edgecolor='none', antialiased=True)
    
    # Plot trajectory
    ax_view.plot(traj_positions[:, 0], traj_positions[:, 1], traj_V, 
                'r-', linewidth=2, alpha=0.7)
    
    # Start and end
    ax_view.scatter([traj_positions[0, 0]], [traj_positions[0, 1]], [traj_V[0]], 
                   color='green', s=100, marker='o', zorder=10)
    ax_view.scatter([traj_positions[-1, 0]], [traj_positions[-1, 1]], [traj_V[-1]], 
                   color='red', s=100, marker='^', zorder=10)
    
    # Labels
    ax_view.set_xlabel('x₁', fontsize=10)
    ax_view.set_ylabel('x₂', fontsize=10)
    ax_view.set_zlabel('V(x)', fontsize=10)
    ax_view.set_title(title, fontsize=12)
    ax_view.view_init(elev=elev, azim=azim)

plt.tight_layout()

multi_path = '../plots/3d_transition_multiview.png'
plt.savefig(multi_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Multi-angle view saved: {multi_path}")
print()

# Summary
print("="*80)
print("  ANIMATION COMPLETE!")
print("="*80)
print()
print("Generated files:")
print(f"  1. {static_path}")
print(f"     - High-quality static 3D visualization")
print()
print(f"  2. {gif_path}")
print(f"     - Animated GIF showing particle motion")
print()
print(f"  3. {multi_path}")
print(f"     - Multiple viewing angles")
print()
print("Trajectory statistics:")
print(f"  - Total time: {traj.times[-1]:.2f}")
print(f"  - Path length: {traj.trajectory_length():.2f}")
print(f"  - Number of steps: {len(traj)}")
print(f"  - Animation frames: {n_frames}")
print()
print("To view:")
print("  - Open the PNG files in any image viewer")
print("  - Open the GIF in a browser or image viewer to see animation")
print()
print("="*80)

plt.close('all')
