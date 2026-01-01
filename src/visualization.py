"""
Publication-quality visualization tools for metastability studies.

This module provides functions to create professional figures for:
- Potential energy landscapes
- Exit time distributions
- Scaling law verification
- Transition path visualization
- High-dimensional projections
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple, Callable
import matplotlib.patches as mpatches


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})


def plot_potential_1d(potential, x_range: Tuple[float, float],
                     n_points: int = 1000,
                     show_minima: bool = True,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot 1D potential energy landscape.
    
    Parameters
    ----------
    potential : Potential
        Potential object (must be 1D)
    x_range : tuple of float
        (x_min, x_max) range for plotting
    n_points : int
        Number of evaluation points
    show_minima : bool
        Whether to mark local minima
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    V = np.array([potential.V(np.array([xi])) for xi in x])
    
    ax.plot(x, V, 'b-', linewidth=2, label='V(x)')
    
    if show_minima:
        minima = potential.find_minima()
        for minimum in minima:
            if x_range[0] <= minimum[0] <= x_range[1]:
                V_min = potential.V(minimum)
                ax.plot(minimum[0], V_min, 'ro', markersize=10, 
                       label='Local minimum', zorder=5)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Potential energy V(x)')
    ax.set_title('Potential Energy Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_potential_2d(potential, x_range: Tuple[float, float],
                     y_range: Tuple[float, float],
                     n_points: int = 200,
                     show_minima: bool = True,
                     contour_levels: int = 30,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot 2D potential energy landscape as contour plot.
    
    Parameters
    ----------
    potential : Potential
        Potential object (must be 2D)
    x_range, y_range : tuple of float
        Ranges for plotting
    n_points : int
        Grid resolution
    show_minima : bool
        Whether to mark local minima
    contour_levels : int
        Number of contour levels
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    V = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            V[i, j] = potential.V(np.array([X[i, j], Y[i, j]]))
    
    # Contour plot
    contour = ax.contourf(X, Y, V, levels=contour_levels, cmap='viridis')
    ax.contour(X, Y, V, levels=contour_levels, colors='k', 
              alpha=0.3, linewidths=0.5)
    
    plt.colorbar(contour, ax=ax, label='V(x, y)')
    
    if show_minima:
        minima = potential.find_minima()
        for minimum in minima:
            if (x_range[0] <= minimum[0] <= x_range[1] and
                y_range[0] <= minimum[1] <= y_range[1]):
                ax.plot(minimum[0], minimum[1], 'r*', markersize=15,
                       markeredgecolor='white', markeredgewidth=1.5,
                       label='Local minimum', zorder=5)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Potential Energy Landscape')
    ax.set_aspect('equal')
    
    return ax


def plot_potential_3d(potential, x_range: Tuple[float, float],
                     y_range: Tuple[float, float],
                     n_points: int = 100,
                     elevation: float = 30,
                     azimuth: float = 45,
                     ax: Optional[Axes3D] = None) -> Axes3D:
    """
    Plot 2D potential as 3D surface.
    
    Parameters
    ----------
    potential : Potential
        Potential object (must be 2D)
    x_range, y_range : tuple of float
        Ranges for plotting
    n_points : int
        Grid resolution
    elevation, azimuth : float
        Viewing angles
    ax : Axes3D, optional
        3D axes to plot on
        
    Returns
    -------
    Axes3D
        3D axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    V = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            V[i, j] = potential.V(np.array([X[i, j], Y[i, j]]))
    
    surf = ax.plot_surface(X, Y, V, cmap='viridis', alpha=0.9,
                          edgecolor='none', antialiased=True)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('V(x₁, x₂)')
    ax.set_title('Potential Energy Surface')
    ax.view_init(elev=elevation, azim=azimuth)
    
    plt.colorbar(surf, ax=ax, shrink=0.5, label='Energy')
    
    return ax


def plot_trajectory_2d(trajectory: np.ndarray,
                      potential=None,
                      x_range: Optional[Tuple[float, float]] = None,
                      y_range: Optional[Tuple[float, float]] = None,
                      show_potential: bool = True,
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot 2D trajectory overlaid on potential landscape.
    
    Parameters
    ----------
    trajectory : np.ndarray, shape (n_steps, 2)
        Trajectory positions
    potential : Potential, optional
        Potential object for background
    x_range, y_range : tuple of float, optional
        Plot ranges (auto-determined if None)
    show_potential : bool
        Whether to show potential contours
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    
    # Auto-determine ranges
    if x_range is None:
        margin = 0.2
        x_span = trajectory[:, 0].max() - trajectory[:, 0].min()
        x_range = (trajectory[:, 0].min() - margin * x_span,
                  trajectory[:, 0].max() + margin * x_span)
    
    if y_range is None:
        margin = 0.2
        y_span = trajectory[:, 1].max() - trajectory[:, 1].min()
        y_range = (trajectory[:, 1].min() - margin * y_span,
                  trajectory[:, 1].max() + margin * y_span)
    
    # Plot potential background
    if show_potential and potential is not None:
        plot_potential_2d(potential, x_range, y_range, 
                         show_minima=True, ax=ax)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', 
           linewidth=2, alpha=0.7, label='Trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', 
           markersize=10, label='Start', zorder=10)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', 
           markersize=10, label='End', zorder=10)
    
    ax.legend()
    ax.set_title('SDE Trajectory')
    
    return ax


def plot_exit_time_distribution(exit_times: np.ndarray,
                                weights: Optional[np.ndarray] = None,
                                n_bins: int = 30,
                                fit_exponential: bool = True,
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot exit time distribution with optional exponential fit.
    
    Parameters
    ----------
    exit_times : np.ndarray
        Exit time samples
    weights : np.ndarray, optional
        Importance weights
    n_bins : int
        Number of histogram bins
    fit_exponential : bool
        Whether to overlay exponential fit
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histogram
    if weights is not None:
        counts, bins, patches = ax.hist(exit_times, bins=n_bins, 
                                       weights=weights, density=True,
                                       alpha=0.7, color='steelblue',
                                       edgecolor='black', label='Data')
    else:
        counts, bins, patches = ax.hist(exit_times, bins=n_bins, 
                                       density=True, alpha=0.7,
                                       color='steelblue', edgecolor='black',
                                       label='Data')
    
    # Exponential fit
    if fit_exponential and len(exit_times) > 1:
        mean_time = np.average(exit_times, weights=weights) if weights is not None else np.mean(exit_times)
        rate = 1.0 / mean_time
        
        t = np.linspace(0, exit_times.max(), 200)
        exponential = rate * np.exp(-rate * t)
        
        ax.plot(t, exponential, 'r-', linewidth=2,
               label=f'Exponential fit (τ={mean_time:.2f})')
    
    ax.set_xlabel('Exit time')
    ax.set_ylabel('Probability density')
    ax.set_title('Exit Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_eyring_kramers_verification(epsilon_values: np.ndarray,
                                     mean_exit_times: np.ndarray,
                                     barrier_height: float,
                                     fitted_barrier: Optional[float] = None,
                                     ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot Eyring-Kramers law verification.
    
    Plots log(τ) vs 1/ε with theoretical and fitted slopes.
    
    Parameters
    ----------
    epsilon_values : np.ndarray
        Noise levels
    mean_exit_times : np.ndarray
        Mean exit times
    barrier_height : float
        Theoretical barrier height
    fitted_barrier : float, optional
        Fitted barrier height from regression
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter valid data
    valid = (epsilon_values > 0) & (mean_exit_times > 0) & np.isfinite(mean_exit_times)
    eps = epsilon_values[valid]
    tau = mean_exit_times[valid]
    
    # Plot data
    ax.semilogy(1.0 / eps, tau, 'o', markersize=8, 
               color='steelblue', label='Simulation data')
    
    # Theoretical line
    inv_eps_range = np.linspace(1.0 / eps.max(), 1.0 / eps.min(), 100)
    # Use first data point to determine prefactor
    prefactor = tau[0] / np.exp(barrier_height / eps[0])
    theoretical = prefactor * np.exp(barrier_height * inv_eps_range)
    
    ax.semilogy(inv_eps_range, theoretical, 'k--', linewidth=2,
               label=f'Theory: exp(ΔV/ε), ΔV={barrier_height:.2f}')
    
    # Fitted line
    if fitted_barrier is not None:
        fitted_prefactor = tau[0] / np.exp(fitted_barrier / eps[0])
        fitted = fitted_prefactor * np.exp(fitted_barrier * inv_eps_range)
        ax.semilogy(inv_eps_range, fitted, 'r-', linewidth=2,
                   label=f'Fit: ΔV={fitted_barrier:.2f}')
    
    ax.set_xlabel('1/ε (inverse temperature)')
    ax.set_ylabel('Mean exit time τ')
    ax.set_title('Eyring-Kramers Law Verification')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    return ax


def plot_dimension_scaling(dimensions: np.ndarray,
                          mean_exit_times: np.ndarray,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot exit time scaling with dimension.
    
    Parameters
    ----------
    dimensions : np.ndarray
        Dimension values
    mean_exit_times : np.ndarray
        Mean exit times
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    valid = (mean_exit_times > 0) & np.isfinite(mean_exit_times)
    d = dimensions[valid]
    tau = mean_exit_times[valid]
    
    ax.semilogy(d, tau, 'o-', markersize=8, linewidth=2,
               color='steelblue', label='Simulation')
    
    # Exponential fit
    if len(d) >= 2:
        coeffs = np.polyfit(d, np.log(tau), deg=1)
        slope, intercept = coeffs
        d_range = np.linspace(d.min(), d.max(), 100)
        fit = np.exp(intercept + slope * d_range)
        ax.semilogy(d_range, fit, 'r--', linewidth=2,
                   label=f'Exponential fit')
    
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Mean exit time τ')
    ax.set_title('Curse of Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    return ax


def plot_transition_paths(paths: List[np.ndarray],
                         potential=None,
                         x_range: Optional[Tuple[float, float]] = None,
                         y_range: Optional[Tuple[float, float]] = None,
                         show_mean_path: bool = True,
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot multiple transition paths with mean path.
    
    Parameters
    ----------
    paths : list of np.ndarray
        List of 2D transition paths
    potential : Potential, optional
        Background potential
    x_range, y_range : tuple of float, optional
        Plot ranges
    show_mean_path : bool
        Whether to show mean path
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    
    # Auto-determine ranges
    if x_range is None or y_range is None:
        all_points = np.vstack(paths)
        margin = 0.2
        x_span = all_points[:, 0].max() - all_points[:, 0].min()
        y_span = all_points[:, 1].max() - all_points[:, 1].min()
        x_range = (all_points[:, 0].min() - margin * x_span,
                  all_points[:, 0].max() + margin * x_span)
        y_range = (all_points[:, 1].min() - margin * y_span,
                  all_points[:, 1].max() + margin * y_span)
    
    # Plot potential background
    if potential is not None:
        plot_potential_2d(potential, x_range, y_range,
                         show_minima=True, contour_levels=20, ax=ax)
    
    # Plot individual paths
    for i, path in enumerate(paths):
        ax.plot(path[:, 0], path[:, 1], 'r-', alpha=0.3, linewidth=1)
    
    # Plot mean path
    if show_mean_path and len(paths) > 0:
        # Interpolate to common length
        n_interp = 100
        interp_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            t_orig = np.linspace(0, 1, len(path))
            t_interp = np.linspace(0, 1, n_interp)
            interp_x = np.interp(t_interp, t_orig, path[:, 0])
            interp_y = np.interp(t_interp, t_orig, path[:, 1])
            interp_paths.append(np.column_stack([interp_x, interp_y]))
        
        if len(interp_paths) > 0:
            mean_path = np.mean(interp_paths, axis=0)
            ax.plot(mean_path[:, 0], mean_path[:, 1], 'b-', 
                   linewidth=3, label='Mean path', zorder=5)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(f'Transition Paths (n={len(paths)})')
    ax.legend()
    
    return ax


def plot_algorithm_comparison(results_dict: dict,
                             metric: str = 'mean_exit_time',
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Compare performance of different algorithms.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary from compare_algorithms()
    metric : str
        Metric to plot: 'mean_exit_time', 'cv', 'efficiency', etc.
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = list(results_dict.keys())
    values = [results_dict[alg][metric] for alg in algorithms]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    bars = ax.bar(algorithms, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Algorithm Comparison: {metric.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels if many algorithms
    if len(algorithms) > 4:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return ax


def plot_high_dim_projection(trajectory: np.ndarray,
                            dim1: int = 0,
                            dim2: int = 1,
                            color_by_time: bool = True,
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot 2D projection of high-dimensional trajectory.
    
    Parameters
    ----------
    trajectory : np.ndarray, shape (n_steps, dim)
        High-dimensional trajectory
    dim1, dim2 : int
        Dimensions to project onto
    color_by_time : bool
        Whether to color trajectory by time
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    x = trajectory[:, dim1]
    y = trajectory[:, dim2]
    
    if color_by_time:
        # Color by time progression
        time = np.arange(len(trajectory))
        scatter = ax.scatter(x, y, c=time, cmap='viridis', 
                           s=20, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Time step')
    else:
        ax.plot(x, y, 'b-', alpha=0.6, linewidth=1.5)
    
    # Mark start and end
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start', zorder=10)
    ax.plot(x[-1], y[-1], 'r^', markersize=10, label='End', zorder=10)
    
    ax.set_xlabel(f'Dimension {dim1}')
    ax.set_ylabel(f'Dimension {dim2}')
    ax.set_title(f'Trajectory Projection (dims {dim1}, {dim2})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def create_summary_figure(potential, trajectory, exit_times,
                         epsilon: float, barrier_height: float,
                         save_path: Optional[str] = None):
    """
    Create comprehensive summary figure with multiple panels.
    
    Parameters
    ----------
    potential : Potential
        Potential object (2D)
    trajectory : np.ndarray
        Example trajectory
    exit_times : np.ndarray
        Exit time samples
    epsilon : float
        Noise level
    barrier_height : float
        Barrier height
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Panel 1: Potential landscape
    ax1 = plt.subplot(2, 3, 1)
    plot_potential_2d(potential, (-2, 2), (-2, 2), ax=ax1)
    ax1.set_title('(a) Potential Energy Landscape')
    
    # Panel 2: Sample trajectory
    ax2 = plt.subplot(2, 3, 2)
    plot_trajectory_2d(trajectory, potential, (-2, 2), (-2, 2), ax=ax2)
    ax2.set_title('(b) Sample Trajectory')
    
    # Panel 3: Exit time distribution
    ax3 = plt.subplot(2, 3, 3)
    plot_exit_time_distribution(exit_times, ax=ax3)
    ax3.set_title('(c) Exit Time Distribution')
    
    # Panel 4: 3D potential
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    plot_potential_3d(potential, (-2, 2), (-2, 2), ax=ax4)
    ax4.set_title('(d) 3D Potential Surface')
    
    # Panel 5: Trajectory time series
    ax5 = plt.subplot(2, 3, 5)
    times = np.arange(len(trajectory)) * 0.01
    ax5.plot(times, trajectory[:, 0], label='x₁')
    ax5.plot(times, trajectory[:, 1], label='x₂')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Position')
    ax5.set_title('(e) Trajectory Time Series')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    mean_time = np.mean(exit_times)
    std_time = np.std(exit_times)
    
    stats_text = f"""
    Simulation Parameters:
    ━━━━━━━━━━━━━━━━━━━━━━
    Noise level ε: {epsilon:.4f}
    Barrier height ΔV: {barrier_height:.2f}
    
    Exit Time Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━
    Mean: {mean_time:.2f}
    Std: {std_time:.2f}
    CV: {std_time/mean_time:.2f}
    N samples: {len(exit_times)}
    
    Theoretical Prediction:
    ━━━━━━━━━━━━━━━━━━━━━━
    τ ∝ exp(ΔV/ε)
    τ ∝ exp({barrier_height/epsilon:.1f})
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    ax6.set_title('(f) Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig
