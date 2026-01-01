"""
Analysis tools for metastability and rare-event simulations.

This module provides functions for:
- Exit time distribution analysis
- Eyring-Kramers law verification
- Transition path analysis
- Uncertainty quantification
- Scaling law verification
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import stats
from scipy.optimize import curve_fit
from dataclasses import dataclass


@dataclass
class ExitTimeStatistics:
    """
    Statistical analysis of exit times.
    
    Attributes
    ----------
    mean : float
        Mean exit time
    std : float
        Standard deviation
    median : float
        Median exit time
    q25, q75 : float
        25th and 75th percentiles
    cv : float
        Coefficient of variation (std/mean)
    n_samples : int
        Number of samples
    """
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    cv: float
    n_samples: int
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for mean using t-distribution.
        
        Parameters
        ----------
        confidence : float
            Confidence level (default 0.95)
            
        Returns
        -------
        tuple of float
            (lower_bound, upper_bound)
        """
        if self.n_samples < 2:
            return (self.mean, self.mean)
        
        # Standard error of mean
        sem = self.std / np.sqrt(self.n_samples)
        
        # t-distribution critical value
        alpha = 1.0 - confidence
        df = self.n_samples - 1
        t_crit = stats.t.ppf(1.0 - alpha/2.0, df)
        
        margin = t_crit * sem
        return (self.mean - margin, self.mean + margin)


def analyze_exit_times(exit_times: np.ndarray, 
                       weights: Optional[np.ndarray] = None) -> ExitTimeStatistics:
    """
    Compute comprehensive statistics for exit times.
    
    Parameters
    ----------
    exit_times : np.ndarray
        Array of exit times
    weights : np.ndarray, optional
        Importance weights for weighted statistics
        
    Returns
    -------
    ExitTimeStatistics
        Statistical summary
    """
    if len(exit_times) == 0:
        return ExitTimeStatistics(
            mean=np.nan, std=np.nan, median=np.nan,
            q25=np.nan, q75=np.nan, cv=np.nan, n_samples=0
        )
    
    if weights is not None:
        # Weighted statistics
        weights = np.array(weights)
        weights /= weights.sum()
        
        mean = np.average(exit_times, weights=weights)
        variance = np.average((exit_times - mean)**2, weights=weights)
        std = np.sqrt(variance)
        
        # Weighted percentiles (approximate)
        sorted_indices = np.argsort(exit_times)
        sorted_times = exit_times[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumsum = np.cumsum(sorted_weights)
        
        median = sorted_times[np.searchsorted(cumsum, 0.5)]
        q25 = sorted_times[np.searchsorted(cumsum, 0.25)]
        q75 = sorted_times[np.searchsorted(cumsum, 0.75)]
    else:
        # Unweighted statistics
        mean = np.mean(exit_times)
        std = np.std(exit_times, ddof=1)
        median = np.median(exit_times)
        q25 = np.percentile(exit_times, 25)
        q75 = np.percentile(exit_times, 75)
    
    cv = std / mean if mean > 0 else np.inf
    
    return ExitTimeStatistics(
        mean=mean, std=std, median=median,
        q25=q25, q75=q75, cv=cv, n_samples=len(exit_times)
    )


def fit_exponential_distribution(exit_times: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit exponential distribution to exit times.
    
    For a Poisson process, exit times follow exponential distribution:
        f(t) = λ exp(-λt)
    
    with mean exit time τ = 1/λ.
    
    Parameters
    ----------
    exit_times : np.ndarray
        Exit time samples
        
    Returns
    -------
    rate : float
        Fitted rate parameter λ
    mean_exit_time : float
        Mean exit time τ = 1/λ
    ks_statistic : float
        Kolmogorov-Smirnov test statistic
    """
    if len(exit_times) < 2:
        return np.nan, np.nan, np.nan
    
    # MLE for exponential: λ = 1/mean
    mean_time = np.mean(exit_times)
    rate = 1.0 / mean_time
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(exit_times, 'expon', args=(0, mean_time))
    
    return rate, mean_time, ks_stat


def verify_eyring_kramers_law(epsilon_values: np.ndarray,
                              mean_exit_times: np.ndarray,
                              barrier_height: float,
                              return_fit: bool = False) -> Dict:
    """
    Verify Eyring-Kramers law for exit times.
    
    The Eyring-Kramers law states:
        τ(ε) ≈ C · exp(ΔV/ε)
    
    where:
    - τ(ε) is mean exit time
    - ΔV is barrier height
    - C is a prefactor depending on local curvatures
    - ε is noise level (temperature)
    
    Taking logarithms:
        log(τ) ≈ log(C) + ΔV/ε
    
    We fit this linear relationship in log(τ) vs 1/ε.
    
    Parameters
    ----------
    epsilon_values : np.ndarray
        Noise levels
    mean_exit_times : np.ndarray
        Corresponding mean exit times
    barrier_height : float
        Theoretical barrier height ΔV
    return_fit : bool
        Whether to return full fit results
        
    Returns
    -------
    dict
        Analysis results containing:
        - fitted_barrier: Fitted barrier height from slope
        - prefactor: Fitted prefactor C
        - r_squared: R² goodness of fit
        - relative_error: |(fitted_barrier - barrier_height) / barrier_height|
    """
    # Remove any invalid data points
    valid = (epsilon_values > 0) & (mean_exit_times > 0) & np.isfinite(mean_exit_times)
    eps = epsilon_values[valid]
    tau = mean_exit_times[valid]
    
    if len(eps) < 2:
        return {'fitted_barrier': np.nan, 'prefactor': np.nan, 
                'r_squared': np.nan, 'relative_error': np.nan}
    
    # Linear regression: log(τ) = a + b/ε
    x = 1.0 / eps
    y = np.log(tau)
    
    # Fit: y = a + b*x
    coeffs = np.polyfit(x, y, deg=1)
    b, a = coeffs  # slope, intercept
    
    fitted_barrier = b
    prefactor = np.exp(a)
    
    # R² statistic
    y_pred = a + b * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Relative error in barrier height
    relative_error = np.abs(fitted_barrier - barrier_height) / barrier_height
    
    results = {
        'fitted_barrier': fitted_barrier,
        'prefactor': prefactor,
        'r_squared': r_squared,
        'relative_error': relative_error,
        'epsilon_values': eps,
        'mean_exit_times': tau
    }
    
    if return_fit:
        results['fit_x'] = x
        results['fit_y'] = y
        results['fit_y_pred'] = y_pred
    
    return results


def analyze_transition_paths(trajectories: List[np.ndarray],
                            x_start: np.ndarray,
                            x_end: np.ndarray) -> Dict:
    """
    Analyze geometry of transition paths.
    
    Computes:
    - Path lengths
    - Path straightness (ratio of Euclidean distance to path length)
    - Tube radius (spread around mean path)
    - Committor values along paths
    
    Parameters
    ----------
    trajectories : list of np.ndarray
        List of transition path trajectories, each shape (n_steps, dim)
    x_start : np.ndarray
        Starting point
    x_end : np.ndarray
        Ending point
        
    Returns
    -------
    dict
        Analysis results
    """
    if len(trajectories) == 0:
        return {}
    
    path_lengths = []
    straightness = []
    
    euclidean_distance = np.linalg.norm(x_end - x_start)
    
    for traj in trajectories:
        # Compute path length
        diffs = np.diff(traj, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(lengths)
        path_lengths.append(total_length)
        
        # Straightness: ratio of Euclidean distance to path length
        if total_length > 0:
            straightness.append(euclidean_distance / total_length)
        else:
            straightness.append(0.0)
    
    # Compute mean path
    # Interpolate all paths to same number of points
    n_interp = 100
    interpolated_paths = []
    
    for traj in trajectories:
        if len(traj) < 2:
            continue
        
        # Parameter along path
        t_orig = np.linspace(0, 1, len(traj))
        t_interp = np.linspace(0, 1, n_interp)
        
        # Interpolate each dimension
        interp_traj = np.zeros((n_interp, traj.shape[1]))
        for d in range(traj.shape[1]):
            interp_traj[:, d] = np.interp(t_interp, t_orig, traj[:, d])
        
        interpolated_paths.append(interp_traj)
    
    if len(interpolated_paths) > 0:
        # Mean path
        mean_path = np.mean(interpolated_paths, axis=0)
        
        # Tube radius: RMS distance from mean path
        deviations = []
        for traj in interpolated_paths:
            dev = np.linalg.norm(traj - mean_path, axis=1)
            deviations.extend(dev)
        
        tube_radius = np.sqrt(np.mean(np.array(deviations)**2))
    else:
        mean_path = None
        tube_radius = np.nan
    
    return {
        'path_lengths': np.array(path_lengths),
        'mean_path_length': np.mean(path_lengths),
        'std_path_length': np.std(path_lengths),
        'straightness': np.array(straightness),
        'mean_straightness': np.mean(straightness),
        'tube_radius': tube_radius,
        'mean_path': mean_path,
        'n_paths': len(trajectories)
    }


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size for importance sampling.
    
    ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²
    
    ESS measures the effective number of independent samples.
    For uniform weights, ESS = n. For highly variable weights, ESS << n.
    
    Parameters
    ----------
    weights : np.ndarray
        Importance weights
        
    Returns
    -------
    float
        Effective sample size
    """
    if len(weights) == 0:
        return 0.0
    
    weights = np.array(weights)
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)
    
    if sum_w2 == 0:
        return 0.0
    
    return sum_w**2 / sum_w2


def bootstrap_confidence_interval(data: np.ndarray,
                                  statistic: callable,
                                  n_bootstrap: int = 1000,
                                  confidence: float = 0.95,
                                  seed: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : np.ndarray
        Data samples
    statistic : callable
        Function computing statistic from data
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    seed : int, optional
        Random seed
        
    Returns
    -------
    estimate : float
        Point estimate
    lower : float
        Lower confidence bound
    upper : float
        Upper confidence bound
    """
    rng = np.random.default_rng(seed)
    
    # Point estimate
    estimate = statistic(data)
    
    # Bootstrap resampling
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.integers(0, n, size=n)
        bootstrap_sample = data[indices]
        bootstrap_stats.append(statistic(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile confidence interval
    alpha = 1.0 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return estimate, lower, upper


def compare_algorithms(results_dict: Dict[str, 'RareEventResult']) -> Dict:
    """
    Compare performance of different rare-event algorithms.
    
    Metrics:
    - Mean exit time estimates
    - Variance / coefficient of variation
    - Computational cost (number of SDE steps)
    - Efficiency: variance × cost
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping algorithm names to RareEventResult objects
        
    Returns
    -------
    dict
        Comparison metrics
    """
    comparison = {}
    
    for name, result in results_dict.items():
        if len(result.exit_times) == 0:
            continue
        
        stats = analyze_exit_times(np.array(result.exit_times),
                                   np.array(result.weights) if result.weights else None)
        
        # Efficiency metric: variance × computational cost
        # Lower is better
        efficiency = stats.std**2 * result.computational_cost
        
        comparison[name] = {
            'mean_exit_time': stats.mean,
            'std_exit_time': stats.std,
            'cv': stats.cv,
            'n_exits': result.n_exits,
            'computational_cost': result.computational_cost,
            'cost_per_exit': result.computational_cost / result.n_exits if result.n_exits > 0 else np.inf,
            'efficiency': efficiency
        }
    
    return comparison


def test_convergence_rate(dt_values: np.ndarray,
                         errors: np.ndarray,
                         expected_order: float) -> Dict:
    """
    Test numerical convergence rate of SDE solver.
    
    For a method with convergence order p:
        error ≈ C · dt^p
    
    Taking logarithms:
        log(error) ≈ log(C) + p · log(dt)
    
    Parameters
    ----------
    dt_values : np.ndarray
        Time step sizes
    errors : np.ndarray
        Corresponding errors
    expected_order : float
        Expected convergence order
        
    Returns
    -------
    dict
        Convergence analysis results
    """
    # Remove invalid points
    valid = (dt_values > 0) & (errors > 0) & np.isfinite(errors)
    dt = dt_values[valid]
    err = errors[valid]
    
    if len(dt) < 2:
        return {'fitted_order': np.nan, 'r_squared': np.nan}
    
    # Log-log regression
    x = np.log(dt)
    y = np.log(err)
    
    coeffs = np.polyfit(x, y, deg=1)
    fitted_order, intercept = coeffs
    
    # R² statistic
    y_pred = intercept + fitted_order * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return {
        'fitted_order': fitted_order,
        'expected_order': expected_order,
        'r_squared': r_squared,
        'relative_error': np.abs(fitted_order - expected_order) / expected_order
    }


def analyze_dimension_scaling(dimensions: np.ndarray,
                             mean_exit_times: np.ndarray,
                             epsilon: float,
                             barrier_per_dim: float = 1.0) -> Dict:
    """
    Analyze how exit time scales with dimension.
    
    For coupled high-dimensional systems, the effective barrier height
    often scales with dimension:
        ΔV_eff ≈ d · ΔV₁
    
    Leading to:
        τ(d) ≈ C · exp(d · ΔV₁ / ε)
    
    This demonstrates the "curse of dimensionality" for rare events.
    
    Parameters
    ----------
    dimensions : np.ndarray
        Dimension values
    mean_exit_times : np.ndarray
        Mean exit times
    epsilon : float
        Noise level
    barrier_per_dim : float
        Barrier height per dimension
        
    Returns
    -------
    dict
        Scaling analysis results
    """
    valid = (mean_exit_times > 0) & np.isfinite(mean_exit_times)
    d = dimensions[valid]
    tau = mean_exit_times[valid]
    
    if len(d) < 2:
        return {'fitted_barrier_per_dim': np.nan, 'r_squared': np.nan}
    
    # Fit: log(τ) = a + b·d where b = ΔV₁/ε
    x = d
    y = np.log(tau)
    
    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = coeffs
    
    fitted_barrier_per_dim = slope * epsilon
    
    # R² statistic
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return {
        'fitted_barrier_per_dim': fitted_barrier_per_dim,
        'expected_barrier_per_dim': barrier_per_dim,
        'r_squared': r_squared,
        'relative_error': np.abs(fitted_barrier_per_dim - barrier_per_dim) / barrier_per_dim
    }
