"""
Rare-event simulation algorithms for metastable systems.

This module implements advanced techniques to accelerate sampling of rare
transitions between metastable states:

1. Importance Sampling: Bias the dynamics to favor rare events
2. Splitting/Cloning: Replicate successful trajectories
3. Adaptive Multilevel Splitting (AMS): Dynamically adapt levels
4. Weighted Ensemble: Maintain ensemble in bins

All algorithms are implemented from scratch with detailed documentation
of the theoretical foundations and practical considerations.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class RareEventResult:
    """
    Container for rare-event simulation results.
    
    Attributes
    ----------
    exit_times : list of float
        Exit times for successful trajectories
    exit_positions : list of np.ndarray
        Exit positions
    n_trajectories : int
        Total number of trajectories simulated
    n_exits : int
        Number of successful exits
    weights : list of float
        Importance weights (for weighted methods)
    computational_cost : int
        Total number of SDE steps computed
    algorithm : str
        Name of algorithm used
    """
    exit_times: List[float] = field(default_factory=list)
    exit_positions: List[np.ndarray] = field(default_factory=list)
    n_trajectories: int = 0
    n_exits: int = 0
    weights: List[float] = field(default_factory=list)
    computational_cost: int = 0
    algorithm: str = ""
    
    def mean_exit_time(self) -> float:
        """Compute mean exit time."""
        if len(self.exit_times) == 0:
            return np.inf
        
        if len(self.weights) > 0:
            # Weighted average
            return np.average(self.exit_times, weights=self.weights)
        else:
            # Unweighted average
            return np.mean(self.exit_times)
    
    def exit_probability(self) -> float:
        """Estimate exit probability."""
        if self.n_trajectories == 0:
            return 0.0
        return self.n_exits / self.n_trajectories
    
    def variance_exit_time(self) -> float:
        """Compute variance of exit time."""
        if len(self.exit_times) < 2:
            return 0.0
        
        if len(self.weights) > 0:
            mean = self.mean_exit_time()
            variance = np.average((np.array(self.exit_times) - mean)**2, 
                                 weights=self.weights)
            return variance
        else:
            return np.var(self.exit_times)
    
    def coefficient_of_variation(self) -> float:
        """Compute coefficient of variation (CV = std/mean)."""
        mean = self.mean_exit_time()
        if mean == 0 or np.isinf(mean):
            return np.inf
        std = np.sqrt(self.variance_exit_time())
        return std / mean


class NaiveMonteCarloSampler:
    """
    Naive Monte Carlo sampling of rare events.
    
    This is the baseline method that simply runs many independent trajectories
    until they exit. For rare events with exponentially small probabilities,
    this method is computationally infeasible.
    
    Computational cost: O(exp(ΔV/ε)) where ΔV is barrier height
    
    This class demonstrates WHY rare-event methods are necessary.
    """
    
    def __init__(self, solver, exit_condition: Callable, 
                 basin_condition: Callable):
        """
        Initialize naive Monte Carlo sampler.
        
        Parameters
        ----------
        solver : SDESolver
            SDE solver instance
        exit_condition : callable
            Function exit_condition(x) -> bool, True when exited target basin
        basin_condition : callable
            Function basin_condition(x) -> bool, True when in starting basin
        """
        self.solver = solver
        self.exit_condition = exit_condition
        self.basin_condition = basin_condition
    
    def sample_exit_times(self, 
                         x0: np.ndarray,
                         n_samples: int,
                         dt: float,
                         max_steps: int = 1000000,
                         seed: Optional[int] = None) -> RareEventResult:
        """
        Sample exit times using naive Monte Carlo.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial position (should satisfy basin_condition)
        n_samples : int
            Number of trajectories to simulate
        dt : float
            Time step
        max_steps : int
            Maximum steps per trajectory
        seed : int, optional
            Random seed
            
        Returns
        -------
        RareEventResult
            Simulation results
        """
        rng = np.random.default_rng(seed)
        
        result = RareEventResult(algorithm="Naive Monte Carlo")
        result.n_trajectories = n_samples
        
        for i in range(n_samples):
            trajectory_seed = rng.integers(0, 2**31)
            traj, exited = self.solver.simulate_until_exit(
                x0, self.exit_condition, dt, max_steps, seed=trajectory_seed
            )
            
            result.computational_cost += len(traj)
            
            if exited:
                result.exit_times.append(traj.times[-1])
                result.exit_positions.append(traj.final_position())
                result.n_exits += 1
        
        return result


class ImportanceSamplingSDE:
    """
    Importance sampling for rare-event simulation.
    
    The key idea is to bias the SDE dynamics to make rare events more likely,
    then reweight the results to obtain unbiased estimates.
    
    Biased SDE:
        dX_t = [-∇V(X_t) + b(X_t)] dt + √(2ε) dW_t
    
    where b(x) is a bias function designed to push trajectories toward the
    target region.
    
    The Radon-Nikodym derivative (importance weight) is:
        w(trajectory) = exp(-∫₀ᵀ [b(X_t)·dX_t - ½|b(X_t)|²dt] / (2ε))
    
    Optimal bias (Girsanov theorem):
        b*(x) = 2ε ∇log(u(x))
    
    where u(x) is the committor function (probability of reaching target before
    returning to source).
    
    In practice, u(x) is unknown, so we use approximations:
    - Linear bias toward target
    - Bias based on reaction coordinate
    """
    
    def __init__(self, solver, exit_condition: Callable,
                 bias_function: Callable):
        """
        Initialize importance sampling.
        
        Parameters
        ----------
        solver : SDESolver
            Base SDE solver
        exit_condition : callable
            Exit condition
        bias_function : callable
            Bias function b(x) -> np.ndarray
        """
        self.solver = solver
        self.exit_condition = exit_condition
        self.bias_function = bias_function
    
    def _compute_weight(self, trajectory: np.ndarray, dt: float) -> float:
        """
        Compute importance weight for a trajectory.
        
        Parameters
        ----------
        trajectory : np.ndarray, shape (n_steps+1, dim)
            Trajectory positions
        dt : float
            Time step
            
        Returns
        -------
        float
            Importance weight
        """
        epsilon = self.solver.epsilon
        log_weight = 0.0
        
        for i in range(len(trajectory) - 1):
            x = trajectory[i]
            x_next = trajectory[i + 1]
            b = self.bias_function(x)
            
            # Increment: b·dX - ½|b|²dt
            dx = x_next - x
            log_weight += np.dot(b, dx) / (2.0 * epsilon)
            log_weight -= 0.5 * np.dot(b, b) * dt / (2.0 * epsilon)
        
        # Avoid numerical overflow
        log_weight = np.clip(log_weight, -100, 100)
        return np.exp(-log_weight)
    
    def sample_exit_times(self,
                         x0: np.ndarray,
                         n_samples: int,
                         dt: float,
                         max_steps: int = 1000000,
                         seed: Optional[int] = None) -> RareEventResult:
        """
        Sample exit times using importance sampling.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial position
        n_samples : int
            Number of biased trajectories
        dt : float
            Time step
        max_steps : int
            Maximum steps per trajectory
        seed : int, optional
            Random seed
            
        Returns
        -------
        RareEventResult
            Weighted simulation results
        """
        rng = np.random.default_rng(seed)
        
        # Create biased solver
        original_grad_V = self.solver.grad_V
        
        def biased_grad_V(x):
            return original_grad_V(x) - self.bias_function(x)
        
        # Temporarily modify solver
        self.solver.grad_V = biased_grad_V
        
        result = RareEventResult(algorithm="Importance Sampling")
        result.n_trajectories = n_samples
        
        for i in range(n_samples):
            trajectory_seed = rng.integers(0, 2**31)
            traj, exited = self.solver.simulate_until_exit(
                x0, self.exit_condition, dt, max_steps, seed=trajectory_seed
            )
            
            result.computational_cost += len(traj)
            
            if exited:
                weight = self._compute_weight(traj.positions, dt)
                result.exit_times.append(traj.times[-1])
                result.exit_positions.append(traj.final_position())
                result.weights.append(weight)
                result.n_exits += 1
        
        # Restore original gradient
        self.solver.grad_V = original_grad_V
        
        return result


class AdaptiveMultilevelSplitting:
    """
    Adaptive Multilevel Splitting (AMS) algorithm.
    
    AMS is a powerful rare-event method that:
    1. Defines a reaction coordinate ξ(x) measuring progress toward target
    2. Adaptively creates levels ξ₀ < ξ₁ < ... < ξ_n
    3. At each level, kills worst trajectories and clones best ones
    4. Estimates probability as product of conditional probabilities
    
    The reaction coordinate should satisfy:
    - ξ(x) = 0 in source basin
    - ξ(x) = 1 in target basin
    - ξ increases along typical transition paths
    
    Theoretical foundation:
        P(A→B) = ∏ᵢ P(reach ξᵢ₊₁ | reached ξᵢ)
    
    Advantages:
    - No bias function needed
    - Automatically adapts to system
    - Efficient for high barriers
    
    Reference: Cérou et al. (2002), "Adaptive multilevel splitting"
    """
    
    def __init__(self, solver, reaction_coordinate: Callable,
                 target_value: float):
        """
        Initialize AMS algorithm.
        
        Parameters
        ----------
        solver : SDESolver
            SDE solver
        reaction_coordinate : callable
            Function ξ(x) -> float measuring progress
        target_value : float
            Target value ξ_target (exit when ξ(x) ≥ target_value)
        """
        self.solver = solver
        self.reaction_coordinate = reaction_coordinate
        self.target_value = target_value
    
    def run(self,
            x0: np.ndarray,
            n_replicas: int,
            dt: float,
            kill_fraction: float = 0.1,
            max_iterations: int = 1000,
            seed: Optional[int] = None) -> RareEventResult:
        """
        Run AMS algorithm.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial position
        n_replicas : int
            Number of parallel replicas
        dt : float
            Time step
        kill_fraction : float
            Fraction of replicas to kill at each level (typically 0.1-0.3)
        max_iterations : int
            Maximum number of iterations
        seed : int, optional
            Random seed
            
        Returns
        -------
        RareEventResult
            AMS results with probability estimate
        """
        rng = np.random.default_rng(seed)
        
        # Initialize replicas
        replicas = [{'position': x0.copy(), 
                    'xi': self.reaction_coordinate(x0),
                    'time': 0.0,
                    'trajectory': [x0.copy()]} 
                   for _ in range(n_replicas)]
        
        result = RareEventResult(algorithm="Adaptive Multilevel Splitting")
        result.n_trajectories = n_replicas
        
        level_probabilities = []
        current_level = self.reaction_coordinate(x0)
        
        for iteration in range(max_iterations):
            # Evolve all replicas for one step
            for rep in replicas:
                step_seed = rng.integers(0, 2**31)
                x_new = self.solver.step(rep['position'], dt, 
                                        np.random.default_rng(step_seed))
                rep['position'] = x_new
                rep['xi'] = self.reaction_coordinate(x_new)
                rep['time'] += dt
                rep['trajectory'].append(x_new.copy())
                result.computational_cost += 1
            
            # Check for exits
            exited_replicas = [rep for rep in replicas 
                             if rep['xi'] >= self.target_value]
            
            if len(exited_replicas) > 0:
                # Record exits
                for rep in exited_replicas:
                    result.exit_times.append(rep['time'])
                    result.exit_positions.append(rep['position'])
                    result.n_exits += 1
                
                # Remove exited replicas
                replicas = [rep for rep in replicas 
                          if rep['xi'] < self.target_value]
                
                if len(replicas) == 0:
                    break
            
            # Adaptive splitting: kill worst, clone best
            if iteration % 10 == 0 and len(replicas) >= 2:
                xi_values = np.array([rep['xi'] for rep in replicas])
                n_kill = max(1, int(kill_fraction * len(replicas)))
                
                # Find worst replicas
                worst_indices = np.argsort(xi_values)[:n_kill]
                
                # Find best replicas to clone
                best_indices = np.argsort(xi_values)[-n_kill:]
                
                # Clone best into worst
                for i, worst_idx in enumerate(worst_indices):
                    best_idx = best_indices[i % len(best_indices)]
                    replicas[worst_idx] = copy.deepcopy(replicas[best_idx])
                
                # Update level
                new_level = np.min(xi_values)
                if new_level > current_level:
                    level_prob = 1.0 - kill_fraction
                    level_probabilities.append(level_prob)
                    current_level = new_level
        
        # Estimate total probability
        if len(level_probabilities) > 0:
            total_prob = np.prod(level_probabilities)
            # Store as weight for mean exit time calculation
            if result.n_exits > 0:
                result.weights = [total_prob] * result.n_exits
        
        return result


class WeightedEnsemble:
    """
    Weighted Ensemble (WE) method for rare-event sampling.
    
    WE maintains an ensemble of trajectories distributed across bins in
    configuration space. At regular intervals:
    1. Bin trajectories by reaction coordinate
    2. Merge or split trajectories within each bin to maintain target count
    3. Adjust weights to preserve probability
    
    This ensures continuous sampling across the entire transition region,
    preventing the ensemble from collapsing into metastable states.
    
    Key parameters:
    - Bin boundaries: partition configuration space
    - Target count per bin: controls resolution
    - Resampling interval: how often to merge/split
    
    Advantages:
    - Maintains steady-state sampling
    - No bias function needed
    - Good for computing transition rates
    
    Reference: Huber & Kim (1996), "Weighted-ensemble Brownian dynamics"
    """
    
    def __init__(self, solver, reaction_coordinate: Callable,
                 bin_boundaries: np.ndarray):
        """
        Initialize Weighted Ensemble method.
        
        Parameters
        ----------
        solver : SDESolver
            SDE solver
        reaction_coordinate : callable
            Reaction coordinate ξ(x)
        bin_boundaries : np.ndarray
            Boundaries defining bins: [ξ₀, ξ₁, ..., ξ_n]
        """
        self.solver = solver
        self.reaction_coordinate = reaction_coordinate
        self.bin_boundaries = np.sort(bin_boundaries)
        self.n_bins = len(bin_boundaries) - 1
    
    def _assign_bin(self, xi: float) -> int:
        """Assign reaction coordinate value to bin."""
        bin_idx = np.searchsorted(self.bin_boundaries, xi) - 1
        return np.clip(bin_idx, 0, self.n_bins - 1)
    
    def _resample_bin(self, walkers: List[Dict], target_count: int,
                     rng: np.random.Generator) -> List[Dict]:
        """
        Resample walkers in a bin to achieve target count.
        
        Parameters
        ----------
        walkers : list of dict
            Walkers in this bin
        target_count : int
            Desired number of walkers
        rng : np.random.Generator
            Random number generator
            
        Returns
        -------
        list of dict
            Resampled walkers
        """
        if len(walkers) == 0:
            return []
        
        if len(walkers) == target_count:
            return walkers
        
        # Total weight in bin
        total_weight = sum(w['weight'] for w in walkers)
        
        # Resample with replacement
        weights = np.array([w['weight'] for w in walkers])
        weights /= weights.sum()
        
        indices = rng.choice(len(walkers), size=target_count, 
                           replace=True, p=weights)
        
        new_walkers = []
        for idx in indices:
            walker = copy.deepcopy(walkers[idx])
            walker['weight'] = total_weight / target_count
            new_walkers.append(walker)
        
        return new_walkers
    
    def run(self,
            x0: np.ndarray,
            n_walkers: int,
            target_per_bin: int,
            dt: float,
            resample_interval: int,
            n_iterations: int,
            exit_condition: Callable,
            seed: Optional[int] = None) -> RareEventResult:
        """
        Run Weighted Ensemble simulation.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial position
        n_walkers : int
            Total number of walkers
        target_per_bin : int
            Target number of walkers per bin
        dt : float
            Time step
        resample_interval : int
            Steps between resampling
        n_iterations : int
            Total number of iterations
        exit_condition : callable
            Exit condition function
        seed : int, optional
            Random seed
            
        Returns
        -------
        RareEventResult
            WE simulation results
        """
        rng = np.random.default_rng(seed)
        
        # Initialize walkers
        walkers = []
        for i in range(n_walkers):
            walkers.append({
                'position': x0.copy(),
                'weight': 1.0 / n_walkers,
                'time': 0.0,
                'xi': self.reaction_coordinate(x0)
            })
        
        result = RareEventResult(algorithm="Weighted Ensemble")
        
        for iteration in range(n_iterations):
            # Evolve all walkers
            for walker in walkers:
                step_seed = rng.integers(0, 2**31)
                x_new = self.solver.step(walker['position'], dt,
                                        np.random.default_rng(step_seed))
                walker['position'] = x_new
                walker['time'] += dt
                walker['xi'] = self.reaction_coordinate(x_new)
                result.computational_cost += 1
                
                # Check for exit
                if exit_condition(x_new):
                    result.exit_times.append(walker['time'])
                    result.exit_positions.append(x_new)
                    result.weights.append(walker['weight'])
                    result.n_exits += 1
            
            # Remove exited walkers
            walkers = [w for w in walkers if not exit_condition(w['position'])]
            
            if len(walkers) == 0:
                break
            
            # Resampling
            if iteration % resample_interval == 0:
                # Bin walkers
                bins = defaultdict(list)
                for walker in walkers:
                    bin_idx = self._assign_bin(walker['xi'])
                    bins[bin_idx].append(walker)
                
                # Resample each bin
                new_walkers = []
                for bin_idx in range(self.n_bins):
                    if bin_idx in bins:
                        resampled = self._resample_bin(bins[bin_idx], 
                                                      target_per_bin, rng)
                        new_walkers.extend(resampled)
                
                walkers = new_walkers
                result.n_trajectories = len(walkers)
        
        return result


def create_linear_bias(x_target: np.ndarray, strength: float = 1.0) -> Callable:
    """
    Create a linear bias function toward target.
    
    b(x) = strength · (x_target - x) / |x_target - x|
    
    Parameters
    ----------
    x_target : np.ndarray
        Target position
    strength : float
        Bias strength
        
    Returns
    -------
    callable
        Bias function
    """
    def bias(x: np.ndarray) -> np.ndarray:
        direction = x_target - x
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return np.zeros_like(x)
        return strength * direction / norm
    
    return bias


def create_reaction_coordinate_1d(x_start: np.ndarray, 
                                  x_end: np.ndarray) -> Callable:
    """
    Create simple 1D reaction coordinate.
    
    ξ(x) = (x - x_start) · (x_end - x_start) / |x_end - x_start|²
    
    This projects position onto the line connecting start and end.
    
    Parameters
    ----------
    x_start : np.ndarray
        Starting position
    x_end : np.ndarray
        Target position
        
    Returns
    -------
    callable
        Reaction coordinate function
    """
    direction = x_end - x_start
    norm_sq = np.dot(direction, direction)
    
    def xi(x: np.ndarray) -> float:
        displacement = x - x_start
        return np.dot(displacement, direction) / norm_sq
    
    return xi
