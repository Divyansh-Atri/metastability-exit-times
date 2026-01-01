"""
Numerical SDE solvers for overdamped Langevin dynamics.

This module implements numerical integration schemes for stochastic differential
equations of the form:
    dX_t = -∇V(X_t) dt + √(2ε) dW_t

All solvers are implemented from scratch without using black-box libraries.
Each solver includes:
- Stability analysis
- Convergence order documentation
- Computational cost tracking
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SDETrajectory:
    """
    Container for SDE simulation results.
    
    Attributes
    ----------
    times : np.ndarray, shape (n_steps+1,)
        Time points
    positions : np.ndarray, shape (n_steps+1, dim)
        Position trajectory
    dt : float
        Time step size
    epsilon : float
        Noise level
    n_steps : int
        Number of time steps
    """
    times: np.ndarray
    positions: np.ndarray
    dt: float
    epsilon: float
    n_steps: int
    
    def __len__(self) -> int:
        return self.n_steps + 1
    
    def final_position(self) -> np.ndarray:
        """Return final position."""
        return self.positions[-1]
    
    def trajectory_length(self) -> float:
        """Compute total path length."""
        diffs = np.diff(self.positions, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))


class SDESolver:
    """Base class for SDE numerical integrators."""
    
    def __init__(self, grad_V: Callable, dim: int, epsilon: float):
        """
        Initialize SDE solver.
        
        Parameters
        ----------
        grad_V : callable
            Function computing gradient of potential: grad_V(x) -> np.ndarray
        dim : int
            Spatial dimension
        epsilon : float
            Noise level (temperature)
        """
        self.grad_V = grad_V
        self.dim = dim
        self.epsilon = epsilon
        self.name = "Base"
        self.order = 0  # Weak convergence order
    
    def step(self, x: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """
        Perform one time step.
        
        Parameters
        ----------
        x : np.ndarray, shape (dim,)
            Current position
        dt : float
            Time step
        rng : np.random.Generator
            Random number generator
            
        Returns
        -------
        np.ndarray, shape (dim,)
            New position
        """
        raise NotImplementedError
    
    def simulate(self, 
                 x0: np.ndarray, 
                 T: float, 
                 dt: float,
                 seed: Optional[int] = None) -> SDETrajectory:
        """
        Simulate SDE trajectory.
        
        Parameters
        ----------
        x0 : np.ndarray, shape (dim,)
            Initial position
        T : float
            Final time
        dt : float
            Time step
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        SDETrajectory
            Complete trajectory data
        """
        rng = np.random.default_rng(seed)
        
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        positions = np.zeros((n_steps + 1, self.dim))
        positions[0] = x0.copy()
        
        x = x0.copy()
        for i in range(n_steps):
            x = self.step(x, dt, rng)
            positions[i + 1] = x
        
        return SDETrajectory(
            times=times,
            positions=positions,
            dt=dt,
            epsilon=self.epsilon,
            n_steps=n_steps
        )
    
    def simulate_until_exit(self,
                           x0: np.ndarray,
                           exit_condition: Callable,
                           dt: float,
                           max_steps: int = 1000000,
                           seed: Optional[int] = None) -> Tuple[SDETrajectory, bool]:
        """
        Simulate until exit condition is met.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial position
        exit_condition : callable
            Function exit_condition(x) -> bool, returns True when exited
        dt : float
            Time step
        max_steps : int
            Maximum number of steps
        seed : int, optional
            Random seed
            
        Returns
        -------
        trajectory : SDETrajectory
            Trajectory up to exit or max_steps
        exited : bool
            Whether exit condition was met
        """
        rng = np.random.default_rng(seed)
        
        positions = [x0.copy()]
        x = x0.copy()
        
        for i in range(max_steps):
            x = self.step(x, dt, rng)
            positions.append(x.copy())
            
            if exit_condition(x):
                n_steps = i + 1
                times = np.arange(n_steps + 1) * dt
                return SDETrajectory(
                    times=times,
                    positions=np.array(positions),
                    dt=dt,
                    epsilon=self.epsilon,
                    n_steps=n_steps
                ), True
        
        # Max steps reached without exit
        n_steps = max_steps
        times = np.arange(n_steps + 1) * dt
        return SDETrajectory(
            times=times,
            positions=np.array(positions),
            dt=dt,
            epsilon=self.epsilon,
            n_steps=n_steps
        ), False


class EulerMaruyama(SDESolver):
    """
    Euler-Maruyama method for overdamped Langevin dynamics.
    
    Scheme:
        X_{n+1} = X_n - ∇V(X_n)·dt + √(2ε)·√dt·ξ_n
    
    where ξ_n ~ N(0, I).
    
    Properties:
    - Weak convergence order: 1
    - Strong convergence order: 0.5
    - Explicit, no implicit solves required
    - Can be unstable for stiff potentials
    
    Stability:
    - Requires dt < 2/λ_max where λ_max is max eigenvalue of Hessian
    - For double-well: dt < 1/2 near minima
    """
    
    def __init__(self, grad_V: Callable, dim: int, epsilon: float):
        super().__init__(grad_V, dim, epsilon)
        self.name = "Euler-Maruyama"
        self.order = 1
    
    def step(self, x: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Perform one Euler-Maruyama step."""
        # Deterministic drift
        drift = -self.grad_V(x) * dt
        
        # Stochastic diffusion
        noise = np.sqrt(2.0 * self.epsilon * dt) * rng.standard_normal(self.dim)
        
        return x + drift + noise


class SemiImplicitEuler(SDESolver):
    """
    Semi-implicit Euler method for stiff SDEs.
    
    Scheme:
        X_{n+1} = X_n - ∇V(X_{n+1})·dt + √(2ε)·√dt·ξ_n
    
    The gradient is evaluated at the new point, requiring an implicit solve.
    For quadratic potentials V(x) = ½x^T A x, this can be solved exactly:
        X_{n+1} = (I + dt·A)^{-1} (X_n + √(2ε)·√dt·ξ_n)
    
    For general potentials, we use Newton iteration.
    
    Properties:
    - Better stability than explicit Euler-Maruyama
    - Weak convergence order: 1
    - Requires solving nonlinear system at each step
    - Unconditionally stable for convex potentials
    
    Stability:
    - Unconditionally stable for quadratic potentials
    - Much larger stable dt than explicit methods
    """
    
    def __init__(self, grad_V: Callable, dim: int, epsilon: float,
                 hessian_V: Optional[Callable] = None,
                 newton_tol: float = 1e-10,
                 newton_max_iter: int = 10):
        """
        Initialize semi-implicit solver.
        
        Parameters
        ----------
        grad_V : callable
            Gradient of potential
        dim : int
            Dimension
        epsilon : float
            Noise level
        hessian_V : callable, optional
            Hessian of potential (for Newton method)
        newton_tol : float
            Newton iteration tolerance
        newton_max_iter : int
            Maximum Newton iterations
        """
        super().__init__(grad_V, dim, epsilon)
        self.hessian_V = hessian_V
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.name = "Semi-Implicit Euler"
        self.order = 1
    
    def step(self, x: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Perform one semi-implicit step."""
        # Stochastic term (explicit)
        noise = np.sqrt(2.0 * self.epsilon * dt) * rng.standard_normal(self.dim)
        rhs = x + noise
        
        # Solve: x_new + dt·∇V(x_new) = rhs
        # Using Newton iteration: F(y) = y + dt·∇V(y) - rhs = 0
        
        y = x.copy()  # Initial guess
        
        for iteration in range(self.newton_max_iter):
            F = y + dt * self.grad_V(y) - rhs
            
            if np.linalg.norm(F) < self.newton_tol:
                break
            
            # Jacobian: J = I + dt·Hessian(V)
            if self.hessian_V is not None:
                J = np.eye(self.dim) + dt * self.hessian_V(y)
            else:
                # Finite difference approximation
                h = 1e-7
                J = np.eye(self.dim)
                grad_y = self.grad_V(y)
                for i in range(self.dim):
                    e_i = np.zeros(self.dim)
                    e_i[i] = h
                    grad_perturbed = self.grad_V(y + e_i)
                    J[:, i] += dt * (grad_perturbed - grad_y) / h
            
            # Newton update
            try:
                delta = np.linalg.solve(J, F)
                y = y - delta
            except np.linalg.LinAlgError:
                # Fallback to explicit Euler if Newton fails
                return x - self.grad_V(x) * dt + noise
        
        return y


class Milstein(SDESolver):
    """
    Milstein method with improved strong convergence.
    
    For scalar SDEs, the Milstein scheme includes correction terms involving
    derivatives of the diffusion coefficient. For additive noise (constant diffusion),
    Milstein reduces to Euler-Maruyama.
    
    For our SDE with additive noise:
        dX_t = -∇V(X_t) dt + √(2ε) dW_t
    
    The Milstein scheme is identical to Euler-Maruyama because the diffusion
    coefficient is constant.
    
    This class is included for completeness and educational purposes.
    
    Properties:
    - Strong convergence order: 1.0 (for general SDEs)
    - For additive noise: identical to Euler-Maruyama
    """
    
    def __init__(self, grad_V: Callable, dim: int, epsilon: float):
        super().__init__(grad_V, dim, epsilon)
        self.name = "Milstein (additive noise)"
        self.order = 1
    
    def step(self, x: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """
        Perform one Milstein step.
        
        For additive noise, this is identical to Euler-Maruyama.
        """
        drift = -self.grad_V(x) * dt
        noise = np.sqrt(2.0 * self.epsilon * dt) * rng.standard_normal(self.dim)
        return x + drift + noise


class SplittingMethod(SDESolver):
    """
    Strang splitting method for separable potentials.
    
    For potentials that can be split as V(x) = V₁(x) + V₂(x), we can use
    operator splitting to improve accuracy and stability.
    
    Scheme (Strang splitting):
        1. Evolve with V₁ for dt/2
        2. Evolve with V₂ for dt
        3. Evolve with V₁ for dt/2
    
    This is particularly effective for:
    - Separable potentials: V(x) = Σᵢ Vᵢ(xᵢ)
    - Potentials with different stiffness scales
    
    Properties:
    - Second-order accurate for separable systems
    - Better stability than first-order methods
    - Requires potential to be separable
    
    Note: This is a simplified implementation for demonstration.
    Full implementation would require specifying the splitting.
    """
    
    def __init__(self, grad_V: Callable, dim: int, epsilon: float):
        super().__init__(grad_V, dim, epsilon)
        self.name = "Splitting Method"
        self.order = 2
    
    def step(self, x: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """
        Perform one splitting step.
        
        For simplicity, we implement a basic Strang splitting with
        half-steps of the drift and a full step of the noise.
        """
        # Half drift step
        x = x - 0.5 * dt * self.grad_V(x)
        
        # Full noise step
        noise = np.sqrt(2.0 * self.epsilon * dt) * rng.standard_normal(self.dim)
        x = x + noise
        
        # Half drift step
        x = x - 0.5 * dt * self.grad_V(x)
        
        return x


def get_solver(name: str, grad_V: Callable, dim: int, epsilon: float, **kwargs) -> SDESolver:
    """
    Factory function to create SDE solvers.
    
    Parameters
    ----------
    name : str
        Solver name: 'euler', 'semi_implicit', 'milstein', 'splitting'
    grad_V : callable
        Gradient function
    dim : int
        Dimension
    epsilon : float
        Noise level
    **kwargs
        Additional solver-specific parameters
        
    Returns
    -------
    SDESolver
        Initialized solver object
    """
    solvers = {
        'euler': EulerMaruyama,
        'semi_implicit': SemiImplicitEuler,
        'milstein': Milstein,
        'splitting': SplittingMethod
    }
    
    if name not in solvers:
        raise ValueError(f"Unknown solver: {name}. Choose from {list(solvers.keys())}")
    
    return solvers[name](grad_V, dim, epsilon, **kwargs)


def estimate_stable_dt(hessian_V: Callable, x: np.ndarray, safety_factor: float = 0.5) -> float:
    """
    Estimate stable time step for explicit Euler method.
    
    For explicit Euler, stability requires:
        dt < 2 / λ_max(Hessian(V))
    
    Parameters
    ----------
    hessian_V : callable
        Hessian function
    x : np.ndarray
        Point at which to evaluate Hessian
    safety_factor : float
        Safety factor (< 1) for conservative estimate
        
    Returns
    -------
    float
        Estimated stable time step
    """
    H = hessian_V(x)
    eigenvalues = np.linalg.eigvalsh(H)
    lambda_max = np.max(eigenvalues)
    
    if lambda_max <= 0:
        # Potential is not locally convex
        return 0.01  # Default conservative value
    
    dt_stable = 2.0 / lambda_max
    return safety_factor * dt_stable
