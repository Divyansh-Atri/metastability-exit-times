"""
Multi-well potential energy landscapes for metastability studies.

This module implements various potential energy functions V(x) that exhibit
multiple local minima separated by energy barriers. These potentials are
fundamental to studying rare transitions in stochastic dynamical systems.

All potentials are implemented with:
- Analytical gradients for exact drift computation
- Hessian matrices for stability analysis
- Barrier height calculations for theoretical predictions
"""

import numpy as np
from typing import Tuple, Callable
from scipy.optimize import minimize


class Potential:
    """Base class for potential energy functions."""
    
    def __init__(self, dim: int):
        """
        Initialize potential.
        
        Parameters
        ----------
        dim : int
            Spatial dimension
        """
        self.dim = dim
    
    def V(self, x: np.ndarray) -> float:
        """
        Evaluate potential energy.
        
        Parameters
        ----------
        x : np.ndarray, shape (dim,)
            Position vector
            
        Returns
        -------
        float
            Potential energy V(x)
        """
        raise NotImplementedError
    
    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate gradient of potential.
        
        Parameters
        ----------
        x : np.ndarray, shape (dim,)
            Position vector
            
        Returns
        -------
        np.ndarray, shape (dim,)
            Gradient ∇V(x)
        """
        raise NotImplementedError
    
    def hessian_V(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate Hessian matrix of potential.
        
        Parameters
        ----------
        x : np.ndarray, shape (dim,)
            Position vector
            
        Returns
        -------
        np.ndarray, shape (dim, dim)
            Hessian matrix ∇²V(x)
        """
        raise NotImplementedError
    
    def find_minima(self) -> list:
        """
        Find local minima of the potential.
        
        Returns
        -------
        list of np.ndarray
            List of local minima positions
        """
        raise NotImplementedError
    
    def barrier_height(self, x_start: np.ndarray, x_end: np.ndarray) -> float:
        """
        Estimate barrier height between two minima.
        
        Parameters
        ----------
        x_start : np.ndarray
            Starting minimum
        x_end : np.ndarray
            Target minimum
            
        Returns
        -------
        float
            Approximate barrier height
        """
        raise NotImplementedError


class SymmetricDoubleWell(Potential):
    """
    Symmetric double-well potential in d dimensions.
    
    V(x) = (x₁² - 1)² + ω² Σᵢ₌₂ᵈ xᵢ²
    
    This potential has two symmetric wells at x₁ = ±1, with a barrier at x₁ = 0.
    The parameter ω controls the stiffness in transverse directions.
    
    Theoretical properties:
    - Barrier height: ΔV = 1
    - Minima: (±1, 0, ..., 0)
    - Saddle point: (0, 0, ..., 0)
    """
    
    def __init__(self, dim: int, omega: float = 2.0):
        """
        Initialize symmetric double-well potential.
        
        Parameters
        ----------
        dim : int
            Spatial dimension (must be ≥ 1)
        omega : float, default=2.0
            Transverse stiffness parameter
        """
        super().__init__(dim)
        self.omega = omega
        self._minima = [np.array([1.0] + [0.0]*(dim-1)),
                        np.array([-1.0] + [0.0]*(dim-1))]
        self._saddle = np.zeros(dim)
        self._barrier = 1.0
    
    def V(self, x: np.ndarray) -> float:
        """Evaluate potential energy."""
        x1_term = (x[0]**2 - 1.0)**2
        transverse_term = self.omega**2 * np.sum(x[1:]**2) if self.dim > 1 else 0.0
        return x1_term + transverse_term
    
    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient."""
        grad = np.zeros(self.dim)
        grad[0] = 4.0 * x[0] * (x[0]**2 - 1.0)
        if self.dim > 1:
            grad[1:] = 2.0 * self.omega**2 * x[1:]
        return grad
    
    def hessian_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian matrix."""
        H = np.zeros((self.dim, self.dim))
        H[0, 0] = 12.0 * x[0]**2 - 4.0
        if self.dim > 1:
            np.fill_diagonal(H[1:, 1:], 2.0 * self.omega**2)
        return H
    
    def find_minima(self) -> list:
        """Return known minima."""
        return self._minima.copy()
    
    def barrier_height(self, x_start: np.ndarray, x_end: np.ndarray) -> float:
        """Return known barrier height."""
        return self._barrier


class AsymmetricDoubleWell(Potential):
    """
    Asymmetric double-well potential.
    
    V(x) = (x₁² - 1)² - α·x₁ + ω² Σᵢ₌₂ᵈ xᵢ²
    
    The parameter α introduces asymmetry, making one well deeper than the other.
    This models systems with biased transitions.
    
    Theoretical properties:
    - For α > 0: right well is deeper
    - Barrier heights are different in forward/backward directions
    - Metastable state depends on noise level
    """
    
    def __init__(self, dim: int, alpha: float = 0.3, omega: float = 2.0):
        """
        Initialize asymmetric double-well potential.
        
        Parameters
        ----------
        dim : int
            Spatial dimension
        alpha : float, default=0.3
            Asymmetry parameter (0 = symmetric)
        omega : float, default=2.0
            Transverse stiffness
        """
        super().__init__(dim)
        self.alpha = alpha
        self.omega = omega
        self._compute_critical_points()
    
    def _compute_critical_points(self):
        """Numerically find minima and saddle points."""
        # Find minima by optimization
        self._minima = []
        for x0 in [-1.5, 1.5]:
            init = np.zeros(self.dim)
            init[0] = x0
            res = minimize(self.V, init, jac=self.grad_V, method='BFGS')
            if res.success:
                self._minima.append(res.x)
        
        # Find saddle point
        init = np.zeros(self.dim)
        res = minimize(lambda x: -self.V(x), init, 
                      jac=lambda x: -self.grad_V(x), method='BFGS')
        if res.success:
            self._saddle = res.x
    
    def V(self, x: np.ndarray) -> float:
        """Evaluate potential energy."""
        x1_term = (x[0]**2 - 1.0)**2 - self.alpha * x[0]
        transverse_term = self.omega**2 * np.sum(x[1:]**2) if self.dim > 1 else 0.0
        return x1_term + transverse_term
    
    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient."""
        grad = np.zeros(self.dim)
        grad[0] = 4.0 * x[0] * (x[0]**2 - 1.0) - self.alpha
        if self.dim > 1:
            grad[1:] = 2.0 * self.omega**2 * x[1:]
        return grad
    
    def hessian_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian matrix."""
        H = np.zeros((self.dim, self.dim))
        H[0, 0] = 12.0 * x[0]**2 - 4.0
        if self.dim > 1:
            np.fill_diagonal(H[1:, 1:], 2.0 * self.omega**2)
        return H
    
    def find_minima(self) -> list:
        """Return computed minima."""
        return self._minima.copy()
    
    def barrier_height(self, x_start: np.ndarray, x_end: np.ndarray) -> float:
        """Compute barrier height numerically."""
        V_start = self.V(x_start)
        V_saddle = self.V(self._saddle)
        return V_saddle - V_start


class MullerBrownPotential(Potential):
    """
    Müller-Brown potential (2D only).
    
    Classic test case from molecular dynamics with three wells
    and complex transition pathways.
    
    V(x, y) = Σᵢ₌₁⁴ Aᵢ exp(aᵢ(x-x̄ᵢ)² + bᵢ(x-x̄ᵢ)(y-ȳᵢ) + cᵢ(y-ȳᵢ)²)
    
    Features:
    - Three local minima
    - Multiple transition pathways
    - Realistic molecular energy landscape
    """
    
    def __init__(self):
        """Initialize Müller-Brown potential (fixed 2D)."""
        super().__init__(2)
        
        # Parameters from Müller & Brown (1979)
        self.A = np.array([-200.0, -100.0, -170.0, 15.0])
        self.a = np.array([-1.0, -1.0, -6.5, 0.7])
        self.b = np.array([0.0, 0.0, 11.0, 0.6])
        self.c = np.array([-10.0, -10.0, -6.5, 0.7])
        self.x_bar = np.array([1.0, 0.0, -0.5, -1.0])
        self.y_bar = np.array([0.0, 0.5, 1.5, 1.0])
        
        # Known minima (approximate)
        self._minima = [
            np.array([-0.558, 1.442]),   # Well 1
            np.array([0.623, 0.028]),     # Well 2
            np.array([-0.050, 0.467])     # Well 3
        ]
    
    def V(self, x: np.ndarray) -> float:
        """Evaluate potential energy."""
        V_total = 0.0
        for i in range(4):
            dx = x[0] - self.x_bar[i]
            dy = x[1] - self.y_bar[i]
            exponent = (self.a[i] * dx**2 + 
                       self.b[i] * dx * dy + 
                       self.c[i] * dy**2)
            V_total += self.A[i] * np.exp(exponent)
        return V_total
    
    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient."""
        grad = np.zeros(2)
        for i in range(4):
            dx = x[0] - self.x_bar[i]
            dy = x[1] - self.y_bar[i]
            exponent = (self.a[i] * dx**2 + 
                       self.b[i] * dx * dy + 
                       self.c[i] * dy**2)
            exp_term = self.A[i] * np.exp(exponent)
            
            grad[0] += exp_term * (2.0 * self.a[i] * dx + self.b[i] * dy)
            grad[1] += exp_term * (self.b[i] * dx + 2.0 * self.c[i] * dy)
        
        return grad
    
    def hessian_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian matrix."""
        H = np.zeros((2, 2))
        for i in range(4):
            dx = x[0] - self.x_bar[i]
            dy = x[1] - self.y_bar[i]
            exponent = (self.a[i] * dx**2 + 
                       self.b[i] * dx * dy + 
                       self.c[i] * dy**2)
            exp_term = self.A[i] * np.exp(exponent)
            
            grad_x = 2.0 * self.a[i] * dx + self.b[i] * dy
            grad_y = self.b[i] * dx + 2.0 * self.c[i] * dy
            
            H[0, 0] += exp_term * (grad_x**2 + 2.0 * self.a[i])
            H[0, 1] += exp_term * (grad_x * grad_y + self.b[i])
            H[1, 0] = H[0, 1]
            H[1, 1] += exp_term * (grad_y**2 + 2.0 * self.c[i])
        
        return H
    
    def find_minima(self) -> list:
        """Return known minima."""
        return self._minima.copy()
    
    def barrier_height(self, x_start: np.ndarray, x_end: np.ndarray) -> float:
        """Estimate barrier height by string method."""
        # Simple linear interpolation to find maximum
        n_points = 100
        path = np.linspace(x_start, x_end, n_points)
        energies = np.array([self.V(p) for p in path])
        V_max = np.max(energies)
        V_start = self.V(x_start)
        return V_max - V_start


class CoupledHighDimWells(Potential):
    """
    Coupled high-dimensional double-well system.
    
    V(x) = Σᵢ₌₁ᵈ [(xᵢ² - 1)² + κ·xᵢ·xᵢ₊₁]
    
    This potential couples adjacent coordinates, creating a high-dimensional
    energy landscape with 2^d local minima. The coupling parameter κ controls
    the interaction strength.
    
    Features:
    - Exponentially many minima (2^d)
    - Frustrated transitions in high dimensions
    - Tests curse of dimensionality
    """
    
    def __init__(self, dim: int, kappa: float = 0.1):
        """
        Initialize coupled high-dimensional potential.
        
        Parameters
        ----------
        dim : int
            Spatial dimension
        kappa : float, default=0.1
            Coupling strength between adjacent coordinates
        """
        super().__init__(dim)
        self.kappa = kappa
        
        # Primary minima (all coordinates at ±1)
        self._minima = [
            np.ones(dim),
            -np.ones(dim)
        ]
    
    def V(self, x: np.ndarray) -> float:
        """Evaluate potential energy."""
        V_total = 0.0
        
        # Double-well terms
        for i in range(self.dim):
            V_total += (x[i]**2 - 1.0)**2
        
        # Coupling terms
        for i in range(self.dim - 1):
            V_total += self.kappa * x[i] * x[i+1]
        
        return V_total
    
    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient."""
        grad = np.zeros(self.dim)
        
        # Double-well gradient
        for i in range(self.dim):
            grad[i] = 4.0 * x[i] * (x[i]**2 - 1.0)
        
        # Coupling gradient
        for i in range(self.dim - 1):
            grad[i] += self.kappa * x[i+1]
            grad[i+1] += self.kappa * x[i]
        
        return grad
    
    def hessian_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian matrix."""
        H = np.zeros((self.dim, self.dim))
        
        # Diagonal terms from double-well
        for i in range(self.dim):
            H[i, i] = 12.0 * x[i]**2 - 4.0
        
        # Off-diagonal coupling terms
        for i in range(self.dim - 1):
            H[i, i+1] = self.kappa
            H[i+1, i] = self.kappa
        
        return H
    
    def find_minima(self) -> list:
        """Return primary minima."""
        return self._minima.copy()
    
    def barrier_height(self, x_start: np.ndarray, x_end: np.ndarray) -> float:
        """Estimate barrier height along coordinate-wise path."""
        # For the all-positive to all-negative transition,
        # the barrier is approximately d (flipping one coordinate at a time)
        return float(self.dim)


def get_potential(name: str, dim: int, **kwargs) -> Potential:
    """
    Factory function to create potential objects.
    
    Parameters
    ----------
    name : str
        Potential name: 'symmetric', 'asymmetric', 'muller', 'coupled'
    dim : int
        Spatial dimension
    **kwargs
        Additional parameters for specific potentials
        
    Returns
    -------
    Potential
        Initialized potential object
    """
    potentials = {
        'symmetric': SymmetricDoubleWell,
        'asymmetric': AsymmetricDoubleWell,
        'muller': MullerBrownPotential,
        'coupled': CoupledHighDimWells
    }
    
    if name not in potentials:
        raise ValueError(f"Unknown potential: {name}. Choose from {list(potentials.keys())}")
    
    if name == 'muller':
        if dim != 2:
            raise ValueError("Müller-Brown potential is only defined in 2D")
        return potentials[name]()
    
    return potentials[name](dim, **kwargs)
