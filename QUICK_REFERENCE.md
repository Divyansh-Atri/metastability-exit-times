# Quick Reference Guide

## Metastability and Rare Transitions Project

---

## File Structure

```
metastability-rare-events/
├── README.md                    # Comprehensive documentation
├── PROJECT_SUMMARY.md           # Project overview and highlights
├── QUICK_REFERENCE.md           # This file
├── requirements.txt             # Python dependencies
├── setup_check.py               # Verify installation
│
├── src/                         # Core implementation
│   ├── __init__.py
│   ├── potentials.py           # Multi-well potentials
│   ├── sde_solvers.py          # Numerical SDE integrators
│   ├── rare_event_algorithms.py # Advanced sampling methods
│   ├── analysis.py             # Statistical analysis
│   └── visualization.py        # Publication-quality plots
│
├── scripts/
│   └── master_simulation.py    # Standalone demonstration
│
├── notebooks/
│   └── master_simulation.ipynb # Interactive exploration
│
├── plots/                       # Generated figures
└── data/                        # Simulation results
```

---

## Quick Start (3 Steps)

### 1. Verify Setup
```bash
python setup_check.py
```

### 2. Run Master Simulation
```bash
cd scripts
python master_simulation.py
```

### 3. Explore Notebook
```bash
jupyter notebook notebooks/master_simulation.ipynb
```

---

## Core Concepts

### The Problem

Study rare transitions in stochastic systems:
```
dX_t = -∇V(X_t) dt + √(2ε) dW_t
```

For small noise `ε`, transitions are exponentially rare:
```
τ ≈ exp(ΔV/ε)
```

### The Challenge

- For ΔV/ε = 10: τ ≈ 22,000 time units
- Naive Monte Carlo: ~2.2 million SDE steps
- Computationally infeasible for small ε

### The Solution

Advanced rare-event methods:
- **AMS**: 100-1000× speedup
- **Importance Sampling**: Bias + reweight
- **Weighted Ensemble**: Maintain distribution

---

## Key Modules

### Potentials (`src/potentials.py`)

```python
from src.potentials import SymmetricDoubleWell

# Create potential
potential = SymmetricDoubleWell(dim=2, omega=2.0)

# Evaluate
V = potential.V(x)              # Energy
grad = potential.grad_V(x)      # Gradient
H = potential.hessian_V(x)      # Hessian
minima = potential.find_minima() # Local minima
```

**Available potentials:**
- `SymmetricDoubleWell`: Classic test case
- `AsymmetricDoubleWell`: Biased transitions
- `MullerBrownPotential`: Molecular dynamics (2D only)
- `CoupledHighDimWells`: High-dimensional

### SDE Solvers (`src/sde_solvers.py`)

```python
from src.sde_solvers import EulerMaruyama

# Create solver
solver = EulerMaruyama(potential.grad_V, dim=2, epsilon=0.2)

# Simulate
traj = solver.simulate(x0, T=10.0, dt=0.01, seed=42)

# Simulate until exit
traj, exited = solver.simulate_until_exit(
    x0, exit_condition, dt=0.01, max_steps=100000
)
```

**Available solvers:**
- `EulerMaruyama`: Explicit, fast
- `SemiImplicitEuler`: Stable, requires Newton
- `Milstein`: Higher-order (same as EM for additive noise)
- `SplittingMethod`: For separable potentials

### Rare-Event Algorithms (`src/rare_event_algorithms.py`)

```python
from src.rare_event_algorithms import AdaptiveMultilevelSplitting

# Create reaction coordinate
from src.rare_event_algorithms import create_reaction_coordinate_1d
xi = create_reaction_coordinate_1d(x_start, x_target)

# Run AMS
ams = AdaptiveMultilevelSplitting(solver, xi, target_value=0.9)
result = ams.run(x0, n_replicas=100, dt=0.01, 
                kill_fraction=0.2, max_iterations=10000)

# Access results
print(f"Exits: {result.n_exits}")
print(f"Mean time: {result.mean_exit_time()}")
print(f"Cost: {result.computational_cost} steps")
```

**Available algorithms:**
- `NaiveMonteCarloSampler`: Baseline
- `ImportanceSamplingSDE`: Bias + reweight
- `AdaptiveMultilevelSplitting`: Most robust
- `WeightedEnsemble`: Steady-state sampling

### Analysis (`src/analysis.py`)

```python
from src.analysis import analyze_exit_times, verify_eyring_kramers_law

# Analyze exit times
stats = analyze_exit_times(exit_times, weights=None)
print(f"Mean: {stats.mean} ± {stats.std}")
print(f"CV: {stats.cv}")

# Verify Eyring-Kramers law
ek_results = verify_eyring_kramers_law(
    epsilon_values, mean_exit_times, barrier_height
)
print(f"Fitted barrier: {ek_results['fitted_barrier']}")
print(f"R²: {ek_results['r_squared']}")
```

### Visualization (`src/visualization.py`)

```python
from src.visualization import plot_potential_2d, plot_exit_time_distribution

# Plot potential
fig, ax = plt.subplots()
plot_potential_2d(potential, (-2, 2), (-2, 2), ax=ax)

# Plot exit times
plot_exit_time_distribution(exit_times, weights=None, ax=ax)
```

---

## Common Tasks

### Task 1: Simulate a Transition

```python
from src.potentials import SymmetricDoubleWell
from src.sde_solvers import EulerMaruyama

# Setup
potential = SymmetricDoubleWell(dim=2, omega=2.0)
solver = EulerMaruyama(potential.grad_V, dim=2, epsilon=0.2)

x0 = np.array([-1.0, 0.0])
def exit_condition(x):
    return x[0] > 0.5

# Simulate
traj, exited = solver.simulate_until_exit(
    x0, exit_condition, dt=0.01, max_steps=50000, seed=42
)

if exited:
    print(f"Exit time: {traj.times[-1]:.2f}")
```

### Task 2: Compare Methods

```python
from src.rare_event_algorithms import (
    NaiveMonteCarloSampler,
    AdaptiveMultilevelSplitting
)

# Naive MC
naive = NaiveMonteCarloSampler(solver, exit_condition, basin_condition)
naive_result = naive.sample_exit_times(x0, n_samples=10, dt=0.01)

# AMS
ams = AdaptiveMultilevelSplitting(solver, reaction_coord, target_value=0.9)
ams_result = ams.run(x0, n_replicas=100, dt=0.01)

# Compare
print(f"Naive cost: {naive_result.computational_cost}")
print(f"AMS cost: {ams_result.computational_cost}")
print(f"Speedup: {naive_result.computational_cost / ams_result.computational_cost:.1f}×")
```

### Task 3: Verify Eyring-Kramers Law

```python
from src.analysis import verify_eyring_kramers_law

# Run simulations for different epsilon
epsilon_values = [0.5, 0.3, 0.2, 0.15]
mean_exit_times = []

for eps in epsilon_values:
    solver_eps = EulerMaruyama(potential.grad_V, dim=2, epsilon=eps)
    ams_eps = AdaptiveMultilevelSplitting(solver_eps, reaction_coord, 0.9)
    result = ams_eps.run(x0, n_replicas=50, dt=0.01)
    
    stats = analyze_exit_times(np.array(result.exit_times))
    mean_exit_times.append(stats.mean)

# Verify law
ek_results = verify_eyring_kramers_law(
    np.array(epsilon_values),
    np.array(mean_exit_times),
    barrier_height=1.0
)

print(f"R²: {ek_results['r_squared']:.6f}")
print(f"Fitted barrier: {ek_results['fitted_barrier']:.4f}")
```

---

## Parameter Guidelines

### Noise Level (ε)

- **ε = 0.5**: Easy, τ ≈ 7
- **ε = 0.2**: Moderate, τ ≈ 148
- **ε = 0.1**: Hard, τ ≈ 22,000
- **ε = 0.05**: Very hard, τ ≈ 5×10⁸

### Time Step (dt)

- **Euler-Maruyama**: dt < 2/λ_max ≈ 0.5
- **Semi-Implicit**: dt < 5.0 (more stable)
- **Recommended**: dt = 0.01 (safe for all)

### AMS Parameters

- **n_replicas**: 50-100 (more = better statistics)
- **kill_fraction**: 0.1-0.3 (0.2 is typical)
- **max_iterations**: 10,000-50,000

### Dimension

- **d = 2**: Easy to visualize
- **d = 5-10**: Moderate challenge
- **d = 20-50**: High-dimensional effects

---

## Troubleshooting

### No exits observed

**Problem**: Naive MC finds no transitions

**Solution**: 
- Use AMS instead
- Increase max_steps
- Increase noise level ε

### Numerical instability

**Problem**: Trajectories explode

**Solution**:
- Reduce time step dt
- Use SemiImplicitEuler
- Check potential is well-defined

### Poor AMS performance

**Problem**: AMS doesn't find exits

**Solution**:
- Check reaction coordinate is monotonic
- Increase n_replicas
- Adjust kill_fraction
- Increase max_iterations

### High variance in estimates

**Problem**: Large error bars

**Solution**:
- Increase sample size
- Use AMS instead of naive MC
- Check importance weights (ESS)

---

## Performance Tips

1. **Use AMS for rare events** (ΔV/ε > 5)
2. **Start with 2D** for visualization
3. **Use dt = 0.01** as default
4. **Run multiple seeds** for uncertainty
5. **Save results** for expensive runs

---

## Example Workflow

```python
# 1. Setup
from src.potentials import SymmetricDoubleWell
from src.sde_solvers import EulerMaruyama
from src.rare_event_algorithms import AdaptiveMultilevelSplitting
from src.analysis import analyze_exit_times
import numpy as np

# 2. Create system
potential = SymmetricDoubleWell(dim=2, omega=2.0)
solver = EulerMaruyama(potential.grad_V, dim=2, epsilon=0.2)

# 3. Define problem
x_start = np.array([-1.0, 0.0])
x_target = np.array([1.0, 0.0])

from src.rare_event_algorithms import create_reaction_coordinate_1d
xi = create_reaction_coordinate_1d(x_start, x_target)

# 4. Run simulation
ams = AdaptiveMultilevelSplitting(solver, xi, target_value=0.9)
result = ams.run(x_start, n_replicas=100, dt=0.01, 
                kill_fraction=0.2, max_iterations=10000, seed=42)

# 5. Analyze
stats = analyze_exit_times(np.array(result.exit_times))
print(f"Mean exit time: {stats.mean:.2f} ± {stats.std:.2f}")
print(f"Computational cost: {result.computational_cost:,} steps")

# 6. Visualize
from src.visualization import plot_exit_time_distribution
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_exit_time_distribution(np.array(result.exit_times), ax=ax)
plt.show()
```

---

## Further Reading

- **README.md**: Complete theoretical background
- **PROJECT_SUMMARY.md**: Project highlights
- **Master simulation**: `scripts/master_simulation.py`
- **Interactive notebook**: `notebooks/master_simulation.ipynb`

---

## Citation

If you use this code in research, please cite:

```
Divyansh Atri (2026). Metastability and Rare Transitions in 
Stochastic Dynamical Systems: A Computational Study.
```

---

**Questions?** See README.md or open an issue.

**Ready to explore rare events!**
