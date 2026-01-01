"""
Metastability and Rare Transitions in Stochastic Dynamical Systems

A PhD-grade computational mathematics project implementing advanced numerical
methods for studying rare events in metastable systems.
"""

from .potentials import (
    Potential,
    SymmetricDoubleWell,
    AsymmetricDoubleWell,
    MullerBrownPotential,
    CoupledHighDimWells,
    get_potential
)

from .sde_solvers import (
    SDESolver,
    SDETrajectory,
    EulerMaruyama,
    SemiImplicitEuler,
    Milstein,
    SplittingMethod,
    get_solver,
    estimate_stable_dt
)

from .rare_event_algorithms import (
    RareEventResult,
    NaiveMonteCarloSampler,
    ImportanceSamplingSDE,
    AdaptiveMultilevelSplitting,
    WeightedEnsemble,
    create_linear_bias,
    create_reaction_coordinate_1d
)

from .analysis import (
    ExitTimeStatistics,
    analyze_exit_times,
    fit_exponential_distribution,
    verify_eyring_kramers_law,
    analyze_transition_paths,
    compute_effective_sample_size,
    bootstrap_confidence_interval,
    compare_algorithms,
    test_convergence_rate,
    analyze_dimension_scaling
)

from .visualization import (
    plot_potential_1d,
    plot_potential_2d,
    plot_potential_3d,
    plot_trajectory_2d,
    plot_exit_time_distribution,
    plot_eyring_kramers_verification,
    plot_dimension_scaling,
    plot_transition_paths,
    plot_algorithm_comparison,
    plot_high_dim_projection,
    create_summary_figure
)

__version__ = '1.0.0'
__author__ = 'Divyansh Atri'

__all__ = [
    # Potentials
    'Potential', 'SymmetricDoubleWell', 'AsymmetricDoubleWell',
    'MullerBrownPotential', 'CoupledHighDimWells', 'get_potential',
    
    # SDE Solvers
    'SDESolver', 'SDETrajectory', 'EulerMaruyama', 'SemiImplicitEuler',
    'Milstein', 'SplittingMethod', 'get_solver', 'estimate_stable_dt',
    
    # Rare Event Algorithms
    'RareEventResult', 'NaiveMonteCarloSampler', 'ImportanceSamplingSDE',
    'AdaptiveMultilevelSplitting', 'WeightedEnsemble',
    'create_linear_bias', 'create_reaction_coordinate_1d',
    
    # Analysis
    'ExitTimeStatistics', 'analyze_exit_times', 'fit_exponential_distribution',
    'verify_eyring_kramers_law', 'analyze_transition_paths',
    'compute_effective_sample_size', 'bootstrap_confidence_interval',
    'compare_algorithms', 'test_convergence_rate', 'analyze_dimension_scaling',
    
    # Visualization
    'plot_potential_1d', 'plot_potential_2d', 'plot_potential_3d',
    'plot_trajectory_2d', 'plot_exit_time_distribution',
    'plot_eyring_kramers_verification', 'plot_dimension_scaling',
    'plot_transition_paths', 'plot_algorithm_comparison',
    'plot_high_dim_projection', 'create_summary_figure'
]
