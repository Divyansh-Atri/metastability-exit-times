#!/usr/bin/env python3
"""
Quick setup and verification script for the metastability project.

This script:
1. Checks Python version
2. Verifies all dependencies are installed
3. Tests that all modules can be imported
4. Runs a minimal simulation to verify everything works
5. Provides next steps
"""

import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check Python version is 3.8+"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [FAIL] ERROR: Python 3.8+ required")
        return False
    print("  [OK] Python version OK")
    return True

def check_dependencies():
    """Check all required packages are installed"""
    print("\nChecking dependencies...")
    
    required = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'jupyter': 'Jupyter'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} not found")
            all_ok = False
    
    return all_ok

def test_imports():
    """Test that all project modules can be imported"""
    print("\nTesting project modules...")
    
    sys.path.insert(0, os.path.dirname(__file__))
    
    modules = [
        'src.potentials',
        'src.sde_solvers',
        'src.rare_event_algorithms',
        'src.analysis',
        'src.visualization'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except Exception as e:
            print(f"  [FAIL] {module}: {e}")
            all_ok = False
    
    return all_ok

def run_minimal_test():
    """Run a minimal simulation to verify functionality"""
    print("\nRunning minimal simulation test...")
    
    try:
        import numpy as np
        from src.potentials import SymmetricDoubleWell
        from src.sde_solvers import EulerMaruyama
        
        # Create simple 2D potential
        potential = SymmetricDoubleWell(dim=2, omega=2.0)
        
        # Test potential evaluation
        x = np.array([0.0, 0.0])
        V = potential.V(x)
        grad = potential.grad_V(x)
        
        print(f"  Potential at origin: V(0) = {V:.4f}")
        print(f"  Gradient at origin: ∇V(0) = [{grad[0]:.4f}, {grad[1]:.4f}]")
        
        # Test SDE solver
        solver = EulerMaruyama(potential.grad_V, dim=2, epsilon=0.5)
        x0 = np.array([-1.0, 0.0])
        
        traj = solver.simulate(x0, T=1.0, dt=0.01, seed=42)
        
        print(f"  Simulated {len(traj)} time steps")
        print(f"  Final position: [{traj.positions[-1, 0]:.4f}, {traj.positions[-1, 1]:.4f}]")
        
        print("  [OK] Minimal test passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print_header("Metastability Project Setup Verification")
    
    print("""
This script verifies that your environment is correctly set up to run
the metastability and rare-event simulation project.
""")
    
    # Run all checks
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Module imports", test_imports),
        ("Minimal simulation", run_minimal_test)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Setup Verification Summary")
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"  {status}: {name}")
    
    if all_passed:
        print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OK] ALL CHECKS PASSED - Your environment is ready!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEXT STEPS:

1. Run the master simulation:
   
   cd scripts
   python master_simulation.py

   This will demonstrate the rare-event challenge and generate figures.

2. Explore the Jupyter notebooks:
   
   jupyter notebook notebooks/

   Start with 01_potential_landscapes.ipynb and work through in order.

3. Read the comprehensive documentation:
   
   See README.md for theoretical background, method descriptions,
   and detailed explanations of all results.

4. Experiment with parameters:
   
   Try different noise levels, dimensions, and potentials to explore
   the rich behavior of metastable systems.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For questions or issues, please refer to the documentation or open an issue.

Happy simulating!
""")
    else:
        print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[FAIL] SOME CHECKS FAILED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please fix the issues above before proceeding.

To install missing dependencies:

   pip install -r requirements.txt

If you continue to have issues, please check:
  • Python version is 3.8 or higher
  • All files are in the correct locations
  • You're running this script from the project root directory

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        sys.exit(1)

if __name__ == '__main__':
    main()
