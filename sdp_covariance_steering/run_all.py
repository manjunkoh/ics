"""
Main script to replicate all results from:
"Discrete-time Optimal Covariance Steering via Semidefinite Programming"
by George Rapakoulias and Panagiotis Tsiotras, CDC 2023.

Generates:
  - Figure 1: Convexified domain illustration
  - Figure 2: 2D quadrotor path planning (Example 1)
  - Figures 3 & 4: Nonlinear quadrotor covariance steering (Example 2)
  - Tables I & II: Runtime comparison
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)


def main():
    print("=" * 70)
    print("Replicating: Discrete-time Optimal Covariance Steering")
    print("               via Semidefinite Programming")
    print("         Rapakoulias & Tsiotras, CDC 2023")
    print("=" * 70)

    # Figure 1: Convexified domain
    print("\n[1/4] Generating Figure 1...")
    from figure1_convexified_domain import plot_figure1
    plot_figure1()

    # Figure 2: Example 1 - 2D quadrotor path planning
    print("\n[2/4] Running Example 1 (2D Quadrotor)...")
    from example1_2d_quadrotor import run_example1
    run_example1()

    # Figures 3 & 4: Example 2 - Nonlinear quadrotor
    print("\n[3/4] Running Example 2 (Nonlinear Quadrotor)...")
    from example2_nonlinear_quadrotor import run_example2
    run_example2()

    # Tables I & II: Runtime comparison
    print("\n[4/4] Running Runtime Comparison...")
    import numpy as np
    np.random.seed(42)
    from runtime_comparison import run_table1, run_table2, generate_tables_figure
    results1 = run_table1()
    results2 = run_table2()
    generate_tables_figure(results1, results2)

    print("\n" + "=" * 70)
    print("All figures generated in sdp_covariance_steering/figures/")
    print("=" * 70)


if __name__ == '__main__':
    main()
