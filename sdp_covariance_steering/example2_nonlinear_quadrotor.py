"""
Example 2: Nonlinear quadrotor covariance steering.

Replicates Figures 3 and 4 from:
"Discrete-time Optimal Covariance Steering via Semidefinite Programming"
by Rapakoulias and Tsiotras, CDC 2023.

Controls the uncertainty around an aggressive reference trajectory
for a nonlinear quadrotor using covariance steering on the linearized
deviation dynamics.

Paper parameters: dt=0.01, N=500 (requires MOSEK).
Without MOSEK, uses dt=0.02, N=250 with SCS.
Use --fast flag for quick testing (dt=0.05, N=100).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cs_sdp_solver import solve_covariance_steering_sdp
from quadrotor_dynamics import (
    compute_nominal_quadrotor_state_and_control,
    compute_jacobians_numerical,
    generate_nominal_trajectory_triple_integrator,
)


def run_example2(fast=False):
    """Run Example 2: Nonlinear quadrotor covariance steering."""
    print("=" * 60)
    print("Example 2: Nonlinear Quadrotor Covariance Steering")
    print("=" * 60)

    # Parameters
    # Check for MOSEK availability
    import cvxpy as cp
    has_mosek = 'MOSEK' in cp.installed_solvers()

    if fast:
        print("  Running in FAST mode (dt=0.05, N=100)")
        dt = 0.05
        N = 100
        tube_start = 10  # k >= 50 in paper (N=500), scaled
    elif has_mosek:
        print("  Running with paper parameters (dt=0.01, N=500, MOSEK)")
        dt = 0.01
        N = 500
        tube_start = 50  # paper: k >= 50
    else:
        print("  MOSEK not available. Using dt=0.02, N=250 with SCS.")
        print("  (Paper uses dt=0.01, N=500 with MOSEK)")
        dt = 0.02
        N = 250
        tube_start = 25  # k >= 50 for N=500, scaled

    n = 9   # state dimension (r, v, q)
    p = 4   # input dimension (τ, ω)
    T = N * dt  # total time

    # Boundary conditions for nominal trajectory (Eq. 23)
    pos_0 = np.array([0.0, 0.5, 0.0])
    pos_N = np.array([0.0, -0.5, 0.0])

    # Four position waypoints (evenly spaced at 1s intervals)
    # Estimated from paper Figure 3: figure-8/infinity sign pattern
    # spanning x in [-4, 4], y in [-2, 2] with altitude variations.
    # The trajectory goes: start → northeast to right lobe → south →
    # back through center → northwest to left lobe → south → end.
    wp_steps = [N // 5, 2 * N // 5, 3 * N // 5, 4 * N // 5]
    waypoints_nom = {
        wp_steps[0]: np.array([3.0, 1.5, 1.0]),     # upper right lobe
        wp_steps[1]: np.array([3.0, -1.5, 0.5]),    # lower right lobe
        wp_steps[2]: np.array([-3.0, 1.5, 1.0]),    # upper left lobe
        wp_steps[3]: np.array([-3.0, -1.5, 0.0]),   # lower left lobe
    }

    # Step 1: Generate nominal trajectory using triple integrator
    print("Step 1: Generating nominal trajectory (minimum jerk)...")
    pos_traj = generate_nominal_trajectory_triple_integrator(
        pos_0, pos_N, waypoints_nom, dt, N
    )
    print(f"  Nominal trajectory generated: {pos_traj.shape}")

    # Step 2: Compute nominal quadrotor state and control
    print("Step 2: Computing nominal quadrotor states/controls...")
    x_nom, u_nom = compute_nominal_quadrotor_state_and_control(pos_traj, dt)
    print(f"  Nominal state: {x_nom.shape}, Nominal control: {u_nom.shape}")

    # Step 3: Linearize dynamics around nominal trajectory
    print("Step 3: Linearizing dynamics around nominal trajectory...")
    A_list = []
    B_list = []
    D_list = []

    for k in range(N):
        Ak, Bk, Dk = compute_jacobians_numerical(x_nom[k], u_nom[k], dt)
        A_list.append(Ak)
        B_list.append(Bk)
        # D_k from linearization + D̃ for discretization/linearization errors
        D_tilde = 0.001 * np.eye(n, 6)
        D_list.append(Dk + D_tilde)

    print(f"  Linearized {N} time steps")

    # Step 4: Set up covariance steering problem
    print("Step 4: Setting up covariance steering problem...")

    # Cost matrices (paper: Q_k = 10*I_9, R_k = 0.1*I_4)
    Q_list = [10.0 * np.eye(n)] * N
    R_list = [0.1 * np.eye(p)] * N

    # Boundary covariances (paper Eq.)
    Sigma_i = np.block([
        [1e-2 * np.eye(3), np.zeros((3, 6))],
        [np.zeros((6, 3)), 1e-3 * np.eye(6)]
    ])
    Sigma_f = np.block([
        [5e-4 * np.eye(3), np.zeros((3, 6))],
        [np.zeros((6, 3)), 1e-3 * np.eye(6)]
    ])

    # Covariance tube constraint: position covariance <= 2e-3*I_3 for k >= tube_start
    covariance_constraints = [{
        'k': list(range(tube_start, N + 1)),
        'indices': [0, 1, 2],
        'bound': 2e-3 * np.eye(3)
    }]

    # Step 5: Solve covariance steering SDP
    print("Step 5: Solving covariance steering SDP...")
    print(f"  Problem: N={N}, n={n}, p={p}")
    print(f"  Tube constraints: k={tube_start} to k={N}")

    import cvxpy as cp

    # Build solver priority list
    has_mosek = 'MOSEK' in cp.installed_solvers()
    solvers_to_try = []
    if has_mosek:
        solvers_to_try.append(('MOSEK', cp.MOSEK))
    solvers_to_try.append(('SCS', cp.SCS))
    solvers_to_try.append(('CLARABEL', cp.CLARABEL))

    # Try progressively relaxed configurations:
    # 1. With tube constraints, equality terminal (paper setup)
    # 2. Without tube constraints, equality terminal
    # 3. Without tube constraints, inequality terminal (Sigma_N <= Sigma_f)
    configs = [
        {'tube': True, 'ineq': False, 'label': 'tube + equality terminal'},
        {'tube': False, 'ineq': False, 'label': 'no tube + equality terminal'},
        {'tube': False, 'ineq': True, 'label': 'no tube + inequality terminal'},
    ]

    solved = False
    for solver_name, solver in solvers_to_try:
        if solved:
            break
        for cfg in configs:
            if solved:
                break
            cov_cc = covariance_constraints if cfg['tube'] else None
            desc = f"{solver_name}, {cfg['label']}"
            print(f"  Trying: {desc}...")
            try:
                Sigma_traj, U_traj, Y_traj, K_traj, cost_cov = \
                    solve_covariance_steering_sdp(
                        A_list, B_list, D_list, Sigma_i, Sigma_f, N,
                        Q_list, R_list,
                        terminal_ineq=cfg['ineq'],
                        covariance_constraints=cov_cc,
                        solver=solver, verbose=False
                    )
                print(f"  Solved with: {desc}")
                solved = True
            except Exception as e:
                print(f"  Failed ({desc}): {e}")

    if not solved:
        raise RuntimeError("All solver configurations failed. "
                           "The problem may be infeasible with these parameters.")

    print(f"  Covariance cost: {cost_cov:.4f}")

    # Plot Figures 3 and 4
    plot_figure3(x_nom, Sigma_traj, N, dt)
    plot_figure4(u_nom, Y_traj, K_traj, Sigma_traj, N, dt)

    return x_nom, u_nom, Sigma_traj, K_traj, Y_traj


def plot_covariance_ellipsoid_3d(ax, center, cov_3x3, n_std=3.0,
                                 color='blue', alpha=0.2):
    """Plot a 3D covariance ellipsoid."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_3x3)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Generate sphere points
    u_pts = np.linspace(0, 2 * np.pi, 20)
    v_pts = np.linspace(0, np.pi, 10)
    x_sph = np.outer(np.cos(u_pts), np.sin(v_pts))
    y_sph = np.outer(np.sin(u_pts), np.sin(v_pts))
    z_sph = np.outer(np.ones_like(u_pts), np.cos(v_pts))

    # Scale by eigenvalues and rotate
    radii = n_std * np.sqrt(eigenvalues)
    for i in range(len(u_pts)):
        for j in range(len(v_pts)):
            pt = np.array([x_sph[i, j], y_sph[i, j], z_sph[i, j]])
            pt_scaled = eigenvectors @ (radii * pt)
            x_sph[i, j] = pt_scaled[0] + center[0]
            y_sph[i, j] = pt_scaled[1] + center[1]
            z_sph[i, j] = pt_scaled[2] + center[2]

    ax.plot_surface(x_sph, y_sph, z_sph, color=color, alpha=alpha,
                    linewidth=0)


def plot_confidence_ellipse_2d(ax, mu, cov_2x2, n_std=3.0, **kwargs):
    """Plot a 2D confidence ellipse given mean and 2x2 covariance."""
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
    eigenvalues = np.maximum(eigenvalues, 0)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])
    ellipse = Ellipse(xy=(mu[0], mu[1]), width=width, height=height,
                      angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def plot_figure3(x_nom, Sigma_traj, N, dt):
    """
    Replicate Figure 3: Covariance Steering around nominal trajectory.
    Paper shows a 2D x-y projection with covariance ellipses.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot 3-sigma position covariance ellipses at selected steps (x-y plane)
    n_ellipses = min(25, N // 5)
    step = max(1, N // n_ellipses)
    ellipse_steps = list(range(0, N + 1, step))
    if N not in ellipse_steps:
        ellipse_steps.append(N)

    # Blue ellipses for intermediate steps
    for k in ellipse_steps:
        if k != 0 and k != N:
            center = x_nom[k, :2]  # r_x, r_y
            cov_xy = Sigma_traj[k, :2, :2]
            plot_confidence_ellipse_2d(ax, center, cov_xy, n_std=3.0,
                                       facecolor='blue', alpha=0.15,
                                       edgecolor='blue', linewidth=0.5)

    # Red ellipses for initial and terminal
    for k in [0, N]:
        center = x_nom[k, :2]
        cov_xy = Sigma_traj[k, :2, :2]
        plot_confidence_ellipse_2d(ax, center, cov_xy, n_std=3.0,
                                   facecolor='red', alpha=0.3,
                                   edgecolor='red', linewidth=1.5)

    # Plot nominal trajectory (black line) on top
    ax.plot(x_nom[:, 0], x_nom[:, 1], 'k-', linewidth=2)

    # Mark start and end
    ax.plot(x_nom[0, 0], x_nom[0, 1], 'ro', markersize=8, zorder=5)
    ax.plot(x_nom[N, 0], x_nom[N, 1], 'ro', markersize=8, zorder=5)

    # Labels
    ax.annotate('initial covariance', xy=(x_nom[0, 0], x_nom[0, 1]),
                xytext=(x_nom[0, 0] + 0.1, x_nom[0, 1] + 0.15),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('terminal covariance', xy=(x_nom[N, 0], x_nom[N, 1]),
                xytext=(x_nom[N, 0] + 0.1, x_nom[N, 1] - 0.15),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('$x$ [m]', fontsize=12)
    ax.set_ylabel('$y$ [m]', fontsize=12)
    ax.set_title('Covariance Steering around nominal trajectory', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01, 'Fig. 3. Uncertainty control around nominal trajectory.',
             fontsize=10, style='italic', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('figures/figure3_quadrotor_trajectory.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  Figure 3 saved to figures/figure3_quadrotor_trajectory.png")


def plot_figure4(u_nom, Y_traj, K_traj, Sigma_traj, N, dt):
    """
    Replicate Figure 4: Required control effort.
    Paper layout: 2x2 grid with τ, ω_x, ω_y, ω_z.
    Each subplot: black line = nominal, light blue = 3-sigma bounds.
    """
    time = np.arange(N) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    control_labels = ['$\\tau$', '$\\omega_x$', '$\\omega_y$', '$\\omega_z$']
    subplot_titles = ['Control effort in $\\tau$', 'Control effort in $\\omega_x$',
                      'Control effort in $\\omega_y$', 'Control effort in $\\omega_z$']

    for i, ax in enumerate(axes.flat):
        # Nominal control (black line)
        ax.plot(time, u_nom[:, i], 'k-', linewidth=1.5)

        # 3-sigma bounds from Y_k (light blue shading)
        sigma_u = np.array([np.sqrt(max(Y_traj[k, i, i], 0)) for k in range(N)])
        upper = u_nom[:, i] + 3 * sigma_u
        lower = u_nom[:, i] - 3 * sigma_u
        ax.fill_between(time, lower, upper, alpha=0.3, color='lightblue')

        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(control_labels[i], fontsize=12)
        ax.set_title(subplot_titles[i], fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01, 'Fig. 4. Required control effort.',
             fontsize=10, style='italic', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('figures/figure4_quadrotor_control.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  Figure 4 saved to figures/figure4_quadrotor_control.png")


if __name__ == '__main__':
    fast = '--fast' in sys.argv
    run_example2(fast=fast)
