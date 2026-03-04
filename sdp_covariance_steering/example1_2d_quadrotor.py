"""
Example 1: Path planning for a quadrotor in a 2D plane.

Replicates Figure 2 from:
"Discrete-time Optimal Covariance Steering via Semidefinite Programming"
by Rapakoulias and Tsiotras, CDC 2023.

The lateral and longitudinal dynamics are modeled as a triple integrator
with state x = [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y].
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from cs_sdp_solver import solve_full_cs_problem

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def build_triple_integrator_2d(dt, N):
    """
    Build discrete-time triple integrator for 2D motion.

    State: [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y] (6 states)
    Input: [u_x, u_y] (2 inputs, change in acceleration / jerk * dt)

    Euler discretization:
        pos_{k+1} = pos_k + dt * vel_k
        vel_{k+1} = vel_k + dt * acc_k
        acc_{k+1} = acc_k + u_k

    A = [I_2,    dt*I_2,  0_2  ]
        [0_2,    I_2,     dt*I_2]
        [0_2,    0_2,     I_2   ]

    B = [0_2]
        [0_2]
        [I_2]

    D = 0.1 * I_6
    """
    n = 6
    p = 2
    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))

    A = np.block([
        [I2, dt * I2, Z2],
        [Z2, I2, dt * I2],
        [Z2, Z2, I2]
    ])

    B = np.block([
        [Z2],
        [Z2],
        [I2]
    ])

    D = 0.1 * np.eye(n)

    A_list = [A] * N
    B_list = [B] * N
    D_list = [D] * N

    return A_list, B_list, D_list


def plot_confidence_ellipse(ax, mu, Sigma, n_std=3.0, **kwargs):
    """Plot a 2D confidence ellipse given mean and covariance."""
    # Extract 2x2 position covariance
    if Sigma.shape[0] > 2:
        cov = Sigma[:2, :2]
    else:
        cov = Sigma

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)

    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])

    ellipse = Ellipse(xy=(mu[0], mu[1]), width=width, height=height,
                      angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def run_example1():
    """Run Example 1: 2D quadrotor path planning."""
    print("=" * 60)
    print("Example 1: 2D Quadrotor Path Planning (Triple Integrator)")
    print("=" * 60)

    # Parameters
    dt = 0.1  # time step
    N = 60    # horizon
    n = 6     # states
    p = 2     # inputs

    # Build system
    A_list, B_list, D_list = build_triple_integrator_2d(dt, N)

    # Boundary conditions
    Sigma_i = np.eye(n)
    Sigma_f = 0.1 * np.eye(n)
    mu_i = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu_f = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Cost matrices
    Q_list = [np.eye(n)] * N
    R_list = [np.eye(p)] * N

    # Waypoints at k=20 and k=40
    # Position components (indices 0, 1) constrained
    waypoints = {
        20: ([0, 1], [14.0, 5.0]),
        40: ([0, 1], [6.0, -4.0]),
    }

    # Chance constraint parameters
    # State constraints: position bounded in [-3, 22] x [-7, 7]
    # P(alpha_x^T x <= beta_x) >= 1 - epsilon_x
    # Using linearized form (15a) around Sigma_r
    alpha_x_list = [
        np.array([1.0, 0, 0, 0, 0, 0]),    # x_1 <= 22
        np.array([-1.0, 0, 0, 0, 0, 0]),   # -x_1 <= 3 => x_1 >= -3
        np.array([0, 1.0, 0, 0, 0, 0]),    # x_2 <= 7
        np.array([0, -1.0, 0, 0, 0, 0]),   # -x_2 <= 7 => x_2 >= -7
    ]
    beta_x_list = [22.0, 3.0, 7.0, 7.0]

    # Control constraints: u bounded in [-25, 25] x [-25, 25]
    alpha_u_list = [
        np.array([1.0, 0]),    # u_1 <= 25
        np.array([-1.0, 0]),   # -u_1 <= 25 => u_1 >= -25
        np.array([0, 1.0]),    # u_2 <= 25
        np.array([0, -1.0]),   # -u_2 <= 25 => u_2 >= -25
    ]
    beta_u_list = [25.0, 25.0, 25.0, 25.0]

    # Linearization reference values
    Sigma_r = 1.2 * np.eye(n)
    Y_r = 15.0 * np.eye(p)

    # Violation probabilities
    epsilon_x = 0.1
    epsilon_u = 0.1

    print("Solving constrained covariance steering problem...")
    results = solve_full_cs_problem(
        A_list, B_list, D_list, Sigma_i, Sigma_f, mu_i, mu_f, N,
        Q_list, R_list,
        terminal_ineq=False,
        waypoints=waypoints,
        alpha_x_list=alpha_x_list, beta_x_list=beta_x_list,
        alpha_u_list=alpha_u_list, beta_u_list=beta_u_list,
        Sigma_r=Sigma_r, Y_r=Y_r,
        epsilon_x=epsilon_x, epsilon_u=epsilon_u,
    )

    print(f"  Mean cost: {results['cost_mean']:.4f}")
    print(f"  Covariance cost: {results['cost_cov']:.4f}")
    print(f"  Total cost: {results['cost_total']:.4f}")

    # Plot Figure 2
    plot_figure2(results, N, dt, waypoints)

    return results


def plot_figure2(results, N, dt, waypoints):
    """
    Replicate Figure 2 from the paper.
    Left: Resulting trajectory with 3-sigma confidence ellipses.
    Right: Required control effort with 3-sigma bounds.
    Green lines = feasible set boundaries.
    Blue ellipses = intermediate covariance.
    Red ellipses = initial, final, and waypoint positions.
    """
    mu_traj = results['mu_traj']
    v_traj = results['v_traj']
    Sigma_traj = results['Sigma_traj']
    K_traj = results['K_traj']
    Y_traj = results['Y_traj']

    n = mu_traj.shape[1]
    p = v_traj.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left panel: Trajectory with confidence ellipses ----
    ax1 = axes[0]

    # Draw feasible region (green lines)
    x_bounds = [-3, 22]
    y_bounds = [-7, 7]
    rect_x = [x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0], x_bounds[0]]
    rect_y = [y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0]]
    ax1.plot(rect_x, rect_y, 'g-', linewidth=2.5)

    # Plot 3-sigma confidence ellipses at regular intervals
    # Blue for intermediate, red for initial/final/waypoints
    ellipse_steps = list(range(0, N + 1, 3))
    for k_special in [0, N] + list(waypoints.keys()):
        if k_special not in ellipse_steps:
            ellipse_steps.append(k_special)
    ellipse_steps = sorted(set(ellipse_steps))

    # Draw blue ellipses first, then red on top
    for k in ellipse_steps:
        if k not in [0, N] and k not in waypoints:
            mu_k = mu_traj[k]
            Sigma_k = Sigma_traj[k]
            plot_confidence_ellipse(ax1, mu_k[:2], Sigma_k[:2, :2], n_std=3.0,
                                    facecolor='blue', alpha=0.15,
                                    edgecolor='blue', linewidth=0.8)

    for k in [0, N] + list(waypoints.keys()):
        mu_k = mu_traj[k]
        Sigma_k = Sigma_traj[k]
        plot_confidence_ellipse(ax1, mu_k[:2], Sigma_k[:2, :2], n_std=3.0,
                                facecolor='red', alpha=0.3,
                                edgecolor='red', linewidth=1.5)

    # Plot mean trajectory (dashed black) on top
    ax1.plot(mu_traj[:, 0], mu_traj[:, 1], 'k--', linewidth=1.5)

    ax1.set_xlabel('Position $x_1$', fontsize=12)
    ax1.set_ylabel('Position $x_2$', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-5, 24])
    ax1.set_ylim([-9, 9])

    # ---- Right panel: Control effort ----
    ax2 = axes[1]
    time_ctrl = np.arange(N) * dt

    # Plot control mean signals
    ax2.plot(time_ctrl, v_traj[:, 0], 'k--', linewidth=1.5, label='$v_1$')
    ax2.plot(time_ctrl, v_traj[:, 1], 'k-.', linewidth=1.5, label='$v_2$')

    # 3-sigma bounds for each control component (light blue shading)
    for i in range(p):
        sigma_u = np.array([np.sqrt(max(Y_traj[k, i, i], 0)) for k in range(N)])
        upper = v_traj[:, i] + 3 * sigma_u
        lower = v_traj[:, i] - 3 * sigma_u
        ax2.fill_between(time_ctrl, lower, upper, alpha=0.25, color='lightblue')

    # Draw control bounds (green lines)
    ax2.axhline(y=25, color='g', linewidth=2.5)
    ax2.axhline(y=-25, color='g', linewidth=2.5)

    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Control Input', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add text labels instead of legend to match paper style
    fig.text(0.13, 0.92, 'Fig. 2. Left: Resulting trajectory, Right: required control effort.\n'
             'Green lines represent the feasible part of the state space and control action space.',
             fontsize=10, style='italic', ha='left', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig('figures/figure2_trajectory_and_control.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  Figure 2 saved to figures/figure2_trajectory_and_control.png")


if __name__ == '__main__':
    run_example1()
