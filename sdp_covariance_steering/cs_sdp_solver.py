"""
Core SDP-based Covariance Steering Solver

Implements the methods from:
"Discrete-time Optimal Covariance Steering via Semidefinite Programming"
by George Rapakoulias and Panagiotis Tsiotras, CDC 2023.

Three problem variants are implemented:
  1. Unconstrained CS with equality terminal covariance (Problem 7)
  2. Unconstrained CS with inequality terminal covariance (Problem 8)
  3. Constrained CS with chance constraints (Problem 17)
"""

import numpy as np
import cvxpy as cp
from scipy.stats import norm


def solve_mean_steering(A_list, B_list, mu_i, mu_f, N, Q_list, R_list,
                        waypoints=None):
    """
    Solve the mean steering problem.

    Parameters
    ----------
    A_list : list of (n, n) arrays
        System matrices for k = 0, ..., N-1.
    B_list : list of (n, p) arrays
        Input matrices for k = 0, ..., N-1.
    mu_i : (n,) array
        Initial mean.
    mu_f : (n,) array
        Terminal mean.
    N : int
        Steering horizon.
    Q_list : list of (n, n) arrays
        State cost matrices.
    R_list : list of (p, p) arrays
        Control cost matrices.
    waypoints : dict, optional
        {time_step: (indices, values)} for position waypoints.

    Returns
    -------
    mu_traj : (N+1, n) array
        Mean trajectory.
    v_traj : (N, p) array
        Feedforward control trajectory.
    cost_mean : float
        Mean steering cost.
    """
    n = mu_i.shape[0]
    p = B_list[0].shape[1]

    # Decision variables
    mu = [cp.Variable(n) for _ in range(N + 1)]
    v = [cp.Variable(p) for _ in range(N)]

    constraints = []
    # Initial and terminal conditions
    constraints.append(mu[0] == mu_i)
    constraints.append(mu[N] == mu_f)

    # Mean dynamics
    for k in range(N):
        constraints.append(mu[k + 1] == A_list[k] @ mu[k] + B_list[k] @ v[k])

    # Waypoint constraints
    if waypoints is not None:
        for k_wp, (indices, values) in waypoints.items():
            for idx, val in zip(indices, values):
                constraints.append(mu[k_wp][idx] == val)

    # Cost
    cost = 0
    for k in range(N):
        cost += cp.quad_form(mu[k], Q_list[k]) + cp.quad_form(v[k], R_list[k])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Mean steering problem failed: {prob.status}")

    mu_traj = np.array([mu[k].value for k in range(N + 1)])
    v_traj = np.array([v[k].value for k in range(N)])
    cost_mean = prob.value

    return mu_traj, v_traj, cost_mean


def solve_covariance_steering_sdp(A_list, B_list, D_list, Sigma_i, Sigma_f,
                                  N, Q_list, R_list,
                                  terminal_ineq=False,
                                  state_chance_constraints=None,
                                  control_chance_constraints=None,
                                  mu_traj=None, v_traj=None,
                                  covariance_constraints=None,
                                  solver=None, verbose=False):
    """
    Solve the covariance steering problem via SDP (Problems 7, 8, or 17).

    Parameters
    ----------
    A_list : list of (n, n) arrays
    B_list : list of (n, p) arrays
    D_list : list of (n, q) arrays
    Sigma_i : (n, n) array
    Sigma_f : (n, n) array
    N : int
    Q_list : list of (n, n) arrays
    R_list : list of (p, p) arrays
    terminal_ineq : bool
        If True, use inequality terminal condition (Sigma_N <= Sigma_f).
    state_chance_constraints : list of dicts, optional
        Each dict has keys: 'ell', 'alpha', 'beta', 'k' (time steps).
        Linearized constraint: ell^T Sigma_k ell + alpha^T mu_k - beta <= 0.
    control_chance_constraints : list of dicts, optional
        Each dict has keys: 'e', 'alpha_u', 'beta_u', 'k' (time steps).
        Linearized constraint: e^T Y_k e + alpha_u^T v_k - beta_u <= 0.
    mu_traj : (N+1, n) array, optional
        Mean trajectory (needed for chance constraints).
    v_traj : (N, p) array, optional
        Feedforward trajectory (needed for control chance constraints).
    covariance_constraints : list of dicts, optional
        Each dict has 'k' (time steps), 'indices' (block indices),
        'bound' (upper bound matrix). Sigma_k[idx, idx] <= bound.

    Returns
    -------
    Sigma_traj : (N+1, n, n) array
    U_traj : (N, p, n) array
    Y_traj : (N, p, p) array
    K_traj : (N, p, n) array
    cost_cov : float
    """
    n = Sigma_i.shape[0]
    p = B_list[0].shape[1]

    # Decision variables
    Sigma = [cp.Variable((n, n), symmetric=True) for _ in range(N + 1)]
    U = [cp.Variable((p, n)) for _ in range(N)]
    Y = [cp.Variable((p, p), symmetric=True) for _ in range(N)]

    constraints = []

    # Initial condition
    constraints.append(Sigma[0] == Sigma_i)

    # Terminal condition
    if terminal_ineq:
        constraints.append(Sigma[N] << Sigma_f)  # Sigma_N <= Sigma_f (PSD)
    else:
        constraints.append(Sigma[N] == Sigma_f)

    # Covariance dynamics (Eq. 7c) and LMI relaxation (Eq. 7b) for each k
    for k in range(N):
        Ak = A_list[k]
        Bk = B_list[k]
        Dk = D_list[k]

        # Covariance propagation (Eq. 7c):
        # Sigma_{k+1} = A_k Sigma_k A_k^T + B_k U_k A_k^T + A_k U_k^T B_k^T
        #              + B_k Y_k B_k^T + D_k D_k^T
        Gk = (Ak @ Sigma[k] @ Ak.T
              + Bk @ U[k] @ Ak.T
              + Ak @ U[k].T @ Bk.T
              + Bk @ Y[k] @ Bk.T
              + Dk @ Dk.T)
        constraints.append(Gk == Sigma[k + 1])

        # LMI relaxation (Schur complement of C_k <= 0):
        # [Sigma_k  U_k^T]
        # [U_k      Y_k  ] >= 0
        LMI_k = cp.bmat([[Sigma[k], U[k].T],
                         [U[k], Y[k]]])
        constraints.append(LMI_k >> 0)

        # Sigma_k must be positive definite
        constraints.append(Sigma[k] >> 0)

    # Terminal Sigma positive definite
    constraints.append(Sigma[N] >> 0)

    # State chance constraints (linearized form, Eq. 16a)
    if state_chance_constraints is not None:
        for cc in state_chance_constraints:
            ell = cc['ell']
            alpha_x = cc['alpha']
            beta_x = cc['beta']
            time_steps = cc.get('k', range(N))
            for k in time_steps:
                # ell^T Sigma_k ell + alpha_x^T mu_k - beta_x <= 0
                lhs = ell @ Sigma[k] @ ell
                if mu_traj is not None:
                    lhs = lhs + alpha_x @ mu_traj[k] - beta_x
                else:
                    lhs = lhs - beta_x
                constraints.append(lhs <= 0)

    # Control chance constraints (linearized form, Eq. 16b)
    if control_chance_constraints is not None:
        for cc in control_chance_constraints:
            e = cc['e']
            alpha_u = cc['alpha_u']
            beta_u = cc['beta_u']
            time_steps = cc.get('k', range(N))
            for k in time_steps:
                # e^T Y_k e + alpha_u^T v_k - beta_u <= 0
                lhs = e @ Y[k] @ e
                if v_traj is not None:
                    lhs = lhs + alpha_u @ v_traj[k] - beta_u
                else:
                    lhs = lhs - beta_u
                constraints.append(lhs <= 0)

    # Additional covariance constraints (e.g., tube constraints for Example 2)
    if covariance_constraints is not None:
        for cc in covariance_constraints:
            time_steps = cc['k']
            indices = cc['indices']
            bound = cc['bound']
            for k in time_steps:
                # Extract submatrix and constrain
                sub = Sigma[k][np.ix_(indices, indices)]
                constraints.append(sub << bound)

    # Cost (Eq. 7a): sum tr(Q_k Sigma_k) + tr(R_k Y_k)
    cost = 0
    for k in range(N):
        cost += cp.trace(Q_list[k] @ Sigma[k]) + cp.trace(R_list[k] @ Y[k])

    prob = cp.Problem(cp.Minimize(cost), constraints)

    # Choose solver
    if solver is None:
        solver = cp.SCS if N > 100 else cp.CLARABEL

    solver_kwargs = {'verbose': verbose}
    if solver == cp.SCS:
        solver_kwargs['max_iters'] = 20000
        solver_kwargs['eps'] = 1e-6

    prob.solve(solver=solver, **solver_kwargs)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Covariance steering SDP failed: {prob.status}")

    # Extract results
    Sigma_traj = np.array([Sigma[k].value for k in range(N + 1)])
    U_traj = np.array([U[k].value for k in range(N)])
    Y_traj = np.array([Y[k].value for k in range(N)])

    # Recover feedback gains K_k = U_k Sigma_k^{-1}
    K_traj = np.zeros((N, p, n))
    for k in range(N):
        K_traj[k] = U_traj[k] @ np.linalg.inv(Sigma_traj[k])

    cost_cov = prob.value

    return Sigma_traj, U_traj, Y_traj, K_traj, cost_cov


def build_linearized_chance_constraints(alpha_x_list, beta_x_list,
                                        alpha_u_list, beta_u_list,
                                        Sigma_r, Y_r, epsilon_x, epsilon_u,
                                        N, mu_traj=None, v_traj=None,
                                        state_time_steps=None,
                                        control_time_steps=None):
    """
    Build linearized chance constraints as in Eq. (15a, 15b).

    The chance constraint P(alpha^T x <= beta) >= 1 - epsilon is linearized
    around reference values Sigma_r and Y_r using the tangent line
    overestimator of sqrt.

    Parameters
    ----------
    alpha_x_list : list of (n,) arrays
        State constraint directions.
    beta_x_list : list of floats
        State constraint bounds.
    alpha_u_list : list of (p,) arrays
        Control constraint directions.
    beta_u_list : list of floats
        Control constraint bounds.
    Sigma_r : (n, n) array
        Reference covariance for linearization.
    Y_r : (p, p) array
        Reference Y for linearization.
    epsilon_x : float
        State constraint violation probability.
    epsilon_u : float
        Control constraint violation probability.
    N : int
        Horizon.
    mu_traj : (N+1, n) array, optional
    v_traj : (N, p) array, optional
    state_time_steps : list, optional
        Time steps for state constraints (default: all).
    control_time_steps : list, optional
        Time steps for control constraints (default: all).

    Returns
    -------
    state_constraints : list of dicts
    control_constraints : list of dicts
    """
    phi_inv_x = norm.ppf(1 - epsilon_x)
    phi_inv_u = norm.ppf(1 - epsilon_u)

    if state_time_steps is None:
        state_time_steps = list(range(N))
    if control_time_steps is None:
        control_time_steps = list(range(N))

    state_constraints = []
    for alpha_x, beta_val in zip(alpha_x_list, beta_x_list):
        # Reference variance: alpha_x^T Sigma_r alpha_x
        var_ref = alpha_x @ Sigma_r @ alpha_x

        # Linearized sqrt: sqrt(x) <= x/(2*sqrt(x0)) + sqrt(x0)/2
        # So the constraint becomes:
        # phi_inv * [alpha^T Sigma alpha / (2*sqrt(var_ref)) + sqrt(var_ref)/2]
        #   + alpha^T mu - beta <= 0
        # Rewritten as:
        # phi_inv/(2*sqrt(var_ref)) * alpha^T Sigma alpha
        #   + alpha^T mu - beta + phi_inv*sqrt(var_ref)/2 <= 0
        # i.e., ell^T Sigma ell + alpha^T mu - beta_eff <= 0
        # where ell = sqrt(phi_inv/(2*sqrt(var_ref))) * alpha (for quadratic form)
        # Actually, let's express as:
        # coeff * alpha^T Sigma alpha + alpha^T mu - beta_eff <= 0
        # coeff = phi_inv / (2*sqrt(var_ref))
        # beta_eff = beta - phi_inv * sqrt(var_ref) / 2

        coeff = phi_inv_x / (2 * np.sqrt(var_ref))
        beta_eff = beta_val - phi_inv_x * np.sqrt(var_ref) / 2

        # ell such that ell^T Sigma ell = coeff * alpha^T Sigma alpha
        # ell = sqrt(coeff) * alpha
        ell = np.sqrt(coeff) * alpha_x

        state_constraints.append({
            'ell': ell,
            'alpha': alpha_x,
            'beta': beta_eff,
            'k': state_time_steps
        })

    control_constraints = []
    for alpha_u, beta_val in zip(alpha_u_list, beta_u_list):
        var_ref = alpha_u @ Y_r @ alpha_u

        coeff = phi_inv_u / (2 * np.sqrt(var_ref))
        beta_eff = beta_val - phi_inv_u * np.sqrt(var_ref) / 2

        e = np.sqrt(coeff) * alpha_u

        control_constraints.append({
            'e': e,
            'alpha_u': alpha_u,
            'beta_u': beta_eff,
            'k': control_time_steps
        })

    return state_constraints, control_constraints


def solve_full_cs_problem(A_list, B_list, D_list, Sigma_i, Sigma_f,
                          mu_i, mu_f, N, Q_list, R_list,
                          terminal_ineq=False,
                          waypoints=None,
                          alpha_x_list=None, beta_x_list=None,
                          alpha_u_list=None, beta_u_list=None,
                          Sigma_r=None, Y_r=None,
                          epsilon_x=0.1, epsilon_u=0.1,
                          state_time_steps=None,
                          control_time_steps=None,
                          covariance_constraints=None):
    """
    Solve the full covariance steering problem (mean + covariance).

    Returns
    -------
    results : dict with keys:
        'mu_traj', 'v_traj', 'Sigma_traj', 'U_traj', 'Y_traj', 'K_traj',
        'cost_mean', 'cost_cov', 'cost_total'
    """
    # Step 1: Solve mean steering
    mu_traj, v_traj, cost_mean = solve_mean_steering(
        A_list, B_list, mu_i, mu_f, N, Q_list, R_list, waypoints=waypoints
    )

    # Step 2: Build chance constraints if provided
    state_cc = None
    control_cc = None
    if alpha_x_list is not None and Sigma_r is not None:
        state_cc, control_cc = build_linearized_chance_constraints(
            alpha_x_list, beta_x_list, alpha_u_list, beta_u_list,
            Sigma_r, Y_r, epsilon_x, epsilon_u, N,
            mu_traj=mu_traj, v_traj=v_traj,
            state_time_steps=state_time_steps,
            control_time_steps=control_time_steps
        )

    # Step 3: Solve covariance steering SDP
    Sigma_traj, U_traj, Y_traj, K_traj, cost_cov = \
        solve_covariance_steering_sdp(
            A_list, B_list, D_list, Sigma_i, Sigma_f, N, Q_list, R_list,
            terminal_ineq=terminal_ineq,
            state_chance_constraints=state_cc,
            control_chance_constraints=control_cc,
            mu_traj=mu_traj, v_traj=v_traj,
            covariance_constraints=covariance_constraints
        )

    # Total cost includes mean cost contribution
    cost_total = cost_mean + cost_cov

    return {
        'mu_traj': mu_traj,
        'v_traj': v_traj,
        'Sigma_traj': Sigma_traj,
        'U_traj': U_traj,
        'Y_traj': Y_traj,
        'K_traj': K_traj,
        'cost_mean': cost_mean,
        'cost_cov': cost_cov,
        'cost_total': cost_total
    }
