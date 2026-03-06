"""
Nonlinear quadrotor dynamics and differential flatness utilities.

Implements the quadrotor model from Eq. (20) in the paper:
    ṙ = v
    v̇ = (1/m)(g e3 + R(q) ê3 τ + w_f)
    q̇ = S(q)(ω + w_m)

State: x = [r_x, r_y, r_z, v_x, v_y, v_z, ψ, φ, θ]^T  (9 states)
Input: u = [τ, ω_x, ω_y, ω_z]^T  (4 inputs)
Disturbance: w = [w_fx, w_fy, w_fz, w_mx, w_my, w_mz]^T  (6 channels)

Uses ZYX Euler angles: q = [ψ, φ, θ]^T (yaw, pitch, roll).
"""

import numpy as np
from scipy.spatial.transform import Rotation


# Physical constants
MASS = 1.0       # quadrotor mass [kg]
GRAVITY = 9.81   # gravitational acceleration [m/s^2]


def rotation_matrix_zyx(psi, phi, theta):
    """
    Compute rotation matrix for ZYX Euler angles.
    R = Rz(psi) * Ry(phi) * Rx(theta)
    """
    cpsi, spsi = np.cos(psi), np.sin(psi)
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)

    R = np.array([
        [cpsi*cphi, cpsi*sphi*sth - spsi*cth, cpsi*sphi*cth + spsi*sth],
        [spsi*cphi, spsi*sphi*sth + cpsi*cth, spsi*sphi*cth - cpsi*sth],
        [-sphi,     cphi*sth,                  cphi*cth]
    ])
    return R


def euler_rate_matrix(psi, phi, theta):
    """
    Compute S(q) matrix: q̇ = S(q) * ω

    Maps body angular rates ω to Euler angle rates q̇.
    Based on the inverse of the kinematic matrix E:
        ω = E(q) * q̇

    E = [0,   -sin(phi),            cos(phi)*cos(theta)]  -- actually let me compute this properly
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)

    # E matrix: ω = E * q̇  (body rates from Euler rates)
    # For ZYX convention: q = [ψ, φ, θ]^T
    # ω_x = -sin(φ)*ψ̇ + θ̇
    # ω_y = cos(φ)*sin(θ)*ψ̇ + cos(θ)*φ̇
    # ω_z = cos(φ)*cos(θ)*ψ̇ - sin(θ)*φ̇
    E = np.array([
        [-sphi,           0,     1],
        [cphi * sth,      cth,   0],
        [cphi * cth,      -sth,  0]
    ])

    # S = E^{-1}
    # det(E) = -sphi*(cth*0 - 0*(-sth)) - 0 + 1*(cphi*sth*(-sth) - cth*cphi*cth)
    # = 0 + 0 + (-cphi*sth^2 - cphi*cth^2) = -cphi
    det_E = -cphi
    if abs(det_E) < 1e-10:
        det_E = np.sign(det_E) * 1e-10 if det_E != 0 else 1e-10

    # Cofactor matrix and inverse
    S = np.zeros((3, 3))
    S[0, 0] = 0                    # cof(1,1)/det
    S[0, 1] = sth / cphi          # cof(1,2)/det
    S[0, 2] = cth / cphi          # cof(1,3)/det
    S[1, 0] = 0
    S[1, 1] = cth
    S[1, 2] = -sth
    S[2, 0] = 1
    S[2, 1] = sth * sphi / cphi
    S[2, 2] = cth * sphi / cphi

    return S


def quadrotor_continuous_dynamics(x, u, w=None):
    """
    Compute continuous-time quadrotor dynamics: ẋ = f(x, u, w).

    Parameters
    ----------
    x : (9,) array - [r_x, r_y, r_z, v_x, v_y, v_z, ψ, φ, θ]
    u : (4,) array - [τ, ω_x, ω_y, ω_z]
    w : (6,) array - [w_fx, w_fy, w_fz, w_mx, w_my, w_mz], optional

    Returns
    -------
    xdot : (9,) array
    """
    if w is None:
        w = np.zeros(6)

    r = x[:3]
    v = x[3:6]
    psi, phi, theta = x[6], x[7], x[8]

    tau = u[0]
    omega = u[1:4]

    w_f = w[:3]
    w_m = w[3:6]

    e3 = np.array([0, 0, 1.0])

    # Rotation matrix
    R = rotation_matrix_zyx(psi, phi, theta)

    # S matrix (Euler rate from body rates)
    S = euler_rate_matrix(psi, phi, theta)

    # Dynamics
    rdot = v
    vdot = (1.0 / MASS) * (-MASS * GRAVITY * e3 + R @ e3 * tau + w_f)
    qdot = S @ (omega + w_m)

    return np.concatenate([rdot, vdot, qdot])


def quadrotor_discrete_dynamics(x, u, w, dt):
    """
    Discrete-time dynamics using first-order Euler discretization (Eq. 21).
    x_{k+1} = x_k + dt * f(x_k, u_k, w_k)
    """
    return x + dt * quadrotor_continuous_dynamics(x, u, w)


def compute_jacobians_numerical(x_nom, u_nom, dt, eps=1e-6):
    """
    Compute Jacobian matrices A_k, B_k, D_k numerically via finite differences.

    For F(x, u, w) = x + dt * f(x, u, w):
    A_k = dF/dx, B_k = dF/du, D_k = dF/dw
    """
    n = 9
    p = 4
    q = 6
    w_nom = np.zeros(q)

    F_nom = quadrotor_discrete_dynamics(x_nom, u_nom, w_nom, dt)

    # A = dF/dx
    A = np.zeros((n, n))
    for i in range(n):
        x_plus = x_nom.copy()
        x_plus[i] += eps
        x_minus = x_nom.copy()
        x_minus[i] -= eps
        F_plus = quadrotor_discrete_dynamics(x_plus, u_nom, w_nom, dt)
        F_minus = quadrotor_discrete_dynamics(x_minus, u_nom, w_nom, dt)
        A[:, i] = (F_plus - F_minus) / (2 * eps)

    # B = dF/du
    B = np.zeros((n, p))
    for i in range(p):
        u_plus = u_nom.copy()
        u_plus[i] += eps
        u_minus = u_nom.copy()
        u_minus[i] -= eps
        F_plus = quadrotor_discrete_dynamics(x_nom, u_plus, w_nom, dt)
        F_minus = quadrotor_discrete_dynamics(x_nom, u_minus, w_nom, dt)
        B[:, i] = (F_plus - F_minus) / (2 * eps)

    # D = dF/dw
    D = np.zeros((n, q))
    for i in range(q):
        w_plus = w_nom.copy()
        w_plus[i] += eps
        w_minus = w_nom.copy()
        w_minus[i] -= eps
        F_plus = quadrotor_discrete_dynamics(x_nom, u_nom, w_plus, dt)
        F_minus = quadrotor_discrete_dynamics(x_nom, u_nom, w_minus, dt)
        D[:, i] = (F_plus - F_minus) / (2 * eps)

    return A, B, D


def compute_nominal_quadrotor_state_and_control(pos_traj, dt):
    """
    Compute nominal quadrotor state and control from position trajectory
    using differential flatness (with ψ = 0 throughout).

    Parameters
    ----------
    pos_traj : (N+1, 3) array
        Nominal position trajectory [r_x, r_y, r_z] at each time step.
    dt : float
        Time step.

    Returns
    -------
    x_nom : (N+1, 9) array
        Nominal state trajectory.
    u_nom : (N, 4) array
        Nominal control trajectory.
    """
    N = pos_traj.shape[0] - 1
    e3 = np.array([0, 0, 1.0])

    # Compute velocities via finite differences
    vel_traj = np.zeros_like(pos_traj)
    for k in range(N):
        vel_traj[k] = (pos_traj[k + 1] - pos_traj[k]) / dt
    vel_traj[N] = vel_traj[N - 1]

    # Compute accelerations via finite differences
    acc_traj = np.zeros_like(pos_traj)
    for k in range(N):
        acc_traj[k] = (vel_traj[min(k + 1, N - 1)] - vel_traj[k]) / dt
    acc_traj[N] = acc_traj[N - 1]

    # Compute nominal states and controls
    x_nom = np.zeros((N + 1, 9))
    u_nom = np.zeros((N, 4))

    for k in range(N + 1):
        r = pos_traj[k]
        v = vel_traj[k]
        a = acc_traj[k]

        # Thrust vector: t_vec = m * (a + g*e3)
        t_vec = MASS * (a + GRAVITY * e3)
        tau = np.linalg.norm(t_vec)

        if tau < 1e-10:
            tau = MASS * GRAVITY
            t_hat = e3
        else:
            t_hat = t_vec / tau

        # With ψ = 0, compute φ and θ from thrust direction
        # R(0, φ, θ) * e3 = t_hat
        # [-sin(φ)*cos(θ), ... no wait
        # R * e3 = [cos(ψ)*sin(φ)*cos(θ) + sin(ψ)*sin(θ),
        #           sin(ψ)*sin(φ)*cos(θ) - cos(ψ)*sin(θ),
        #           cos(φ)*cos(θ)]
        # With ψ = 0:
        # R * e3 = [sin(φ)*cos(θ), -sin(θ), cos(φ)*cos(θ)]
        # So: t_hat_y = -sin(θ) => θ = -arcsin(t_hat_y)
        #     t_hat_x / t_hat_z = sin(φ)/cos(φ) = tan(φ) => φ = arctan2(t_hat_x, t_hat_z)

        theta = -np.arcsin(np.clip(t_hat[1], -1, 1))
        phi = np.arctan2(t_hat[0], t_hat[2])

        x_nom[k] = np.array([r[0], r[1], r[2], v[0], v[1], v[2], 0, phi, theta])

    # Compute control inputs
    for k in range(N):
        a = acc_traj[k]
        t_vec = MASS * (a + GRAVITY * e3)
        tau = np.linalg.norm(t_vec)

        # Angular velocity from Euler angle rate
        psi_k = 0
        phi_k = x_nom[k, 7]
        theta_k = x_nom[k, 8]

        if k < N - 1:
            phi_next = x_nom[k + 1, 7]
            theta_next = x_nom[k + 1, 8]
            phi_dot = (phi_next - phi_k) / dt
            theta_dot = (theta_next - theta_k) / dt
        elif k > 0:
            # Use backward difference at last step to avoid spike
            phi_prev = x_nom[k - 1, 7]
            theta_prev = x_nom[k - 1, 8]
            phi_dot = (phi_k - phi_prev) / dt
            theta_dot = (theta_k - theta_prev) / dt
        else:
            phi_dot = 0
            theta_dot = 0
        psi_dot = 0

        qdot = np.array([psi_dot, phi_dot, theta_dot])

        # ω = E(q) * q̇
        cphi, sphi = np.cos(phi_k), np.sin(phi_k)
        cth, sth = np.cos(theta_k), np.sin(theta_k)

        E = np.array([
            [-sphi,           0,     1],
            [cphi * sth,      cth,   0],
            [cphi * cth,      -sth,  0]
        ])
        omega = E @ qdot

        u_nom[k] = np.array([tau, omega[0], omega[1], omega[2]])

    return x_nom, u_nom


def generate_nominal_trajectory_triple_integrator(pos_0, pos_N, waypoints,
                                                  dt, N):
    """
    Generate a smooth nominal trajectory using a triple integrator model
    (minimum jerk path).

    Parameters
    ----------
    pos_0 : (3,) array
        Initial position.
    pos_N : (3,) array
        Final position.
    waypoints : dict
        {time_step: (3,) array of positions}
    dt : float
        Time step.
    N : int
        Number of steps.

    Returns
    -------
    pos_traj : (N+1, 3) array
    """
    import cvxpy as cp

    # State: [pos(3), vel(3), acc(3)] (9 states per axis, but we solve 3D)
    # Using triple integrator: pos_{k+1} = pos_k + dt*vel_k
    #                          vel_{k+1} = vel_k + dt*acc_k
    #                          acc_{k+1} = acc_k + u_k

    n_trip = 9  # triple integrator state dim (3D)
    p_trip = 3  # input dim (3D jerk)

    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    A_trip = np.block([
        [I3, dt * I3, Z3],
        [Z3, I3, dt * I3],
        [Z3, Z3, I3]
    ])
    B_trip = np.block([[Z3], [Z3], [I3]])

    # Decision variables
    x = [cp.Variable(n_trip) for _ in range(N + 1)]
    u = [cp.Variable(p_trip) for _ in range(N)]

    constraints = []
    # Initial conditions: position, zero velocity, zero acceleration
    constraints.append(x[0][:3] == pos_0)
    constraints.append(x[0][3:6] == np.zeros(3))
    constraints.append(x[0][6:9] == np.zeros(3))

    # Terminal conditions
    constraints.append(x[N][:3] == pos_N)
    constraints.append(x[N][3:6] == np.zeros(3))
    constraints.append(x[N][6:9] == np.zeros(3))

    # Dynamics
    for k in range(N):
        constraints.append(x[k + 1] == A_trip @ x[k] + B_trip @ u[k])

    # Waypoint constraints (position only)
    for k_wp, pos_wp in waypoints.items():
        constraints.append(x[k_wp][:3] == pos_wp)

    # Minimum jerk cost: minimize sum of ||u_k||^2 = sum of ||jerk||^2
    cost = sum(cp.sum_squares(u[k]) for k in range(N))

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Nominal trajectory generation failed: {prob.status}")

    pos_traj = np.array([x[k].value[:3] for k in range(N + 1)])
    return pos_traj
