"""
Replication of:
"Nonlinear Uncertainty Control with Iterative Covariance Steering"
by Jack Ridderhof, Kazuhide Okamoto, and Panagiotis Tsiotras (2019)
arXiv:1903.10919v2

Implements the iCS algorithm (Procedure 1) for a double integrator
with quadratic drag subject to additive Brownian noise.
"""

import numpy as np
from scipy.linalg import expm, cholesky, block_diag
from scipy.stats import norm as normal_dist, chi2
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('Agg')

# ============================================================
# Problem parameters (Section V)
# ============================================================
nx = 4   # state dimension: x = (xi1, xi2, v1, v2)
nu = 2   # control dimension
nw = 2   # noise dimension

cd_drag = 0.005    # drag coefficient
gamma = 0.01       # noise scale
sigma = 15.0       # time dilation tf - t0
N = 25             # number of discrete steps
dtau = 1.0 / N     # normalized time step
wxf = 1000.0       # terminal mean error weight

# Boundary conditions (eqs 72-73)
x_bar_0 = np.array([1.0, 8.0, 2.0, 0.0])
Px0 = 0.01 * np.eye(nx)
x_bar_f = np.array([1.0, 2.0, -1.0, 0.0])
Pxf = 0.1 * np.eye(nx)

# Cost function weights (eq 74)
ell_weight = 10.0     # l(x,u) = 10*||u||^2
Qx_diag = 5.0         # Qx = 5*I
Qu_diag = 1.0          # Qu = I

# Chance constraint (eq 71): P(|e1^T xi| <= 6) >= 0.9
cc_limit = 6.0
cc_prob_total = 0.1
cc_prob_half = cc_prob_total / 2  # split for two halfspaces

# Initial guess (eq 75)
u_init = np.array([-0.3, -0.1])

# Continuous dynamics matrices (eq 69)
A_cont = np.zeros((nx, nx))
A_cont[0, 2] = 1.0  # dxi1/dt = v1
A_cont[1, 3] = 1.0  # dxi2/dt = v2

B_cont = np.zeros((nx, nu))
B_cont[2, 0] = 1.0
B_cont[3, 1] = 1.0

G_cont = np.zeros((nx, nw))
G_cont[2, 0] = gamma
G_cont[3, 1] = gamma


# ============================================================
# Dynamics
# ============================================================
def nonlinear_f(x, u):
    """Continuous-time nonlinear dynamics f(x, u) from eqs (66-67)."""
    v = x[2:]
    v_norm = np.linalg.norm(v)
    dxi = v.copy()
    dv = u - cd_drag * v_norm * v
    return np.concatenate([dxi, dv])


def linearize_normalized(x_hat, u_hat):
    """
    Linearize the time-normalized system about (x_hat, u_hat).
    Returns A_tau, B_tau, r_tau (eqs 10-11, 70).
    """
    v_hat = x_hat[2:]
    v_norm = np.linalg.norm(v_hat)

    # Eq (70): A_tau = sigma*(A - cd*E2*(v*v^T/||v|| + ||v||*I2)*E2^T)
    A_tau = sigma * A_cont.copy()
    if v_norm > 1e-12:
        drag_jac = cd_drag * (np.outer(v_hat, v_hat) / v_norm
                              + v_norm * np.eye(2))
        A_tau[2:, 2:] -= sigma * drag_jac

    B_tau = sigma * B_cont  # B_tau = sigma*B (constant)

    # Eq (11): r_tau = sigma*f(x_hat, u_hat) - A_tau*x_hat - B_tau*u_hat
    f_val = nonlinear_f(x_hat, u_hat)
    r_tau = sigma * f_val - A_tau @ x_hat - B_tau @ u_hat

    return A_tau, B_tau, r_tau


def discretize_exact(A_tau, B_tau, r_tau):
    """
    Exact ZOH discretization using matrix exponential (eqs 17a-c, 18).
    Returns A_d, B_d, r_d, and noise covariance Sigma_noise.

    The noise covariance accounts for the sqrt(sigma)*G factor in eq (7).
    Sigma_noise = sigma * integral Phi(s) G G^T Phi(s)^T ds
    """
    # State transition matrix: Phi(tau_{k+1}, tau_k) = expm(A_tau * dtau)
    A_d = expm(A_tau * dtau)

    # Augmented matrix for B_d and r_d integrals
    n_aug = nx + nu + 1
    M = np.zeros((n_aug, n_aug))
    M[:nx, :nx] = A_tau * dtau
    M[:nx, nx:nx+nu] = B_tau * dtau
    M[:nx, nx+nu:] = r_tau.reshape(-1, 1) * dtau
    eM = expm(M)
    B_d = eM[:nx, nx:nx+nu]
    r_d = eM[:nx, nx+nu].flatten()

    # Noise covariance via Van Loan's method
    # For normalized system: diffusion = sqrt(sigma)*G_cont
    # So GGT = sigma * G_cont @ G_cont^T
    GGT = sigma * (G_cont @ G_cont.T)

    M2 = np.zeros((2*nx, 2*nx))
    M2[:nx, :nx] = -A_tau * dtau
    M2[:nx, nx:] = GGT * dtau
    M2[nx:, nx:] = A_tau.T * dtau
    eM2 = expm(M2)
    Sigma_noise = eM2[nx:, nx:].T @ eM2[:nx, nx:]
    Sigma_noise = (Sigma_noise + Sigma_noise.T) / 2

    return A_d, B_d, r_d, Sigma_noise


def build_concatenated_system(x_ref, u_ref):
    """
    Build concatenated system matrices (eq 29).
    The Y process covariance is:
      Py = Ai @ Px0 @ Ai^T + sum of propagated noise covariances

    Returns per-step matrices and concatenated system.
    """
    Ad_list = []
    Bd_list = []
    rd_list = []
    Sig_list = []  # noise covariance per step

    for k in range(N):
        A_tau, B_tau, r_tau = linearize_normalized(x_ref[k], u_ref[k])
        A_d, B_d, r_d, Sig = discretize_exact(A_tau, B_tau, r_tau)
        Ad_list.append(A_d)
        Bd_list.append(B_d)
        rd_list.append(r_d)
        Sig_list.append(Sig)

    dim_X = (N + 1) * nx
    dim_U = N * nu

    # Build state transition from step 0 to each step k
    # Phi_{k,0} = A_{k-1} * A_{k-2} * ... * A_0
    Phi_list = [np.eye(nx)]  # Phi_{0,0} = I
    for k in range(N):
        Phi_list.append(Ad_list[k] @ Phi_list[k])

    # Ai_mat: (N+1)*nx x nx, maps x0 to X
    Ai_mat = np.zeros((dim_X, nx))
    for k in range(N + 1):
        Ai_mat[k*nx:(k+1)*nx, :] = Phi_list[k]

    # Build transition Phi_{k, j+1} for j < k
    def phi_from_to(k, j_start):
        """Compute Phi from step j_start to step k: A_{k-1}...A_{j_start}"""
        P = np.eye(nx)
        for l in range(j_start, k):
            P = Ad_list[l] @ P
        return P

    # Bi_mat: (N+1)*nx x N*nu
    Bi_mat = np.zeros((dim_X, dim_U))
    for k in range(1, N + 1):
        for j in range(k):
            Phi_kj1 = phi_from_to(k, j + 1)
            Bi_mat[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = Phi_kj1 @ Bd_list[j]

    # Ri_vec: (N+1)*nx
    Ri_vec = np.zeros(dim_X)
    for k in range(1, N + 1):
        for j in range(k):
            Phi_kj1 = phi_from_to(k, j + 1)
            Ri_vec[k*nx:(k+1)*nx] += Phi_kj1 @ rd_list[j]

    # Build Py directly (more efficient than constructing Gi_mat)
    # Py_{k,l} = Phi_{k,0} Px0 Phi_{l,0}^T
    #          + sum_{j=0}^{min(k,l)-1} Phi_{k,j+1} Sig_j Phi_{l,j+1}^T
    Py = np.zeros((dim_X, dim_X))
    for k in range(N + 1):
        for l in range(k, N + 1):
            block = Phi_list[k] @ Px0 @ Phi_list[l].T
            for j in range(min(k, l)):
                Phi_kj1 = phi_from_to(k, j + 1)
                Phi_lj1 = phi_from_to(l, j + 1)
                block += Phi_kj1 @ Sig_list[j] @ Phi_lj1.T
            Py[k*nx:(k+1)*nx, l*nx:(l+1)*nx] = block
            if k != l:
                Py[l*nx:(l+1)*nx, k*nx:(k+1)*nx] = block.T
    Py = (Py + Py.T) / 2

    return (Ad_list, Bd_list, rd_list, Sig_list,
            Ai_mat, Bi_mat, Ri_vec, Py)


def propagate_nonlinear_mean(v_seq, x0):
    """
    Propagate nonlinear mean dynamics (Remark 3, eq 65):
    x_bar_dot = sigma * f(x_bar, v)
    over normalized time [0, 1] using RK4.
    """
    x_bar = np.zeros((N + 1, nx))
    x_bar[0] = x0.copy()

    for k in range(N):
        n_sub = 20
        h = dtau / n_sub
        xc = x_bar[k].copy()
        for _ in range(n_sub):
            k1 = sigma * nonlinear_f(xc, v_seq[k])
            k2 = sigma * nonlinear_f(xc + 0.5*h*k1, v_seq[k])
            k3 = sigma * nonlinear_f(xc + 0.5*h*k2, v_seq[k])
            k4 = sigma * nonlinear_f(xc + h*k3, v_seq[k])
            xc += (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x_bar[k + 1] = xc

    return x_bar


def solve_cs_subproblem(Ai_mat, Bi_mat, Ri_vec, Py, Py_sqrt,
                         x_ref, u_ref, iteration,
                         use_hard_terminal=True, cc_relax_factor=1.0):
    """
    Solve Problem 4: iCS Convex Subproblem (eq 64 + constraints).
    """
    dim_X = (N + 1) * nx
    dim_U = N * nu

    # Decision variables
    V = cp.Variable(dim_U)
    K_vars = [cp.Variable((nu, nx)) for _ in range(N)]
    eta_xf = cp.Variable(nonneg=True)

    # Precompute block structure
    L = Py_sqrt  # lower-triangular Cholesky of Py
    L_rows = [L[k*nx:(k+1)*nx, :] for k in range(N + 1)]

    Bi_blk = np.zeros((N + 1, N, nx, nu))
    for k in range(N + 1):
        for j in range(N):
            Bi_blk[k, j] = Bi_mat[k*nx:(k+1)*nx, j*nu:(j+1)*nu]

    # Mean trajectory: X_bar = Ai*x0 + Bi*V + Ri
    X_bar = Ai_mat @ x_bar_0 + Bi_mat @ V + Ri_vec

    # ---------------------------------------------------------------
    # Cost (eqs 47, 48, 64)
    # ---------------------------------------------------------------
    # L(V) = sum 10*||vk||^2
    cost_ell = ell_weight * cp.sum_squares(V)

    # Covariance cost terms via Frobenius norms
    Qx_s = np.sqrt(Qx_diag)  # sqrt(5) for scaling inside sum_squares
    Qu_s = np.sqrt(Qu_diag)  # sqrt(1) = 1

    cov_terms = []
    for k in range(N):
        # M_k = (I + Bi@K) block row k applied to Py^{1/2}
        Mk = L_rows[k]
        for j in range(N):
            B_kj = Bi_blk[k, j]
            if np.any(np.abs(B_kj) > 1e-15):
                Mk = Mk + B_kj @ K_vars[j] @ L_rows[j]
        # tr(Qx_k^{1/2} Mk Mk^T Qx_k^{1/2}) = ||Qx_s * Mk||_F^2
        cov_terms.append(cp.sum_squares(Qx_s * Mk))

    for k in range(N):
        cov_terms.append(cp.sum_squares(Qu_s * K_vars[k] @ L_rows[k]))

    cost_trace = sum(cov_terms)
    cost_main = (sigma / N) * (cost_ell + cost_trace)
    cost_total = cost_main + wxf * eta_xf

    # ---------------------------------------------------------------
    # Constraints
    # ---------------------------------------------------------------
    constraints = []

    # Terminal mean (eq 49 or relaxed eq 63)
    EN = np.zeros((nx, dim_X))
    EN[:, N*nx:(N+1)*nx] = np.eye(nx)
    terminal_mean = EN @ X_bar

    if use_hard_terminal:
        constraints.append(terminal_mean == x_bar_f)
        constraints.append(eta_xf == 0)
    else:
        constraints.append(cp.norm(terminal_mean - x_bar_f, 'inf') <= eta_xf)

    # Terminal covariance (eq 51):
    # ||Py^{1/2} (I+Bi@K)^T EN^T Pxf^{-1/2}||_2 <= 1
    Pxf_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Pxf)))

    # Helper: compute (I+Bi@K)^T @ Ek^T @ vec for a given k and vec
    def compute_IpBK_T_Ek_T_vec(k, vec):
        """Returns (N+1)*nx CVXPY expression for (I+Bi@K)^T @ Ek^T @ vec."""
        blocks = []
        for j in range(N):
            B_kj = Bi_blk[k, j]
            if j == k and k < N:
                blocks.append(vec + K_vars[j].T @ (B_kj.T @ vec))
            elif np.any(np.abs(B_kj) > 1e-15):
                blocks.append(K_vars[j].T @ (B_kj.T @ vec))
            else:
                blocks.append(np.zeros(nx))
        # Block N
        blocks.append(vec if k == N else np.zeros(nx))
        return cp.hstack(blocks)

    # Build M_cov column by column
    M_cov_cols = []
    for c in range(nx):
        e_c = Pxf_inv_sqrt[:, c]
        vec_full = compute_IpBK_T_Ek_T_vec(N, e_c)
        M_cov_cols.append(L @ vec_full)

    M_cov = cp.reshape(cp.hstack(M_cov_cols), (dim_X, nx), order='F')
    constraints.append(cp.norm(M_cov, 2) <= 1)

    # Chance constraints (eq 61): P(|xi1| <= 6) >= 0.9
    cc_vectors = [
        np.array([1.0, 0.0, 0.0, 0.0]),   # xi1 <= 6
        np.array([-1.0, 0.0, 0.0, 0.0]),   # -xi1 <= 6
    ]
    cc_prob_use = min(cc_prob_half * cc_relax_factor, 0.499)
    inv_cdf_cc = normal_dist.ppf(1.0 - cc_prob_use)

    for k in range(1, N):
        for am in cc_vectors:
            mean_part = am @ X_bar[k*nx:(k+1)*nx] - cc_limit
            vec_full = compute_IpBK_T_Ek_T_vec(k, am)
            std_val = cp.norm(L @ vec_full, 2)
            constraints.append(mean_part + inv_cdf_cc * std_val <= 0)

    # Trust region (eq 62)
    Delta_x = 5.0
    Delta_u = 3.0
    p_tr = 0.1
    inv_cdf_tr_x = normal_dist.ppf(1.0 - p_tr / (2 * nx))
    inv_cdf_tr_u = normal_dist.ppf(1.0 - p_tr / (2 * nu))

    for k in range(1, N):
        for j_comp in range(nx):
            ej = np.zeros(nx)
            ej[j_comp] = 1.0
            x_bar_k = X_bar[k*nx:(k+1)*nx]
            mean_dev = x_ref[k, j_comp] - x_bar_k[j_comp]

            vec_full = compute_IpBK_T_Ek_T_vec(k, ej)
            std_val = cp.norm(L @ vec_full, 2)

            constraints.append(mean_dev + inv_cdf_tr_x * std_val <= Delta_x)
            constraints.append(-mean_dev + inv_cdf_tr_x * std_val <= Delta_x)

    for k in range(N):
        for j_comp in range(nu):
            ej_u = np.zeros(nu)
            ej_u[j_comp] = 1.0
            v_k = V[k*nu:(k+1)*nu]
            mean_dev_u = u_ref[k, j_comp] - v_k[j_comp]
            Kk_T_ej = K_vars[k].T @ ej_u
            std_val_u = cp.norm(L[k*nx:(k+1)*nx, :].T @ Kk_T_ej, 2)
            constraints.append(mean_dev_u + inv_cdf_tr_u * std_val_u <= Delta_u)
            constraints.append(-mean_dev_u + inv_cdf_tr_u * std_val_u <= Delta_u)

    # Solve - prefer CLARABEL (interior point) for better accuracy
    problem = cp.Problem(cp.Minimize(cost_total), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except (cp.SolverError, Exception):
        problem.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-7)

    if problem.status in ['infeasible', 'infeasible_inaccurate']:
        print(f"  Problem status: {problem.status}")
        return None, None, None, None, problem.status

    V_opt = V.value
    K_opt = [Kk.value for Kk in K_vars]
    eta_opt = eta_xf.value if eta_xf.value is not None else 0.0

    return V_opt, K_opt, eta_opt, problem.value, problem.status


def run_ics(imax=10, tol=1e-3):
    """Run the iCS algorithm (Procedure 1)."""
    u_ref = np.tile(u_init, (N, 1))
    K_ref = [np.zeros((nu, nx)) for _ in range(N)]

    history = {
        'x_ref': [], 'u_ref': [], 'x_bar': [], 'u_bar': [],
        'K_list': [], 'V_opt': [], 'Py': [],
        'Ai_mat': [], 'Bi_mat': [], 'Ri_vec': [],
        'Ad_list': [], 'Bd_list': [], 'Sig_list': [],
    }

    for i in range(1, imax + 1):
        print(f"\n=== iCS Iteration {i} ===")

        # Step 2: Propagate nonlinear mean dynamics
        x_ref = propagate_nonlinear_mean(u_ref, x_bar_0)
        print(f"  x_ref[-1] = {x_ref[-1]}")

        # Steps 3-5: Linearize, discretize, build system
        (Ad_list, Bd_list, rd_list, Sig_list,
         Ai_mat, Bi_mat, Ri_vec, Py) = build_concatenated_system(x_ref, u_ref)

        # Cholesky of Py
        eigvals = np.linalg.eigvalsh(Py)
        reg = max(0, -eigvals.min() + 1e-10)
        Py_sqrt = cholesky(Py + reg * np.eye(Py.shape[0]), lower=True)

        history['x_ref'].append(x_ref.copy())
        history['u_ref'].append(u_ref.copy())

        # Step 6: Solve convex subproblem
        if i == 1:
            cc_relax, use_hard = 5.0, False
        elif i == 2:
            cc_relax, use_hard = 3.0, False
        elif i == 3:
            cc_relax, use_hard = 1.5, True
        else:
            cc_relax, use_hard = 1.0, True

        print(f"  Solving (cc_relax={cc_relax}, hard_term={use_hard})...")
        V_opt, K_opt, eta_opt, cost_val, status = solve_cs_subproblem(
            Ai_mat, Bi_mat, Ri_vec, Py, Py_sqrt,
            x_ref, u_ref, i,
            use_hard_terminal=use_hard, cc_relax_factor=cc_relax)

        if V_opt is None:
            print(f"  Solver failed: {status}")
            break

        print(f"  Status: {status}, Cost: {cost_val:.4f}, eta: {eta_opt:.6f}")

        v_opt = V_opt.reshape(N, nu)
        x_bar_opt = (Ai_mat @ x_bar_0 + Bi_mat @ V_opt + Ri_vec).reshape(N+1, nx)

        history['x_bar'].append(x_bar_opt.copy())
        history['u_bar'].append(v_opt.copy())
        history['K_list'].append([K.copy() for K in K_opt])
        history['V_opt'].append(V_opt.copy())
        history['Py'].append(Py.copy())
        history['Ai_mat'].append(Ai_mat.copy())
        history['Bi_mat'].append(Bi_mat.copy())
        history['Ri_vec'].append(Ri_vec.copy())
        history['Ad_list'].append(Ad_list)
        history['Bd_list'].append(Bd_list)
        history['Sig_list'].append(Sig_list)

        # Check convergence
        max_diff = np.max(np.abs(v_opt - u_ref))
        print(f"  Max control diff: {max_diff:.6f}")

        if max_diff <= tol and i > 1:
            print(f"\n  *** Converged in {i} iterations ***")
            break

        u_ref = v_opt.copy()
        K_ref = [K.copy() for K in K_opt]

    return history


def monte_carlo_simulation(history, n_trials=5000):
    """
    Run Monte Carlo simulation with the final converged solution.

    Uses full state feedback: uk = vk + Kk*(xk - x_bar_k)
    This is the standard practical implementation where the measured
    state deviation from the reference mean is used for feedback.

    Open-loop: uk = vk (no feedback)
    """
    idx = len(history['V_opt']) - 1
    V_opt = history['V_opt'][idx]
    K_list = history['K_list'][idx]
    v_opt = V_opt.reshape(N, nu)

    # Pre-compute reference mean trajectory (nonlinear propagation)
    x_bar_ref = propagate_nonlinear_mean(v_opt, x_bar_0)

    np.random.seed(42)

    x_cl = np.zeros((n_trials, N + 1, nx))
    u_cl = np.zeros((n_trials, N, nu))
    x_ol = np.zeros((n_trials, N + 1, nx))

    n_sub = 20

    for trial in range(n_trials):
        x0 = np.random.multivariate_normal(x_bar_0, Px0)
        x_cl[trial, 0] = x0.copy()
        x_ol[trial, 0] = x0.copy()

        for k in range(N):
            # Brownian noise increment
            dw = np.random.randn(nw)
            noise = np.sqrt(sigma * dtau) * G_cont @ dw

            h = dtau / n_sub

            # --- Closed-loop (full state feedback) ---
            # uk = vk + Kk * (xk - x_bar_k)
            deviation = x_cl[trial, k] - x_bar_ref[k]
            u_k = v_opt[k] + K_list[k] @ deviation
            u_cl[trial, k] = u_k

            # Propagate nonlinear dynamics with RK4 + noise
            xc = x_cl[trial, k].copy()
            for _ in range(n_sub):
                f1 = sigma * nonlinear_f(xc, u_k)
                f2 = sigma * nonlinear_f(xc + 0.5*h*f1, u_k)
                f3 = sigma * nonlinear_f(xc + 0.5*h*f2, u_k)
                f4 = sigma * nonlinear_f(xc + h*f3, u_k)
                xc += (h/6) * (f1 + 2*f2 + 2*f3 + f4)
            x_cl[trial, k+1] = xc + noise

            # --- Open-loop ---
            xc_ol = x_ol[trial, k].copy()
            for _ in range(n_sub):
                f1 = sigma * nonlinear_f(xc_ol, v_opt[k])
                f2 = sigma * nonlinear_f(xc_ol + 0.5*h*f1, v_opt[k])
                f3 = sigma * nonlinear_f(xc_ol + 0.5*h*f2, v_opt[k])
                f4 = sigma * nonlinear_f(xc_ol + h*f3, v_opt[k])
                xc_ol += (h/6) * (f1 + 2*f2 + 2*f3 + f4)
            x_ol[trial, k+1] = xc_ol + noise

    return x_cl, u_cl, x_ol, x_bar_ref


def confidence_ellipse_params(cov_2d, confidence=0.9):
    """Compute 90% confidence ellipse parameters from 2x2 covariance."""
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    eigvals = np.maximum(eigvals, 0)
    chi2_val = chi2.ppf(confidence, df=2)
    width = 2 * np.sqrt(chi2_val * eigvals[1])
    height = 2 * np.sqrt(chi2_val * eigvals[0])
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return width, height, angle


def plot_figure1(history):
    """
    Replicate Figure 1: State and control during successive solutions.
    Dashed = reference x_hat^i and u_hat^i.
    Solid = mean state and control after ith step, x_bar^i and u_bar^i.
    First iteration in blue, final iteration in bold.
    """
    n_iters = len(history['x_bar'])
    time_s = np.linspace(0, sigma, N + 1)
    time_u = time_s[:-1]

    # Color scheme matching paper: blue -> green -> yellow -> orange -> red
    if n_iters <= 1:
        iter_colors = ['tab:blue']
    else:
        cmap = matplotlib.colormaps.get_cmap('jet').resampled(n_iters)
        iter_colors = [cmap(i / (n_iters - 1)) for i in range(n_iters)]
        iter_colors[0] = (0.0, 0.0, 0.8, 1.0)  # blue first

    fig, axes = plt.subplots(3, 2, figsize=(10, 7.5))
    state_labels = [r'$\xi_1$', r'$\xi_2$', r'$v_1$', r'$v_2$']
    control_labels = [r'$u_1$', r'$u_2$']

    for idx in range(4):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        for i in range(n_iters):
            lw_solid = 2.5 if i == n_iters - 1 else 1.0
            lw_dash = 1.5 if i == n_iters - 1 else 0.7
            # Dashed: reference trajectory x_hat^i
            ax.plot(time_s, history['x_ref'][i][:, idx],
                    '--', color=iter_colors[i], linewidth=lw_dash)
            # Solid: optimized mean x_bar^i
            ax.plot(time_s, history['x_bar'][i][:, idx],
                    '-', color=iter_colors[i], linewidth=lw_solid)
        ax.set_ylabel(state_labels[idx], fontsize=13)
        ax.tick_params(labelsize=10)

    for idx in range(2):
        ax = axes[2, idx]
        for i in range(n_iters):
            lw_solid = 2.5 if i == n_iters - 1 else 1.0
            lw_dash = 1.5 if i == n_iters - 1 else 0.7
            ax.plot(time_u, history['u_ref'][i][:, idx],
                    '--', color=iter_colors[i], linewidth=lw_dash)
            ax.plot(time_u, history['u_bar'][i][:, idx],
                    '-', color=iter_colors[i], linewidth=lw_solid)
        ax.set_ylabel(control_labels[idx], fontsize=13)
        ax.tick_params(labelsize=10)

    # Only bottom row gets x-axis label
    for col in range(2):
        axes[2, col].set_xlabel('Time', fontsize=12)

    fig.suptitle('Fig. 1. State and control during successive solutions.\n'
                 'Dashed: reference $\\hat{x}^i$, $\\hat{u}^i$. '
                 'Solid: mean $\\bar{x}^i$, $\\bar{u}^i$. '
                 'Blue=iter 1, bold=final.',
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig('/home/user/ics/figure1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")


def plot_figure2(history, x_cl, x_ol, x_bar_ref):
    """
    Replicate Figure 2: Position space (xi2 vs xi1) with:
    - Colored mean trajectories per iteration (blue=first)
    - Black dashed chance constraint lines (xi1 = +/- 6)
    - Black 90% confidence ellipses (from linear analysis)
    - Gray 90% confidence ellipses (from Monte Carlo)
    - Dark gray: closed-loop MC sample trajectories
    - Light gray: open-loop MC sample trajectories
    """
    n_iters = len(history['x_bar'])
    idx_final = n_iters - 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Iteration colors (matching Figure 1)
    if n_iters <= 1:
        iter_colors = ['tab:blue']
    else:
        cmap = matplotlib.colormaps.get_cmap('jet').resampled(n_iters)
        iter_colors = [cmap(i / (n_iters - 1)) for i in range(n_iters)]
        iter_colors[0] = (0.0, 0.0, 0.8, 1.0)

    # Open-loop MC (light gray) - plotted first (background)
    n_show = min(200, x_ol.shape[0])
    for trial in range(n_show):
        ax.plot(x_ol[trial, :, 0], x_ol[trial, :, 1],
                '-', color='#cccccc', linewidth=0.3, alpha=0.4)

    # Closed-loop MC (dark gray) - plotted on top of open-loop
    for trial in range(n_show):
        ax.plot(x_cl[trial, :, 0], x_cl[trial, :, 1],
                '-', color='#888888', linewidth=0.3, alpha=0.5)

    # Mean trajectories per iteration (colored, on top)
    for i in range(n_iters):
        lw = 2.5 if i == n_iters - 1 else 1.2
        ax.plot(history['x_bar'][i][:, 0], history['x_bar'][i][:, 1],
                '-', color=iter_colors[i], linewidth=lw)

    # Chance constraint boundaries (vertical dashed lines)
    ax.axvline(x=cc_limit, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=-cc_limit, color='black', linestyle='--', linewidth=1.5)

    # 90% confidence ellipses from linear analysis (black)
    Py_final = history['Py'][idx_final]
    Bi_final = history['Bi_mat'][idx_final]
    K_final = history['K_list'][idx_final]

    K_full = np.zeros((N * nu, (N + 1) * nx))
    for k in range(N):
        K_full[k*nu:(k+1)*nu, k*nx:(k+1)*nx] = K_final[k]

    IpBK = np.eye((N+1)*nx) + Bi_final @ K_full
    Px_full = IpBK @ Py_final @ IpBK.T

    # Show ellipses at selected steps
    ellipse_steps = [0, 5, 10, 15, 20, N]

    for k in ellipse_steps:
        Pxk = Px_full[k*nx:(k+1)*nx, k*nx:(k+1)*nx]
        P_pos = Pxk[:2, :2]
        mean_pos = history['x_bar'][idx_final][k, :2]
        w, h, angle = confidence_ellipse_params(P_pos, confidence=0.9)
        ell = Ellipse(xy=mean_pos, width=w, height=h, angle=angle,
                      fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(ell)

    # 90% confidence ellipses from MC (gray)
    for k in ellipse_steps:
        pos_samples = x_cl[:, k, :2]
        mc_mean = np.mean(pos_samples, axis=0)
        mc_cov = np.cov(pos_samples.T)
        w, h, angle = confidence_ellipse_params(mc_cov, confidence=0.9)
        ell_mc = Ellipse(xy=mc_mean, width=w, height=h, angle=angle,
                         fill=False, edgecolor='gray', linewidth=1.5,
                         linestyle='-')
        ax.add_patch(ell_mc)

    # Mark start and end points
    ax.plot(x_bar_0[0], x_bar_0[1], 'ko', markersize=8, zorder=10)
    ax.plot(x_bar_f[0], x_bar_f[1], 'k*', markersize=12, zorder=10)

    ax.set_xlabel(r'$\xi_1$', fontsize=14)
    ax.set_ylabel(r'$\xi_2$', fontsize=14)
    ax.tick_params(labelsize=11)

    # Match paper's axis range approximately
    ax.set_xlim([-8, 10])
    ax.set_ylim([-3, 10])

    fig.suptitle('Fig. 2. Successive iterations (colored), chance constraint '
                 '(black dashed),\n90% confidence ellipses: linear (black), '
                 'MC (gray). MC trails: CL (dark gray), OL (light gray).',
                 fontsize=10, y=0.98)
    plt.tight_layout()
    plt.savefig('/home/user/ics/figure2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Replicating: Nonlinear Uncertainty Control with")
    print("Iterative Covariance Steering (Ridderhof et al., 2019)")
    print("=" * 60)

    history = run_ics(imax=10, tol=1e-3)

    if len(history['x_bar']) > 0:
        # --- Linear Analysis ---
        idx_f = len(history['x_bar']) - 1
        Py_f = history['Py'][idx_f]
        Bi_f = history['Bi_mat'][idx_f]
        K_f = history['K_list'][idx_f]

        K_full = np.zeros((N * nu, (N + 1) * nx))
        for k in range(N):
            K_full[k*nu:(k+1)*nu, k*nx:(k+1)*nx] = K_f[k]
        IpBK = np.eye((N+1)*nx) + Bi_f @ K_full
        Px_full = IpBK @ Py_f @ IpBK.T

        Pxf_linear = Px_full[N*nx:(N+1)*nx, N*nx:(N+1)*nx]
        print(f"\n--- Linear Analysis ---")
        print(f"Terminal covariance (linear) diag: {np.diag(Pxf_linear)}")
        print(f"Terminal covariance bound (Pxf):   {np.diag(Pxf)}")
        print(f"Constraint satisfied: {np.all(np.diag(Pxf_linear) <= np.diag(Pxf))}")

        # --- Figure 1 ---
        print("\nGenerating Figure 1...")
        plot_figure1(history)

        # --- Monte Carlo ---
        print("\nRunning Monte Carlo (5000 trials)...")
        x_cl, u_cl, x_ol, x_bar_ref = monte_carlo_simulation(history, 5000)

        # Report results (eqs 76-77)
        xf_mean = np.mean(x_cl[:, -1, :], axis=0)
        xf_cov = np.cov(x_cl[:, -1, :].T)
        print(f"\n--- Monte Carlo Results (5000 trials) ---")
        print(f"Final state mean (MC):   {xf_mean}")
        print(f"Expected (eq 76):        [1.004, 1.997, -1.000, -0.001]")
        print(f"\nFinal state covariance (MC):\n{xf_cov}")
        print(f"\nExpected (eq 77):")
        print(f"[[0.018, -0.001,  0.004,  0.000],")
        print(f" [-0.001, 0.016,  0.000,  0.004],")
        print(f" [ 0.004, 0.000,  0.001,  0.000],")
        print(f" [ 0.000, 0.004,  0.000,  0.001]]")

        print(f"\nChance constraint violations P(|xi1| > {cc_limit}):")
        max_viol_pct = 0
        max_viol_step = 0
        for k in range(N + 1):
            viol = np.mean(np.abs(x_cl[:, k, 0]) > cc_limit) * 100
            if viol > 0:
                print(f"  Step {k:2d}: {viol:5.2f}%")
            if viol > max_viol_pct:
                max_viol_pct = viol
                max_viol_step = k
        print(f"Max violation: {max_viol_pct:.2f}% at step {max_viol_step} "
              f"(paper: 9.14% at step 11, limit 10%)")

        # --- Figure 2 ---
        print("\nGenerating Figure 2...")
        plot_figure2(history, x_cl, x_ol, x_bar_ref)

        print("\n" + "=" * 60)
        print("Replication complete. Figures saved:")
        print("  figure1.png - State and control trajectories")
        print("  figure2.png - Position space with MC and ellipses")
        print("=" * 60)
