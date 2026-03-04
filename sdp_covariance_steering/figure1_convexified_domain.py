"""
Figure 1: Example of a convexified domain for a 1-dimensional system.

Replicates Figure 1 from:
"Discrete-time Optimal Covariance Steering via Semidefinite Programming"
by Rapakoulias and Tsiotras, CDC 2023.

Shows the non-convex domain defined by the original chance constraint
(involving sqrt) and its convexification via linearization of sqrt.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_figure1():
    """
    Replicate Figure 1: Convexified domain for a 1D system.

    The original constraint is:
        Phi^{-1}(1-eps) * sqrt(alpha^T Sigma alpha) + alpha^T mu - beta <= 0

    For a 1D system with alpha=1, this becomes:
        c * sqrt(sigma^2) + mu - beta <= 0
        c * sigma + mu <= beta

    where c = Phi^{-1}(1-eps) and sigma^2 is the variance.

    The boundary curve is: mu = beta - c * sqrt(sigma^2)
    which is concave in sigma^2.

    The linearized (convexified) constraint replaces sqrt with its tangent line:
        sqrt(x) approx x/(2*sqrt(x0)) + sqrt(x0)/2
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Parameters
    beta = 5.0
    c = 1.645  # Phi^{-1}(0.95)

    # sigma^2 range (variance)
    sigma2 = np.linspace(0, 5, 500)

    # Original (non-convex) boundary: mu = beta - c * sqrt(sigma^2)
    mu_boundary = beta - c * np.sqrt(sigma2)

    # Reference point for linearization
    sigma2_ref = 2.0
    # Tangent line of sqrt at sigma2_ref:
    # sqrt(x) ≈ x/(2*sqrt(x0)) + sqrt(x0)/2
    # So mu = beta - c * [x/(2*sqrt(x0)) + sqrt(x0)/2]
    mu_linearized = beta - c * (sigma2 / (2 * np.sqrt(sigma2_ref))
                                + np.sqrt(sigma2_ref) / 2)

    # Plot
    # Non-convex domain (shaded area below the curve)
    ax.fill_between(sigma2, mu_boundary, -2, alpha=0.15, color='blue',
                    label='Non-convex domain')

    # Convexified domain (shaded area below the tangent line)
    ax.fill_between(sigma2, np.minimum(mu_linearized, mu_boundary),
                    -2, alpha=0.25, color='green',
                    label='Convexified domain')

    # Original boundary curve
    ax.plot(sigma2, mu_boundary, 'b-', linewidth=2.5,
            label='Non-convex boundary')

    # Linearized boundary
    ax.plot(sigma2, mu_linearized, 'r--', linewidth=2,
            label='Linearized boundary')

    # Mark the reference point
    mu_ref = beta - c * np.sqrt(sigma2_ref)
    ax.plot(sigma2_ref, mu_ref, 'ro', markersize=8, zorder=5)
    ax.annotate(f'$\\sigma^2_r = {sigma2_ref:.1f}$',
                xy=(sigma2_ref, mu_ref),
                xytext=(sigma2_ref + 0.5, mu_ref + 0.5),
                fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('$\\alpha_x^T \\Sigma_k \\alpha_x$', fontsize=13)
    ax.set_ylabel('$\\alpha_x^T \\mu_k$', fontsize=13)
    ax.set_title('Convexification of feasible domain', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5.5])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/figure1_convexified_domain.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("Figure 1 saved to figures/figure1_convexified_domain.png")


if __name__ == '__main__':
    plot_figure1()
