"""
Two-dimensional Chebyshev approximation of the inverse central chi-square CDF.

Paper methodology:
    F_delta^{-1}(u) approximated by S(x,y) = sum_{m=0}^M sum_{n=0}^N c_{mn} T_m(y) T_n(x)

where:
    - y = (2*delta - (delta_max+delta_min))/(delta_max-delta_min) maps delta to [-1,1]
    - x = 2*(xi-xi_min)/(xi_max-xi_min) - 1 maps the transformed u-variable to [-1,1]
    - xi(u) is a region-specific monotone transformation

Region boundaries (appendix-implied fixed values for delta in [0.1,0.2]):
    Region 1: u in [0, 0.8],       xi(u) = u
    Region 2: u in [0.8, 0.95],    xi(u) = log((1-u)*Gamma(delta/2))
    Region 3: u in [0.95, 1-1e-8], xi(u) = log(-log((1-u)*Gamma(delta/2)))

For smaller delta intervals the Region 3 lower boundary is adjusted adaptively to ensure
(1-u)*Gamma(delta/2) < 1 throughout Region 3 (required for the log-log transform).

Truncation orders from the paper for O(10^-8) approximation (M, N):
    delta in [0.1,  0.2]:  R1=(5,13),  R2=(3,11),  R3=(3,13)
    delta in [0.01, 0.02]: R1=(4,33),  R2=(3,13),  R3=(3,12)
    delta in [0.001,0.002]:R1=(4,39),  R2=(2,35),  R3=(3,13)
    Note: R2 N raised to 35 (paper's 13 is insufficient for the 6.9-unit xi range
    forced by the adaptive u23=0.9996 required for valid Region 3 at this delta)

Custom implementation – Context7 found no library equivalent for this two-dimensional
Chebyshev inverse chi-square sampler.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import special, stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tail truncation: u is clipped to [0, U_MAX] before **evaluation**
U_MAX: float = 1.0 - 1e-8

#: Region 3 xi upper bound for **coefficient computation**.
#: The paper states "replace the upper w-limit by 20", where
#: w = -log((1-u)*Gamma(delta/2)), so xi = log(w) → xi_max = log(20).
#: This extends coefficient nodes beyond U_MAX (to u ≈ 1-1e-10) so that
#: no U_MAX clipping plateau corrupts the F_matrix near the right boundary.
XI_MAX_R3: float = np.log(20.0)  # ≈ 2.9957

#: Default u-region lower boundaries (paper: [0.1,0.2] interval)
DEFAULT_U12: float = 0.8   # boundary between Region 1 and 2
DEFAULT_U23: float = 0.95  # boundary between Region 2 and 3

#: Published truncation orders per delta interval and region
#: Keys: (delta_min, delta_max), Values: {region_id: (M, N)}
TRUNCATION_ORDERS: Dict[Tuple[float, float], Dict[int, Tuple[int, int]]] = {
    (0.1, 0.2):    {1: (5, 13), 2: (3, 11), 3: (3, 13)},
    (0.01, 0.02):  {1: (4, 33), 2: (3, 13), 3: (3, 12)},
    (0.001, 0.002):{1: (4, 39), 2: (2, 35), 3: (3, 13)},  # R2 N=35 (see module docstring)
}

# ---------------------------------------------------------------------------
# Chebyshev polynomial utilities
# ---------------------------------------------------------------------------


def chebyshev_T(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Chebyshev polynomial T_n(x) using the three-term recurrence.

    Parameters
    ----------
    n : int
        Polynomial degree (n >= 0).
    x : np.ndarray
        Evaluation points in [-1, 1].

    Returns
    -------
    np.ndarray
        T_n(x) values.
    """
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    t_prev, t_curr = np.ones_like(x), x.copy()
    for _ in range(2, n + 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
    return t_curr


def delta_to_y(delta: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    """Map delta from [delta_min, delta_max] to y in [-1, 1]."""
    return (2.0 * delta - (delta_max + delta_min)) / (delta_max - delta_min)


def y_to_delta(y: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    """Inverse of delta_to_y: map y in [-1,1] back to delta."""
    return 0.5 * (y * (delta_max - delta_min) + delta_max + delta_min)


def xi_to_x(xi: np.ndarray, xi_min: float, xi_max: float) -> np.ndarray:
    """Map xi linearly from [xi_min, xi_max] to x in [-1, 1]."""
    return 2.0 * (xi - xi_min) / (xi_max - xi_min) - 1.0


def x_to_xi(x: np.ndarray, xi_min: float, xi_max: float) -> np.ndarray:
    """Inverse of xi_to_x."""
    return xi_min + 0.5 * (x + 1.0) * (xi_max - xi_min)


# ---------------------------------------------------------------------------
# Region transformations
# ---------------------------------------------------------------------------


def u_to_xi(u: np.ndarray, region: int, delta: float) -> np.ndarray:
    """
    Transform u to xi using the region-specific monotone mapping.

    Region 1: xi = u                                   (identity)
    Region 2: xi = log((1-u) * Gamma(delta/2))
    Region 3: xi = log(-log((1-u) * Gamma(delta/2)))  [requires (1-u)*Gamma<1]

    Parameters
    ----------
    u : np.ndarray
        Probability values in the region's u-interval.
    region : int
        Region identifier: 1, 2, or 3.
    delta : float
        Degrees of freedom.

    Returns
    -------
    np.ndarray
        Transformed values xi.
    """
    u = np.asarray(u, dtype=float)
    gam_half = special.gamma(0.5 * delta)

    if region == 1:
        return u.copy()
    elif region == 2:
        return np.log((1.0 - u) * gam_half)
    elif region == 3:
        g = (1.0 - u) * gam_half
        # Clip g to (0, 1) to ensure the inner log is negative
        g = np.clip(g, 1e-300, 1.0 - 1e-15)
        return np.log(-np.log(g))
    else:
        raise ValueError(f"region must be 1, 2, or 3; got {region}")


def compute_u23_adaptive(delta_min: float) -> float:
    """
    Compute the Region 2/3 boundary u_23 so that Region 3's xi-transform is valid
    for all delta in [delta_min, delta_max].

    Validity condition for Region 3: (1-u)*Gamma(delta/2) < 1, i.e.,
        u > 1 - 1/Gamma(delta/2)
    The most restrictive constraint is at delta = delta_min where Gamma is largest.

    Parameters
    ----------
    delta_min : float
        Lower bound of the delta interval.

    Returns
    -------
    float
        Minimum u value for which Region 3 is valid across the entire delta interval.
    """
    gam = special.gamma(0.5 * delta_min)
    u23_required = 1.0 - 1.0 / gam
    # Add a small buffer and use at least the default 0.95
    u23 = max(DEFAULT_U23, u23_required + 1e-4)
    # Ensure there is room for Region 3 (at least 1e-6 wide)
    u23 = min(u23, U_MAX - 1e-6)
    return u23


def get_region_u_bounds(
    delta_min: float, delta_max: float
) -> List[Tuple[float, float]]:
    """
    Return the (u_lo, u_hi) intervals for each of the three regions.

    The Region 2/3 boundary is adapted for small delta to ensure that the
    log-log transform in Region 3 remains well-defined.

    Parameters
    ----------
    delta_min, delta_max : float
        Delta interval bounds.

    Returns
    -------
    list of (u_lo, u_hi) tuples for regions 1, 2, 3.
    """
    u23 = compute_u23_adaptive(delta_min)
    return [
        (0.0, DEFAULT_U12),   # Region 1
        (DEFAULT_U12, u23),   # Region 2
        (u23, U_MAX),         # Region 3
    ]


def get_xi_bounds(
    region: int,
    delta_min: float,
    delta_max: float,
    u_bounds: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Compute xi_min, xi_max for the x-mapping.

    Region 1 (xi = u)
    -----------------
    Delta-independent; bounds equal the u-bounds.

    Region 2 (xi = log((1-u)*Gamma(delta/2)))
    ------------------------------------------
    Uses the **union** (outer envelope) of the four corner xi values so that
    every valid (delta, u) pair maps to x ∈ [-1, 1].

    Region 3 (xi = log(-log((1-u)*Gamma(delta/2))))
    ------------------------------------------------
    xi_min  : union minimum across all four corners (at u_lo and u_hi, both
              delta endpoints).  Some xi nodes near xi_min map to u < u_lo
              for the larger-delta rows of F_matrix; those u values are in
              Region 2 territory but the inverse CDF is smooth there.
    xi_max  : **fixed at log(20)** as specified by the paper ("replace the
              upper w-limit by 20", where w = -log((1-u)*Gamma(delta/2))).
              This extends coefficient nodes to u ≈ 1−1e-10, which is beyond
              U_MAX = 1−1e-8.  Without this extension the nodes near xi_max
              get clipped to U_MAX during F_matrix computation, creating an
              artificial plateau that corrupts the Chebyshev coefficients and
              causes O(0.1) approximation error at the tail boundary.
              During online *evaluation* u is still clipped to U_MAX.

    Parameters
    ----------
    region : int
        Region identifier (1, 2, or 3).
    delta_min, delta_max : float
        Delta interval bounds.
    u_bounds : (float, float)
        (u_lo, u_hi) for this region.

    Returns
    -------
    (xi_min, xi_max) : tuple of float
    """
    u_lo, u_hi = u_bounds
    if region == 1:
        return (u_lo, u_hi)

    # For Region 3: xi_max is the paper's fixed w=20 upper limit.
    if region == 3:
        # xi_min from union of all four corners (u_lo and u_hi at both delta endpoints)
        corners_xi: list[float] = []
        for delta in [delta_min, delta_max]:
            for u in [u_lo, u_hi]:
                xi = float(u_to_xi(np.array([u]), region, delta)[0])
                if np.isfinite(xi):
                    corners_xi.append(xi)
        xi_min = min(corners_xi) if corners_xi else -4.0
        xi_max = XI_MAX_R3  # log(20) ≈ 2.9957
        assert xi_min < xi_max, (
            f"Region 3 xi range degenerate: xi_min={xi_min}, xi_max={xi_max} "
            f"delta=[{delta_min},{delta_max}], u=[{u_lo},{u_hi}]"
        )
        return (xi_min, xi_max)

    # Region 2: union of all four corners
    corners_xi2: list[float] = []
    for delta in [delta_min, delta_max]:
        for u in [u_lo, u_hi]:
            xi = float(u_to_xi(np.array([u]), region, delta)[0])
            if np.isfinite(xi):
                corners_xi2.append(xi)

    assert len(corners_xi2) >= 2, (
        f"Too few finite corner xi values for region={region}, "
        f"delta=[{delta_min},{delta_max}], u=[{u_lo},{u_hi}]"
    )
    xi_min2 = min(corners_xi2)
    xi_max2 = max(corners_xi2)

    assert xi_min2 < xi_max2, (
        f"Union xi range is degenerate: xi_min={xi_min2}, xi_max={xi_max2} "
        f"for region={region}, delta=[{delta_min},{delta_max}], u=[{u_lo},{u_hi}]"
    )
    return (xi_min2, xi_max2)


# ---------------------------------------------------------------------------
# Coefficient computation (pseudospectral / DCT approach)
# ---------------------------------------------------------------------------


def _chebyshev_gauss_nodes(n_pts: int) -> np.ndarray:
    """
    Return n_pts Chebyshev-Gauss (interior) quadrature nodes in [-1, 1].

    x_k = cos(pi*(2k+1) / (2*n_pts)), k=0,...,n_pts-1
    """
    k = np.arange(n_pts)
    return np.cos(np.pi * (2.0 * k + 1.0) / (2.0 * n_pts))


def compute_chebyshev_coefficients(
    delta_min: float,
    delta_max: float,
    region: int,
    M: int,
    N: int,
    u_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Compute the (M+1) x (N+1) two-dimensional Chebyshev coefficient matrix for
    F_delta^{-1}(u) over the given delta interval and u-region.

    Uses the pseudospectral (Chebyshev-Gauss collocation) approach:
        1. Place (M+1) y-nodes and (N+1) x-nodes at Chebyshev-Gauss points.
        2. Evaluate q(delta, u) = scipy.stats.chi2.ppf(u, df=delta) at all nodes.
        3. Compute coefficients via 2D Chebyshev discrete transform.

    Convention: c_{mn} indexes delta dimension as m (rows) and u dimension as n (cols).

    Parameters
    ----------
    delta_min, delta_max : float
        Bounds of the delta interval.
    region : int
        Region identifier: 1, 2, or 3.
    M : int
        Maximum Chebyshev degree in the delta (y) dimension.
    N : int
        Maximum Chebyshev degree in the u-transformed (x) dimension.
    u_bounds : (float, float), optional
        Override the default u bounds for this region.

    Returns
    -------
    dict with keys: coefficient_matrix (M+1, N+1), xi_min, xi_max, k1, k2,
                    M, N, delta_min, delta_max, region, u_bounds, transform_type.
    """
    if u_bounds is None:
        region_bounds = get_region_u_bounds(delta_min, delta_max)
        u_bounds = region_bounds[region - 1]

    xi_min, xi_max = get_xi_bounds(region, delta_min, delta_max, u_bounds)
    k1 = 2.0 / (xi_max - xi_min)
    k2 = -1.0 - 2.0 * xi_min / (xi_max - xi_min)  # x = k1*xi + k2

    # Chebyshev-Gauss quadrature nodes (use M+1 and N+1 nodes for exactness)
    My = M + 1  # number of y (delta) nodes
    Nx = N + 1  # number of x (u) nodes

    y_nodes = _chebyshev_gauss_nodes(My)  # in [-1,1], delta dimension
    x_nodes = _chebyshev_gauss_nodes(Nx)  # in [-1,1], xi dimension

    delta_nodes = y_to_delta(y_nodes, delta_min, delta_max)
    xi_nodes = x_to_xi(x_nodes, xi_min, xi_max)

    # Map xi nodes back to u via the inverse transform, then evaluate chi2.ppf
    F_matrix = np.zeros((My, Nx))  # F[j, k] = q(delta_j, u(xi_k))

    for j, (delta_j, y_j) in enumerate(zip(delta_nodes, y_nodes)):
        # Compute u from xi for this region and delta.
        # For Region 3 with xi_max = log(20) (paper's w=20 upper limit), nodes
        # near xi_max correspond to u ≈ 1−1e-10, which is beyond U_MAX=1−1e-8.
        # We must NOT clip to U_MAX here: doing so creates a plateau in F_matrix
        # (chi2.ppf saturates at the same value for multiple xi nodes), which
        # corrupts the Chebyshev coefficients and causes O(0.1) tail errors.
        # The upper clip is loosened to 1−1e-15 (avoids log(0) in chi2.ppf
        # internals while letting u reach the true collocation value).
        # The U_MAX clip is applied only during online *evaluation*.
        u_j = _xi_to_u(xi_nodes, region, delta_j)
        u_j = np.clip(u_j, 1e-15, 1.0 - 1e-15)
        # Evaluate reference inverse CDF
        F_matrix[j, :] = stats.chi2.ppf(u_j, df=delta_j)

    # Compute 2D Chebyshev coefficients via discrete transform
    # c_{mn} = (eps_m * eps_n / (My * Nx)) * sum_{j,k} F[j,k] * T_m(y_j) * T_n(x_k)
    C = np.zeros((M + 1, N + 1))
    for m in range(M + 1):
        Tm_y = chebyshev_T(m, y_nodes)       # shape (My,)
        for n in range(N + 1):
            Tn_x = chebyshev_T(n, x_nodes)   # shape (Nx,)
            # Outer product weighted sum
            s = np.dot(Tm_y, F_matrix.dot(Tn_x))
            eps_m = 1.0 if m > 0 else 0.5
            eps_n = 1.0 if n > 0 else 0.5
            C[m, n] = (4.0 * eps_m * eps_n / (My * Nx)) * s

    transform_type = ["identity", "log_gamma", "loglog_gamma"][region - 1]
    return {
        "coefficient_matrix": C,
        "xi_min": xi_min,
        "xi_max": xi_max,
        "k1": k1,
        "k2": k2,
        "M": M,
        "N": N,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "region": region,
        "u_bounds": list(u_bounds),
        "transform_type": transform_type,
    }


def _xi_to_u(xi: np.ndarray, region: int, delta: float) -> np.ndarray:
    """
    Inverse of u_to_xi: compute u from xi for the given region and delta.

    Region 1: u = xi
    Region 2: log((1-u)*G) = xi  =>  1-u = exp(xi)/G  =>  u = 1 - exp(xi)/G
    Region 3: log(-log((1-u)*G)) = xi  =>  (1-u)*G = exp(-exp(xi))
              =>  u = 1 - exp(-exp(xi))/G
    """
    xi = np.asarray(xi, dtype=float)
    gam = special.gamma(0.5 * delta)
    if region == 1:
        return xi.copy()
    elif region == 2:
        return 1.0 - np.exp(xi) / gam
    elif region == 3:
        return 1.0 - np.exp(-np.exp(xi)) / gam
    else:
        raise ValueError(f"region must be 1, 2, or 3; got {region}")


# ---------------------------------------------------------------------------
# Clenshaw recurrence evaluation
# ---------------------------------------------------------------------------


def clenshaw_1d(x: float, coeffs: np.ndarray) -> float:
    """
    Evaluate a 1D Chebyshev series sum_{n=0}^N c_n T_n(x) using Clenshaw recurrence.

    Parameters
    ----------
    x : float
        Evaluation point in [-1, 1].
    coeffs : np.ndarray
        Coefficient array of length N+1.

    Returns
    -------
    float
        Series value at x.
    """
    N = len(coeffs) - 1
    if N == 0:
        return coeffs[0]
    b_next = 0.0
    b_curr = 0.0
    for n in range(N, 0, -1):
        b_prev = coeffs[n] + 2.0 * x * b_curr - b_next
        b_next = b_curr
        b_curr = b_prev
    return coeffs[0] + x * b_curr - b_next


def clenshaw_2d(x: float, y: float, C: np.ndarray) -> float:
    """
    Evaluate the 2D Chebyshev series S(x,y)=sum_{m,n} c_{mn} T_m(y) T_n(x) using
    nested Clenshaw recurrence.

    Strategy: for each row m, evaluate g_m(x) = sum_n c_{mn} T_n(x) using 1D Clenshaw,
    then evaluate S = sum_m g_m T_m(y) using a second 1D Clenshaw.

    Parameters
    ----------
    x, y : float
        Evaluation points, each in [-1, 1].
    C : np.ndarray, shape (M+1, N+1)
        Coefficient matrix c_{mn}.

    Returns
    -------
    float
        Series value at (x, y).
    """
    M = C.shape[0] - 1
    # Step 1: collapse x-dimension for each m
    g = np.array([clenshaw_1d(x, C[m, :]) for m in range(M + 1)])
    # Step 2: evaluate sum_m g_m T_m(y)
    return clenshaw_1d(y, g)


def clenshaw_2d_array(
    x: np.ndarray, y: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """
    Vectorised version of clenshaw_2d for arrays of (x, y) pairs.

    Parameters
    ----------
    x, y : np.ndarray, shape (n_pts,)
    C : np.ndarray, shape (M+1, N+1)

    Returns
    -------
    np.ndarray, shape (n_pts,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pts = x.shape[0]
    M, N = C.shape[0] - 1, C.shape[1] - 1

    # Vectorised 1D Clenshaw in x-direction for all rows m
    # g[m, i] = sum_n c_{mn} T_n(x_i)
    g = np.zeros((M + 1, n_pts))
    for m in range(M + 1):
        b_next = np.zeros(n_pts)
        b_curr = np.zeros(n_pts)
        for n in range(N, 0, -1):
            b_prev = C[m, n] + 2.0 * x * b_curr - b_next
            b_next = b_curr
            b_curr = b_prev
        g[m] = C[m, 0] + x * b_curr - b_next

    # Vectorised 1D Clenshaw in y-direction
    b_next = np.zeros(n_pts)
    b_curr = np.zeros(n_pts)
    for m in range(M, 0, -1):
        b_prev = g[m] + 2.0 * y * b_curr - b_next
        b_next = b_curr
        b_curr = b_prev
    return g[0] + y * b_curr - b_next


def direct_sum_2d(x: float, y: float, C: np.ndarray) -> float:
    """
    Evaluate 2D Chebyshev series by direct double summation (for verification).

    Parameters
    ----------
    x, y : float
        Evaluation points in [-1, 1].
    C : np.ndarray, shape (M+1, N+1)
    """
    M, N = C.shape[0] - 1, C.shape[1] - 1
    result = 0.0
    for m in range(M + 1):
        Tm_y = chebyshev_T(m, np.array([y]))[0]
        for n in range(N + 1):
            Tn_x = chebyshev_T(n, np.array([x]))[0]
            result += C[m, n] * Tm_y * Tn_x
    return result


# ---------------------------------------------------------------------------
# Derivative coefficients
# ---------------------------------------------------------------------------


def derivative_coefficients_y(C: np.ndarray) -> np.ndarray:
    """
    Compute the derivative coefficient array for d/dy of the Chebyshev series
    sum_m c_m T_m(y), using the recursion:

        b_m = b_{m+2} + 2*(m+1) * c_{m+1}   (with b_{M+1} = b_{M+2} = 0)

    This generalises to 2D by applying the recurrence column-by-column (each
    n-index independently).

    The standard Chebyshev derivative recursion produces the T_0 coefficient
    in the "half-convention" (doubled relative to the plain-convention sum used
    by Clenshaw).  After the recursion, divide b[0] by 2 to convert back to
    the plain convention: sum_m d_m T_m(y) where d_0 is not halved.

    Parameters
    ----------
    C : np.ndarray, shape (M+1, N+1)

    Returns
    -------
    np.ndarray, shape (M+1, N+1)
        Derivative coefficient matrix dC/dy (plain convention, matches Clenshaw).
    """
    M, N_1 = C.shape
    M -= 1
    dC = np.zeros_like(C)
    for n in range(N_1):
        b = np.zeros(M + 3)  # b[0..M+2], with b[M+1]=b[M+2]=0
        for m in range(M, -1, -1):
            b[m] = b[m + 2] + 2.0 * (m + 1) * C[m + 1, n] if m + 1 <= M else b[m + 2]
        dC[:, n] = b[:M + 1]
    # Convert T_0 row from half-convention to plain convention (Custom — paper Eq. A.1)
    dC[0, :] *= 0.5
    return dC


def derivative_coefficients_x(C: np.ndarray) -> np.ndarray:
    """
    Compute the derivative coefficient array for d/dx of the 2D Chebyshev series,
    applying the 1D derivative recursion row-by-row (each m-index independently).

    Same half-convention correction as derivative_coefficients_y applies: the
    recursion doubles the T_0 (n=0) component; divide the first column by 2 to
    return plain-convention coefficients compatible with Clenshaw evaluation.

    Parameters
    ----------
    C : np.ndarray, shape (M+1, N+1)

    Returns
    -------
    np.ndarray, shape (M+1, N+1)
        Derivative coefficient matrix dC/dx (plain convention, matches Clenshaw).
    """
    M1, N = C.shape
    N -= 1
    dC = np.zeros_like(C)
    for m in range(M1):
        b = np.zeros(N + 3)
        for n in range(N, -1, -1):
            b[n] = b[n + 2] + 2.0 * (n + 1) * C[m, n + 1] if n + 1 <= N else b[n + 2]
        dC[m, :] = b[:N + 1]
    # Convert T_0 column from half-convention to plain convention (Custom — paper Eq. A.1)
    dC[:, 0] *= 0.5
    return dC


# ---------------------------------------------------------------------------
# Online evaluator
# ---------------------------------------------------------------------------


def evaluate_inverse_chi2(delta: float, u: float, coeff_block: Dict) -> float:
    """
    Evaluate F_delta^{-1}(u) using a precomputed Chebyshev coefficient block.

    Parameters
    ----------
    delta : float
    u : float, clipped to [0, U_MAX]
    coeff_block : dict
        Single region's coefficient block as returned by compute_chebyshev_coefficients.

    Returns
    -------
    float
    """
    u = np.clip(u, 0.0, U_MAX)
    delta_min = coeff_block["delta_min"]
    delta_max = coeff_block["delta_max"]
    region = coeff_block["region"]
    xi_min = coeff_block["xi_min"]
    xi_max = coeff_block["xi_max"]
    C = np.asarray(coeff_block["coefficient_matrix"])

    y = float(delta_to_y(np.array([delta]), delta_min, delta_max)[0])
    xi = float(u_to_xi(np.array([u]), region, delta)[0])
    x = float(xi_to_x(np.array([xi]), xi_min, xi_max)[0])

    # Clip x, y to [-1, 1] for numerical safety
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)

    # Chi-square quantiles are non-negative; clip polynomial noise near zero
    return max(0.0, clenshaw_2d(x, y, C))


def evaluate_inverse_chi2_batch(
    delta: float,
    u_array: np.ndarray,
    coeff_blocks: List[Dict],
    region_u_bounds: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Evaluate F_delta^{-1}(u) for an array of u values using precomputed blocks.

    Routes each u value to the appropriate region block.

    Parameters
    ----------
    delta : float
    u_array : np.ndarray
    coeff_blocks : list of 3 dicts (one per region)
    region_u_bounds : list of 3 (u_lo, u_hi) tuples

    Returns
    -------
    np.ndarray
    """
    u_array = np.clip(np.asarray(u_array, dtype=float), 0.0, U_MAX)
    result = np.empty_like(u_array)

    for region_idx, (block, (u_lo, u_hi)) in enumerate(
        zip(coeff_blocks, region_u_bounds)
    ):
        mask = (u_array >= u_lo) & (u_array < u_hi)
        if not np.any(mask):
            continue
        u_reg = u_array[mask]
        region = region_idx + 1
        delta_min = block["delta_min"]
        delta_max = block["delta_max"]
        xi_min = block["xi_min"]
        xi_max = block["xi_max"]
        C = np.asarray(block["coefficient_matrix"])

        y = delta_to_y(np.full(len(u_reg), delta), delta_min, delta_max)
        xi = u_to_xi(u_reg, region, delta)
        x = xi_to_x(xi, xi_min, xi_max)
        x = np.clip(x, -1.0, 1.0)
        y = np.clip(y, -1.0, 1.0)
        result[mask] = clenshaw_2d_array(x, y, C)

    # Handle upper boundary exactly at U_MAX
    mask_max = u_array >= U_MAX
    if np.any(mask_max):
        u_lo3, u_hi3 = region_u_bounds[2]
        block = coeff_blocks[2]
        u_clipped = np.full(mask_max.sum(), U_MAX - 1e-12)
        delta_min = block["delta_min"]
        delta_max = block["delta_max"]
        xi_min = block["xi_min"]
        xi_max = block["xi_max"]
        C = np.asarray(block["coefficient_matrix"])
        y = delta_to_y(np.full(len(u_clipped), delta), delta_min, delta_max)
        xi = u_to_xi(u_clipped, 3, delta)
        x = xi_to_x(xi, xi_min, xi_max)
        x = np.clip(x, -1.0, 1.0)
        y = np.clip(y, -1.0, 1.0)
        result[mask_max] = clenshaw_2d_array(x, y, C)

    # Chi-square quantiles are non-negative; clip polynomial noise near zero.
    # This matters for very small delta where the true quantile underflows to 0
    # but the polynomial may return a small negative value.
    np.maximum(result, 0.0, out=result)
    return result


def evaluate_derivative_delta(
    delta: float,
    u: float,
    coeff_block: Dict,
) -> float:
    """
    Evaluate the analytical partial derivative d/d(delta) of F_delta^{-1}(u)
    from transformed Chebyshev coefficients using the chain rule.

    d/d(delta) S(x(xi(u,delta)), y(delta)) =
        dS/dy * dy/d(delta) + dS/dx * dx/d(delta)

    where dx/d(delta) comes from the dependence of xi on delta (Regions 2 and 3)
    through Gamma(delta/2).

    Parameters
    ----------
    delta : float
    u : float
    coeff_block : dict

    Returns
    -------
    float
    """
    u = float(np.clip(u, 0.0, U_MAX))
    delta_min = coeff_block["delta_min"]
    delta_max = coeff_block["delta_max"]
    region = coeff_block["region"]
    xi_min = coeff_block["xi_min"]
    xi_max = coeff_block["xi_max"]
    C = np.asarray(coeff_block["coefficient_matrix"])

    y = float(delta_to_y(np.array([delta]), delta_min, delta_max)[0])
    xi_val = float(u_to_xi(np.array([u]), region, delta)[0])
    x = float(xi_to_x(np.array([xi_val]), xi_min, xi_max)[0])
    x = float(np.clip(x, -1.0, 1.0))
    y = float(np.clip(y, -1.0, 1.0))

    # Derivative coefficients
    dC_dy = derivative_coefficients_y(C)
    dC_dx = derivative_coefficients_x(C)

    dS_dy = clenshaw_2d(x, y, dC_dy)
    dS_dx = clenshaw_2d(x, y, dC_dx)

    # dy/d(delta)
    dy_ddelta = 2.0 / (delta_max - delta_min)

    # dx/d(delta): chain through xi
    # d(xi)/d(delta) for each region
    gam = special.gamma(0.5 * delta)
    psi = special.digamma(0.5 * delta)  # d/d(delta) Gamma(delta/2) = Gamma(delta/2)*psi(delta/2) * (1/2)
    dgam_ddelta = 0.5 * gam * psi

    if region == 1:
        dxi_ddelta = 0.0
    elif region == 2:
        # xi = log((1-u)*gam)  =>  d(xi)/d(delta) = dgam/d(delta) / gam
        dxi_ddelta = dgam_ddelta / gam
    elif region == 3:
        # xi = log(-log((1-u)*gam))
        # Let g = (1-u)*gam,  inner = -log(g)
        # xi = log(inner)
        g = (1.0 - u) * gam
        g = max(g, 1e-300)
        g = min(g, 1.0 - 1e-15)
        inner = -np.log(g)
        dg_ddelta = (1.0 - u) * dgam_ddelta
        dxi_ddelta = -dg_ddelta / (g * inner)

    # dx/d(xi) = k1 = 2/(xi_max - xi_min)
    dx_dxi = 2.0 / (xi_max - xi_min)
    dx_ddelta = dx_dxi * dxi_ddelta

    return dS_dy * dy_ddelta + dS_dx * dx_ddelta


# ---------------------------------------------------------------------------
# Coefficient storage
# ---------------------------------------------------------------------------


def save_coefficients(filepath: str, coeff_block: Dict) -> None:
    """
    Save a Chebyshev coefficient block to a JSON file.

    Schema fields: delta_interval, region_id, u_bounds, transform_type,
    k1, k2, M, N, coefficient_matrix, tolerance, date.
    """
    data = {
        "delta_interval": [coeff_block["delta_min"], coeff_block["delta_max"]],
        "region_id": coeff_block["region"],
        "u_bounds": coeff_block["u_bounds"],
        "transform_type": coeff_block["transform_type"],
        "k1": coeff_block["k1"],
        "k2": coeff_block["k2"],
        "xi_min": coeff_block["xi_min"],
        "xi_max": coeff_block["xi_max"],
        "M": coeff_block["M"],
        "N": coeff_block["N"],
        "coefficient_matrix": np.asarray(coeff_block["coefficient_matrix"]).tolist(),
        "tolerance": 1e-8,
        "date": str(date.today()),
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_coefficients(filepath: str) -> Dict:
    """Load a Chebyshev coefficient block from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    block = {
        "delta_min": data["delta_interval"][0],
        "delta_max": data["delta_interval"][1],
        "region": data["region_id"],
        "u_bounds": tuple(data["u_bounds"]),
        "transform_type": data["transform_type"],
        "k1": data["k1"],
        "k2": data["k2"],
        "xi_min": data["xi_min"],
        "xi_max": data["xi_max"],
        "M": data["M"],
        "N": data["N"],
        "coefficient_matrix": np.array(data["coefficient_matrix"]),
        "tolerance": data["tolerance"],
    }
    return block


def build_all_coefficient_blocks(
    output_dir: str,
    delta_intervals: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, List[Dict]]:
    """
    Build and save Chebyshev coefficient blocks for all specified delta intervals.

    Parameters
    ----------
    output_dir : str
        Directory in which to save coefficient JSON files.
    delta_intervals : list of (delta_min, delta_max), optional
        Defaults to all three published intervals.

    Returns
    -------
    dict mapping str(delta_interval) -> list of 3 region dicts.
    """
    if delta_intervals is None:
        delta_intervals = list(TRUNCATION_ORDERS.keys())

    os.makedirs(output_dir, exist_ok=True)
    all_blocks: Dict[str, List[Dict]] = {}

    for delta_min, delta_max in delta_intervals:
        key = f"delta_{delta_min}_{delta_max}"
        orders = TRUNCATION_ORDERS.get((delta_min, delta_max), None)
        if orders is None:
            # Fall back to moderate orders if interval not in the table
            orders = {1: (4, 15), 2: (3, 12), 3: (3, 12)}

        region_u_bounds = get_region_u_bounds(delta_min, delta_max)
        blocks: List[Dict] = []
        for region in [1, 2, 3]:
            M, N = orders[region]
            u_bounds = region_u_bounds[region - 1]
            print(
                f"  Building delta=[{delta_min},{delta_max}] Region {region} "
                f"M={M} N={N} u=[{u_bounds[0]:.4f},{u_bounds[1]:.6f}]"
            )
            block = compute_chebyshev_coefficients(
                delta_min, delta_max, region, M, N, u_bounds
            )
            # Save to file
            fname = os.path.join(
                output_dir,
                f"cheb_{delta_min}_{delta_max}_r{region}.json",
            )
            save_coefficients(fname, block)
            blocks.append(block)

        all_blocks[key] = blocks

    return all_blocks


def load_all_coefficient_blocks(
    coeff_dir: str,
    delta_intervals: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, List[Dict]]:
    """
    Load precomputed coefficient blocks from a directory.

    Parameters
    ----------
    coeff_dir : str
        Directory containing JSON coefficient files.
    delta_intervals : list, optional
        Which intervals to load. Defaults to all three published intervals.

    Returns
    -------
    dict mapping str(delta_interval) -> list of 3 region dicts.
    """
    if delta_intervals is None:
        delta_intervals = list(TRUNCATION_ORDERS.keys())

    all_blocks: Dict[str, List[Dict]] = {}
    for delta_min, delta_max in delta_intervals:
        key = f"delta_{delta_min}_{delta_max}"
        blocks = []
        for region in [1, 2, 3]:
            fname = os.path.join(
                coeff_dir, f"cheb_{delta_min}_{delta_max}_r{region}.json"
            )
            blocks.append(load_coefficients(fname))
        all_blocks[key] = blocks
    return all_blocks
