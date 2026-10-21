"""
Central and non-central chi-square samplers based on direct inversion and
the Ahrens-Dieter benchmark method.

Direct inversion uses the two-dimensional Chebyshev approximation of
F_delta^{-1}(u) from chebyshev.py.

Non-central chi-square sampler decomposition:
  - For lambda <= 10: chi2_delta(lambda) = chi2_delta + chi2_{2N}, N~Poisson(lambda/2)
    where chi2_{2N} = -2 * sum_{i=1}^N log(U_i)
  - For lambda > 10: recursive splitting: repeatedly peel off a Poisson(5) component
    until residual non-centrality <= 10.

Ahrens-Dieter (1982) GS algorithm for Gamma(alpha, 1) with alpha in (0, 1)
is used as the benchmark central chi-square method.

Custom implementation – Context7 found no library equivalent.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import special, stats

from src.chebyshev import (
    U_MAX,
    evaluate_inverse_chi2_batch,
    get_region_u_bounds,
    load_all_coefficient_blocks,
)

# ---------------------------------------------------------------------------
# Ahrens-Dieter GS algorithm for Gamma(alpha) with alpha in (0, 1)
# ---------------------------------------------------------------------------


def _ahrens_dieter_gamma(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate n samples from Gamma(alpha, 1) using the Ahrens-Dieter (1982)
    GS algorithm. Valid for alpha in (0, 1).

    Algorithm GS (Ahrens & Dieter 1982):
        b = (e + alpha) / e
        Loop:
            U1, U2 ~ Uniform(0,1)
            p = b * U1
            if p > 1:
                x = -log((b - p) / alpha)
                if U2 <= x^(alpha-1): accept
            else:
                x = p^(1/alpha)
                if U2 <= exp(-x): accept
        return x

    Parameters
    ----------
    alpha : float
        Shape parameter in (0, 1).
    n : int
        Number of samples.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of shape (n,)
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")

    b = (np.e + alpha) / np.e
    result = np.empty(n)
    idx = 0
    # Over-generate to reduce loop iterations; batch acceptance-rejection
    batch = max(n * 4, 1000)
    while idx < n:
        u1 = rng.uniform(size=batch)
        u2 = rng.uniform(size=batch)
        p = b * u1
        big = p > 1.0
        # Guard against p=0 (exact floating-point zero from rng.uniform):
        # (b-p)/alpha could also be 0 when u1=1; clip to avoid log(0).
        p_safe_log = np.clip((b - p) / alpha, 1e-300, None)
        p_safe_pow = np.where(big, 1.0, np.maximum(p, 1e-300))  # avoid 0^(1/alpha)
        x = np.where(big, -np.log(p_safe_log), p_safe_pow ** (1.0 / alpha))
        # Suppress divide/overflow that arise from 0^(alpha-1) in the vectorised
        # numpy expressions before the np.where mask is applied.
        # Note: for the small branch (big=False), x can underflow to 0.0 due to
        # p_safe_pow^(1/alpha) with very large 1/alpha.  Those x=0 samples are
        # valid (Gamma sample ≈ 0) and should NOT be rejected; exp(-0)=1 so they
        # always satisfy the acceptance criterion.  Only the big branch (x from
        # -log(...)) must be strictly positive; guard it separately.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            accept = np.where(
                big,
                u2 <= np.clip(x, 0, 500) ** (alpha - 1.0),
                u2 <= np.exp(-np.clip(x, 0, 500)),
            ) & (~big | (x > 0))  # for big branch only: reject x<=0
        accepted = x[accept]
        take = min(len(accepted), n - idx)
        result[idx : idx + take] = accepted[:take]
        idx += take
    return result


def sample_ahrens_dieter(
    delta: float,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate central chi-square samples with df=delta using the Ahrens-Dieter
    GS algorithm (benchmark method for small delta).

    chi2(delta) = 2 * Gamma(delta/2, 1)

    Parameters
    ----------
    delta : float
        Degrees of freedom (should be in (0, 2) for the GS algorithm; for
        delta >= 2 we fall back to numpy's standard gamma sampler).
    n_samples : int
    seed : int, optional

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    alpha = 0.5 * delta
    if alpha >= 1.0:
        # For alpha >= 1 use numpy's built-in gamma sampler (Marsaglia-Tsang)
        return 2.0 * rng.gamma(shape=alpha, scale=1.0, size=n_samples)
    # Decompose: Gamma(alpha) where alpha in (0,1) via GS
    return 2.0 * _ahrens_dieter_gamma(alpha, n_samples, rng)


# ---------------------------------------------------------------------------
# Direct inversion central chi-square sampler
# ---------------------------------------------------------------------------


def sample_central_chi2_direct(
    delta: float,
    n_samples: int,
    coeff_blocks: List[Dict],
    region_u_bounds: List[Tuple[float, float]],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate central chi-square samples with df=delta using the direct
    inversion Chebyshev approximation F_delta^{-1}(U), U~Uniform(0,1).

    Parameters
    ----------
    delta : float
        Degrees of freedom.
    n_samples : int
    coeff_blocks : list of 3 dicts
        Precomputed Chebyshev coefficient blocks for the 3 regions.
    region_u_bounds : list of 3 (u_lo, u_hi) tuples
    seed : int, optional

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, U_MAX, size=n_samples)
    return evaluate_inverse_chi2_batch(delta, u, coeff_blocks, region_u_bounds)


def sample_central_chi2_scipy(
    delta: float,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Reference central chi-square sampler using scipy (for validation).

    Parameters
    ----------
    delta : float
    n_samples : int
    seed : int, optional
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, U_MAX, size=n_samples)
    return stats.chi2.ppf(u, df=delta)


# ---------------------------------------------------------------------------
# Non-central chi-square sampler
# ---------------------------------------------------------------------------


def _sample_central_chi2_for_noncentral(
    delta: float,
    n: int,
    rng: np.random.Generator,
    coeff_blocks: Optional[List[Dict]] = None,
    region_u_bounds: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Internal helper: sample central chi-square for use in the non-central
    decomposition. Uses Chebyshev direct inversion if blocks are provided,
    otherwise numpy's built-in.
    """
    if coeff_blocks is not None and region_u_bounds is not None:
        u = rng.uniform(0.0, U_MAX, size=n)
        return evaluate_inverse_chi2_batch(delta, u, coeff_blocks, region_u_bounds)
    else:
        # Fall back: use numpy's noncentral_chisquare with nc=0 or standard gamma
        alpha = 0.5 * delta
        return 2.0 * rng.gamma(shape=alpha, scale=1.0, size=n)


def _poisson_chi2_sum(n_total: int, lam_half: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample chi2_{2N} = -2 * sum_{i=1}^N log(U_i) where N ~ Poisson(lam_half).

    This equals 2 * Gamma(N, 1) = sum of N exponential(2) variates.
    Vectorised: draw N_i ~ Poisson for each path, then accumulate.

    Parameters
    ----------
    n_total : int
        Number of paths.
    lam_half : float
        Mean of the Poisson distribution (= lambda/2).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of shape (n_total,)
    """
    N_draws = rng.poisson(lam_half, size=n_total)
    result = np.zeros(n_total)
    # Group by Poisson count for efficiency
    unique_n, inverse = np.unique(N_draws, return_inverse=True)
    for k, uk in enumerate(unique_n):
        if uk == 0:
            continue
        mask = inverse == k
        cnt = mask.sum()
        # chi2_{2N} = sum of uk exponential(2) = 2 * Gamma(uk, 1)
        result[mask] = 2.0 * rng.gamma(shape=float(uk), scale=1.0, size=cnt)
    return result


def sample_noncentral_chi2(
    delta: float,
    lam: float,
    n_samples: int,
    coeff_blocks: Optional[List[Dict]] = None,
    region_u_bounds: Optional[List[Tuple[float, float]]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate non-central chi-square samples chi2_{delta}(lambda) using the
    decomposition described in the paper.

    For lambda <= 10:
        Z = chi2_delta (central) + chi2_{2N},  N ~ Poisson(lambda/2)
        chi2_{2N} = -2 * sum_{i=1}^N log(U_i)

    For lambda > 10:
        Recursive decomposition: split off Poisson mean 5 (= lambda_split/2)
        until residual <= 10, then apply the lambda <= 10 branch.

    Parameters
    ----------
    delta : float
        Degrees of freedom.
    lam : float
        Non-centrality parameter.
    n_samples : int
    coeff_blocks : list of 3 dicts, optional
        Chebyshev blocks for central chi-square inversion.
    region_u_bounds : list of 3 tuples, optional
    seed : int, optional

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    if lam < 0:
        raise ValueError(f"Non-centrality lam must be >= 0; got {lam}")

    # Handle lambda > 10 by recursive Poisson splitting
    # Each split peels off lam_split = 10 from the total non-centrality
    # Total = sum of splits + residual
    accumulated_poisson = np.zeros(n_samples)
    residual_lam = lam

    LAM_SPLIT = 10.0
    while residual_lam > LAM_SPLIT:
        lam_piece = min(LAM_SPLIT, residual_lam)
        accumulated_poisson += _poisson_chi2_sum(n_samples, lam_piece / 2.0, rng)
        residual_lam -= lam_piece

    # Residual branch (lam <= 10): central + one Poisson component
    central = _sample_central_chi2_for_noncentral(
        delta, n_samples, rng, coeff_blocks, region_u_bounds
    )
    poisson_part = _poisson_chi2_sum(n_samples, residual_lam / 2.0, rng)

    return central + poisson_part + accumulated_poisson


# ---------------------------------------------------------------------------
# Theoretical moments of non-central chi-square
# ---------------------------------------------------------------------------


def theoretical_noncentral_moments(
    delta: float, lam: float, max_order: int = 10
) -> np.ndarray:
    """
    Compute the first max_order raw moments E[X^k] for k=1,...,max_order
    of X ~ ncx2(df=delta, nc=lam).

    Uses the moment generating function approach:
        E[X^k] = d^k/dt^k M(t)|_{t=0}
    where M(t) = (1-2t)^{-delta/2} * exp(lam*t/(1-2t))

    For k <= 4, scipy.stats.ncx2 provides moments; for higher orders we use
    the raw moment recursion derived from the characteristic function.

    The recursion (raw moments): if mu_k = E[X^k],
        mu_1 = delta + lam
        Recursion from cumulants:
            kappa_r = 2^{r-1} * (r-1)! * (delta + r*lam)
        mu_k from kappa via standard moment-cumulant relations.

    Parameters
    ----------
    delta : float
    lam : float
    max_order : int

    Returns
    -------
    np.ndarray of shape (max_order,), moments mu_1 ... mu_{max_order}.
    """
    # Cumulants of ncx2(delta, lam): kappa_r = 2^{r-1}*(r-1)!*(delta + r*lam)
    kappa = np.zeros(max_order + 1)  # 1-indexed: kappa[r] for r=1..max_order
    for r in range(1, max_order + 1):
        kappa[r] = (2.0 ** (r - 1)) * float(math.factorial(r - 1)) * (delta + r * lam)

    # Convert cumulants to raw moments via recursive formula
    # mu_n = kappa_n + sum_{j=1}^{n-1} C(n-1,j-1) * kappa_j * mu_{n-j}
    mu = np.zeros(max_order + 1)  # mu[0]=1 (0th moment = 1)
    mu[0] = 1.0
    from math import comb
    for n in range(1, max_order + 1):
        s = kappa[n]
        for j in range(1, n):
            s += comb(n - 1, j - 1) * kappa[j] * mu[n - j]
        mu[n] = s

    return mu[1:]  # return mu_1 ... mu_{max_order}


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def time_sampling(
    method: str,
    delta: float,
    n_samples: int,
    coeff_blocks: Optional[List[Dict]] = None,
    region_u_bounds: Optional[List[Tuple[float, float]]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Time the generation of n_samples from the chosen method.

    Parameters
    ----------
    method : str
        One of 'direct', 'ahrens_dieter', 'scipy'.
    delta : float
    n_samples : int
    coeff_blocks, region_u_bounds : optional (required for 'direct')
    seed : int

    Returns
    -------
    (samples, elapsed_seconds)
    """
    # Warm-up run (excluded from timing)
    n_warmup = min(1000, n_samples // 100)
    if method == "direct":
        sample_central_chi2_direct(delta, n_warmup, coeff_blocks, region_u_bounds, seed=seed)
    elif method == "ahrens_dieter":
        sample_ahrens_dieter(delta, n_warmup, seed=seed)
    elif method == "scipy":
        sample_central_chi2_scipy(delta, n_warmup, seed=seed)

    t0 = time.perf_counter()
    if method == "direct":
        samples = sample_central_chi2_direct(
            delta, n_samples, coeff_blocks, region_u_bounds, seed=seed + 1
        )
    elif method == "ahrens_dieter":
        samples = sample_ahrens_dieter(delta, n_samples, seed=seed + 1)
    elif method == "scipy":
        samples = sample_central_chi2_scipy(delta, n_samples, seed=seed + 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = time.perf_counter() - t0
    return samples, elapsed


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_sampler_blocks(
    coeff_dir: str,
    delta_intervals: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Tuple[List[Dict], List[Tuple[float, float]]]]:
    """
    Load precomputed Chebyshev coefficient blocks and compute region u-bounds.

    Returns
    -------
    dict mapping str(interval_key) -> (coeff_blocks, region_u_bounds)
    """
    from src.chebyshev import TRUNCATION_ORDERS, load_all_coefficient_blocks

    if delta_intervals is None:
        delta_intervals = list(TRUNCATION_ORDERS.keys())

    all_raw = load_all_coefficient_blocks(coeff_dir, delta_intervals)
    result = {}
    for delta_min, delta_max in delta_intervals:
        key = f"delta_{delta_min}_{delta_max}"
        blocks = all_raw[key]
        region_u_bounds = get_region_u_bounds(delta_min, delta_max)
        result[key] = (blocks, region_u_bounds)
    return result
