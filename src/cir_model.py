"""
CIR model implementation: exact sampling, Full Truncation Euler (FTE), and
QE-M (Quadratic-Exponential with Martingale correction) simulation.

Model:  dX_t = (a + b*X_t) dt + c*sqrt(X_t) dW_t
Convention: b < 0 (mean-reverting), a > 0.
Long-run mean: mu_inf = -a/b.

Exact transition (Cox-Ingersoll-Ross 1985):
    X_t | X_s ~ C(t-s) * chi2(df=delta, nc=lambda)
where
    delta  = 4*a / c^2
    C(tau) = c^2*(1 - exp(b*tau)) / (-4*b)   [>0 for b<0]
    lambda = X_s * 4*(-b)*exp(b*tau) / (c^2*(1 - exp(b*tau)))

Note on chi-square sampler: In the full implementation described in the paper,
sampling would use a custom Chebyshev-based sampler (chi_square_sampler.py).
Here we use numpy's built-in noncentral chi-square generator, equivalent to
the 'direct inversion' baseline.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def cir_delta(a, c):
    """Return degrees-of-freedom parameter delta = 4*a / c^2."""
    return 4.0 * a / c**2


def cir_C(t, b, c):
    """
    Return the squared-Bessel time-change C(t) = c^2*(1 - exp(b*t)) / (-4*b).

    With b < 0 and t > 0 this is strictly positive and equals the scale factor
    in the exact CIR transition distribution.
    """
    return c**2 * (1.0 - np.exp(b * t)) / (-4.0 * b)


def cir_transition_params(x0, t, a, b, c):
    """
    Return (delta, scale, lam) characterising the exact CIR transition.

    X_t | X_0  ~  scale * chi2(df=delta, nc=lam)

    Parameters
    ----------
    x0 : float or ndarray
        Initial value(s).
    t : float
        Time horizon (t - s with s = 0).
    a, b, c : float
        CIR parameters (b < 0, a > 0, c > 0).

    Returns
    -------
    delta : float
    scale : float
    lam   : same shape as x0
    """
    delta = cir_delta(a, c)
    scale = cir_C(t, b, c)
    lam = np.asarray(x0) * (4.0 * (-b) * np.exp(b * t)) / (c**2 * (1.0 - np.exp(b * t)))
    return delta, scale, lam


# ---------------------------------------------------------------------------
# Exact sampling via noncentral chi-square
# ---------------------------------------------------------------------------

def sample_cir_terminal(x0, T, a, b, c, n_paths, seed=None):
    """
    Sample X_T exactly from the CIR model using the noncentral chi-square transition.

    Parameters
    ----------
    x0 : float
    T  : float
    a, b, c : float   (b < 0)
    n_paths : int
    seed : int or None

    Returns
    -------
    X_T : ndarray, shape (n_paths,)
    """
    rng = np.random.default_rng(seed)
    delta, scale, lam = cir_transition_params(x0, T, a, b, c)
    return scale * rng.noncentral_chisquare(delta, float(lam), size=n_paths)


def sample_cir_path(x0, times, a, b, c, n_paths, seed=None):
    """
    Sample CIR paths at multiple monitoring times exactly.

    Uses the Markov property: sequentially applies the exact transition from
    each monitoring time to the next.

    Parameters
    ----------
    x0 : float
    times : array-like, shape (M,)   sorted ascending
    a, b, c : float   (b < 0)
    n_paths : int
    seed : int or None

    Returns
    -------
    X : ndarray, shape (n_paths, M)
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    M = len(times)
    X = np.zeros((n_paths, M))
    X_cur = np.full(n_paths, float(x0))
    t_prev = 0.0

    for i, t in enumerate(times):
        dt = t - t_prev
        delta, scale, lam_vec = cir_transition_params(X_cur, dt, a, b, c)
        X_cur = scale * rng.noncentral_chisquare(delta, lam_vec)
        X[:, i] = X_cur
        t_prev = t

    return X


# ---------------------------------------------------------------------------
# Full Truncation Euler (FTE)
# ---------------------------------------------------------------------------

def simulate_fte(x0, T, a, b, c, n_paths, n_steps, seed=None):
    """
    Full Truncation Euler simulation of the CIR model to terminal time T.

    Scheme (Lord et al. 2010):
        X_{n+1} = X_n + (a + b*X_n)*dt + c*sqrt(max(X_n, 0))*sqrt(dt)*Z_n
    Only the diffusion coefficient is truncated; the drift uses raw X_n.

    Parameters
    ----------
    x0 : float
    T  : float
    a, b, c : float   (b < 0)
    n_paths : int
    n_steps : int     total number of Euler steps over [0, T]
    seed : int or None

    Returns
    -------
    X_T : ndarray, shape (n_paths,)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    X = np.full(n_paths, float(x0))

    for _ in range(n_steps):
        Z = rng.standard_normal(n_paths)
        X = X + (a + b * X) * dt + c * np.sqrt(np.maximum(X, 0.0)) * sqrt_dt * Z

    return X


def simulate_fte_path(x0, times, a, b, c, n_paths, n_steps_per_interval, seed=None):
    """
    FTE simulation of CIR along monitoring times.

    Parameters
    ----------
    x0 : float
    times : array-like, shape (M,)   sorted ascending
    a, b, c : float
    n_paths : int
    n_steps_per_interval : int   sub-steps between consecutive monitoring dates
    seed : int or None

    Returns
    -------
    X : ndarray, shape (n_paths, M)
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    M = len(times)
    X_out = np.zeros((n_paths, M))
    X = np.full(n_paths, float(x0))
    t_prev = 0.0

    for i, t in enumerate(times):
        dt_interval = t - t_prev
        n_sub = int(n_steps_per_interval)
        dt = dt_interval / n_sub
        sqrt_dt = np.sqrt(dt)
        for _ in range(n_sub):
            Z = rng.standard_normal(n_paths)
            X = X + (a + b * X) * dt + c * np.sqrt(np.maximum(X, 0.0)) * sqrt_dt * Z
        X_out[:, i] = X
        t_prev = t

    return X_out


# ---------------------------------------------------------------------------
# QE-M (Quadratic-Exponential with Martingale correction)
# ---------------------------------------------------------------------------

def _qe_step(X, dt, a, b, c, rng, psi_c=1.5):
    """
    One step of the Quadratic-Exponential (QE) scheme (Andersen 2008).

    Algorithm
    ---------
    1. Compute exact conditional mean m and variance v using the noncentral
       chi-square parametrisation of the CIR transition.
    2. psi = v / m^2.
    3. psi <= psi_c  →  quadratic (Gaussian) branch:
           b^2    = 2/psi - 1 + sqrt(2/psi)*sqrt(2/psi - 1)
           a_coef = m / (1 + b^2)
           X_new  = a_coef * (sqrt(b^2) + Z)^2,  Z ~ N(0,1)
       psi  > psi_c  →  exponential branch:
           p    = (psi - 1)/(psi + 1)
           beta = (1 - p)/m
           X_new = 0  if U <= p, else  log((1-p)/(1-U)) / beta,  U ~ U(0,1)

    Martingale correction note
    --------------------------
    In Andersen (2008) the M-correction adjusts a *log-price* process driven by
    a stochastic-volatility CIR. Since we price puts on the CIR process X directly,
    the log-price martingale correction does not apply. Moment-consistency (first
    two conditional moments) is guaranteed by the QE moment-matching construction.

    Parameters
    ----------
    X     : ndarray, shape (n_paths,)   current state (>= 0)
    dt    : float                        time step
    a, b, c : float
    rng   : numpy Generator
    psi_c : float                        branch switch threshold (default 1.5)

    Returns
    -------
    X_new : ndarray, shape (n_paths,)
    """
    # Exact conditional parameters for X_{t+dt} | X_t
    delta = cir_delta(a, c)
    scale = cir_C(dt, b, c)
    lam = np.maximum(X, 0.0) * (4.0 * (-b) * np.exp(b * dt)) / (
        c**2 * (1.0 - np.exp(b * dt))
    )

    # Exact conditional mean and variance: X ~ scale * chi2(delta, lam)
    m = scale * (delta + lam)          # E[scale*chi2(df,nc)] = scale*(df+nc)
    v = scale**2 * 2.0 * (delta + 2.0 * lam)  # Var = scale^2 * 2*(df+2*nc)

    psi = v / (m**2)
    X_new = np.empty_like(X)

    # --- Gaussian branch (psi <= psi_c) ---
    mask_g = psi <= psi_c
    if np.any(mask_g):
        psi_g = psi[mask_g]
        m_g = m[mask_g]
        b_sq = 2.0 / psi_g - 1.0 + np.sqrt(2.0 / psi_g) * np.sqrt(2.0 / psi_g - 1.0)
        a_coef = m_g / (1.0 + b_sq)
        Z = rng.standard_normal(int(np.sum(mask_g)))
        X_new[mask_g] = a_coef * (np.sqrt(b_sq) + Z) ** 2

    # --- Exponential branch (psi > psi_c) ---
    mask_e = ~mask_g
    if np.any(mask_e):
        psi_e = psi[mask_e]
        m_e = m[mask_e]
        p = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p) / m_e
        U = rng.uniform(size=int(np.sum(mask_e)))
        # Inverse CDF of mixture: Dirac(0) w.p. p, Exp(beta) w.p. (1-p)
        safe_denom = np.maximum(1.0 - U, 1e-300)
        X_new[mask_e] = np.where(U <= p, 0.0,
                                  np.log((1.0 - p) / safe_denom) / beta)

    return X_new


def simulate_qem(x0, T, a, b, c, n_paths, n_steps, seed=None, psi_c=1.5):
    """
    QE-M simulation of CIR to terminal time T.

    See _qe_step for the full algorithm description.

    Parameters
    ----------
    x0 : float
    T  : float
    a, b, c : float   (b < 0)
    n_paths : int
    n_steps : int     total QE steps over [0, T]
    seed : int or None
    psi_c : float     branch switch threshold (default 1.5)

    Returns
    -------
    X_T : ndarray, shape (n_paths,)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    X = np.full(n_paths, float(x0))

    for _ in range(n_steps):
        X = _qe_step(X, dt, a, b, c, rng, psi_c)

    return X


def simulate_qem_path(x0, times, a, b, c, n_paths, n_steps_per_interval,
                      seed=None, psi_c=1.5):
    """
    QE-M simulation of CIR along monitoring times.

    Parameters
    ----------
    x0 : float
    times : array-like, shape (M,)   sorted ascending
    a, b, c : float
    n_paths : int
    n_steps_per_interval : int   sub-steps between consecutive monitoring dates
    seed : int or None
    psi_c : float

    Returns
    -------
    X : ndarray, shape (n_paths, M)
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    M = len(times)
    X_out = np.zeros((n_paths, M))
    X = np.full(n_paths, float(x0))
    t_prev = 0.0

    for i, t in enumerate(times):
        dt_interval = t - t_prev
        n_sub = int(n_steps_per_interval)
        dt = dt_interval / n_sub
        for _ in range(n_sub):
            X = _qe_step(X, dt, a, b, c, rng, psi_c)
        X_out[:, i] = X
        t_prev = t

    return X_out


# ---------------------------------------------------------------------------
# CIR moment formulas (analytical)
# ---------------------------------------------------------------------------

def cir_mean(x0, t, a, b):
    """Exact conditional mean E[X_t | X_0 = x0]."""
    return x0 * np.exp(b * t) + (-a / b) * (1.0 - np.exp(b * t))


def cir_variance(x0, t, a, b, c):
    """Exact conditional variance Var[X_t | X_0 = x0]."""
    eb = np.exp(b * t)
    return (x0 * c**2 * eb * (1.0 - eb) / (-b)
            + a * c**2 * (1.0 - eb)**2 / (2.0 * b**2))
