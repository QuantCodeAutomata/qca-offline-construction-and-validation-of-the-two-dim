"""
Option pricing for the CIR interest-rate model.

Supported payoffs
-----------------
European put  : E[(K - X_T)^+]
Asian put     : E[(K - A_T)^+]  where A_T = (1/M) * sum_{m=1}^{M} X_{t_m}
                monitoring dates t_m = m*T/M
Basket put    : E[(K - sum_i w_i X_T^i)^+]  independent CIR assets

Note on chi-square sampler
--------------------------
The paper's exact sampler uses a Chebyshev-approximation-based noncentral
chi-square generator (chi_square_sampler.py, created separately).
Here 'direct inversion' uses numpy's built-in noncentral_chisquare, which is
statistically equivalent and allows the code to run standalone.
"""

import numpy as np
from scipy.stats import ncx2

from cir_model import (
    cir_transition_params,
    sample_cir_terminal,
    sample_cir_path,
    simulate_fte,
    simulate_fte_path,
    simulate_qem,
    simulate_qem_path,
)


# ---------------------------------------------------------------------------
# Exact European put by numerical integration
# ---------------------------------------------------------------------------

def exact_european_cir_put_price(x0, T, K, a, b, c):
    """
    Exact European put price E[(K - X_T)^+] in closed form.

    Derivation
    ----------
    X_T ~ scale * Y,  Y ~ ncx2(df=delta, nc=lam).

    E[(K - X_T)^+] = K * P(X_T <= K) - E[X_T * 1_{X_T <= K}]

    Using the noncentral chi-square moment identity
        E[Y * 1_{Y <= t}] = delta * F(t; delta+2, lam) + lam * F(t; delta+4, lam)
    where F(·; k, l) = ncx2.cdf(·; k, l), we get:

        price = K * F(K/scale, delta,   lam)
              - scale * [delta * F(K/scale, delta+2, lam)
                         + lam * F(K/scale, delta+4, lam)]

    This requires only three CDF evaluations and is numerically fast.

    Parameters
    ----------
    x0, T, K : float
    a, b, c  : float   CIR parameters (b < 0)

    Returns
    -------
    price : float
    """
    if K <= 0.0:
        return 0.0
    delta, scale, lam = cir_transition_params(x0, T, a, b, c)
    t = K / scale   # upper limit in Y-space
    cdf0 = ncx2.cdf(t, delta,     float(lam))
    cdf2 = ncx2.cdf(t, delta + 2, float(lam))
    cdf4 = ncx2.cdf(t, delta + 4, float(lam))
    price = K * cdf0 - scale * (delta * cdf2 + float(lam) * cdf4)
    return float(max(price, 0.0))


# ---------------------------------------------------------------------------
# Monte Carlo European put
# ---------------------------------------------------------------------------

def mc_european_cir_put(x0, T, K, a, b, c, n_paths,
                        method='exact', n_steps=None, seed=None):
    """
    Monte Carlo European put price E[(K - X_T)^+].

    Parameters
    ----------
    x0, T, K : float
    a, b, c  : float   (b < 0)
    n_paths  : int
    method   : {'exact', 'fte', 'qem'}
        'exact' – exact noncentral chi-square sampling (direct inversion)
        'fte'   – Full Truncation Euler with n_steps steps
        'qem'   – QE-M scheme with n_steps steps
    n_steps  : int   required for 'fte' and 'qem'
    seed     : int or None

    Returns
    -------
    price : float
    std   : float   Monte Carlo standard error
    """
    if method == 'exact':
        X_T = sample_cir_terminal(x0, T, a, b, c, n_paths, seed=seed)
    elif method == 'fte':
        if n_steps is None:
            raise ValueError("n_steps required for method='fte'")
        X_T = simulate_fte(x0, T, a, b, c, n_paths, n_steps, seed=seed)
    elif method == 'qem':
        if n_steps is None:
            raise ValueError("n_steps required for method='qem'")
        X_T = simulate_qem(x0, T, a, b, c, n_paths, n_steps, seed=seed)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'exact', 'fte', 'qem'.")

    payoffs = np.maximum(K - X_T, 0.0)
    price = float(np.mean(payoffs))
    std = float(np.std(payoffs) / np.sqrt(n_paths))
    return price, std


# ---------------------------------------------------------------------------
# Monte Carlo Asian put
# ---------------------------------------------------------------------------

def mc_asian_cir_put(x0, T, K, a, b, c, M_fixings, n_paths,
                     method='exact', n_steps=None, seed=None):
    """
    Monte Carlo Asian put price E[(K - A_T)^+].

    Monitoring dates: t_m = m*T/M for m = 1, ..., M.
    Arithmetic average: A_T = (1/M) * sum_{m=1}^{M} X_{t_m}.

    Parameters
    ----------
    x0, T, K  : float
    a, b, c   : float   (b < 0)
    M_fixings : int     number of monitoring dates
    n_paths   : int
    method    : {'exact', 'fte', 'qem'}
    n_steps   : int
        For 'fte'/'qem': sub-steps per monitoring interval.
        Step size h = (T/M_fixings) / n_steps.
    seed      : int or None

    Returns
    -------
    price : float
    std   : float   Monte Carlo standard error
    """
    times = np.linspace(T / M_fixings, T, M_fixings)

    if method == 'exact':
        X = sample_cir_path(x0, times, a, b, c, n_paths, seed=seed)
    elif method == 'fte':
        if n_steps is None:
            raise ValueError("n_steps required for method='fte'")
        X = simulate_fte_path(x0, times, a, b, c, n_paths, n_steps, seed=seed)
    elif method == 'qem':
        if n_steps is None:
            raise ValueError("n_steps required for method='qem'")
        X = simulate_qem_path(x0, times, a, b, c, n_paths, n_steps, seed=seed)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    averages = np.mean(X, axis=1)          # shape (n_paths,)
    payoffs = np.maximum(K - averages, 0.0)
    price = float(np.mean(payoffs))
    std = float(np.std(payoffs) / np.sqrt(n_paths))
    return price, std


# ---------------------------------------------------------------------------
# Monte Carlo basket put
# ---------------------------------------------------------------------------

def mc_basket_cir_put(x0_vec, T, K, weights, params_list, n_paths,
                      method='exact', n_steps=None, seed=None):
    """
    Monte Carlo basket put price E[(K - sum_i w_i X_T^i)^+].

    Each asset i follows an independent CIR process with parameters params_list[i].

    Parameters
    ----------
    x0_vec     : array-like, shape (d,)   initial values
    T          : float
    K          : float
    weights    : array-like, shape (d,)   basket weights (should sum to 1)
    params_list: list of (a_i, b_i, c_i) tuples
    n_paths    : int
    method     : {'exact', 'fte', 'qem'}
    n_steps    : int   required for 'fte'/'qem'
    seed       : int or None

    Returns
    -------
    price : float
    std   : float   Monte Carlo standard error
    """
    x0_vec = np.asarray(x0_vec, dtype=float)
    weights = np.asarray(weights, dtype=float)
    d = len(x0_vec)

    # Use a seed sequence to give each asset a reproducible independent stream
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(d)

    basket = np.zeros(n_paths)
    for i, ((a_i, b_i, c_i), x0_i, w_i, cs) in enumerate(
            zip(params_list, x0_vec, weights, child_seeds)):
        seed_i = int(cs.generate_state(1)[0])
        if method == 'exact':
            X_i = sample_cir_terminal(x0_i, T, a_i, b_i, c_i, n_paths, seed=seed_i)
        elif method == 'fte':
            if n_steps is None:
                raise ValueError("n_steps required for method='fte'")
            X_i = simulate_fte(x0_i, T, a_i, b_i, c_i, n_paths, n_steps, seed=seed_i)
        elif method == 'qem':
            if n_steps is None:
                raise ValueError("n_steps required for method='qem'")
            X_i = simulate_qem(x0_i, T, a_i, b_i, c_i, n_paths, n_steps, seed=seed_i)
        else:
            raise ValueError(f"Unknown method '{method}'.")
        basket += w_i * X_i

    payoffs = np.maximum(K - basket, 0.0)
    price = float(np.mean(payoffs))
    std = float(np.std(payoffs) / np.sqrt(n_paths))
    return price, std
