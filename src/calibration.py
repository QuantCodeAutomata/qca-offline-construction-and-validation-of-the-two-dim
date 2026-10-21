"""
Levenberg-Marquardt calibration of the CIR model to European or Asian put prices.

Parameter vector: theta = (delta, b, c)
    delta = 4*a/c^2  (degrees of freedom, > 0)
    b < 0            (mean-reversion speed)
    c > 0            (volatility)
    a = delta*c^2/4  (derived)

Gradients are computed via central finite differences applied to the
exact analytical pricing formula (European) or Monte Carlo estimator (Asian).
The LM update uses the J^T J + mu*I normal-equations convention.
"""

import numpy as np
from pricing import exact_european_cir_put_price, mc_european_cir_put, mc_asian_cir_put


# ---------------------------------------------------------------------------
# Gradient helpers (finite differences)
# ---------------------------------------------------------------------------

_DEFAULT_H = 1e-5   # central-difference step


def _fd_grad(price_fn, theta, h=_DEFAULT_H):
    """
    Central finite-difference gradient of price_fn(theta) with respect to theta.

    Parameters
    ----------
    price_fn : callable  theta -> scalar price
    theta    : ndarray, shape (p,)
    h        : float     perturbation step

    Returns
    -------
    grad : ndarray, shape (p,)
    """
    grad = np.empty_like(theta)
    for j in range(len(theta)):
        th_p = theta.copy(); th_p[j] += h
        th_m = theta.copy(); th_m[j] -= h
        grad[j] = (price_fn(th_p) - price_fn(th_m)) / (2.0 * h)
    return grad


def _theta_to_abc(theta):
    """Convert (delta, b, c) -> (a, b, c)."""
    delta, b, c = theta
    a = delta * c**2 / 4.0
    return a, b, c


# ---------------------------------------------------------------------------
# Price + gradient for European put
# ---------------------------------------------------------------------------

def cir_european_price_and_grad(theta, K, T, x0, n_paths=None, seed=None):
    """
    European put price and gradient with respect to theta = (delta, b, c).

    Uses the exact analytical formula for the price and central finite
    differences for the gradient.

    Parameters
    ----------
    theta  : array-like (3,)   [delta, b, c]
    K, T   : float
    x0     : float
    n_paths: ignored (kept for API consistency with Asian variant)
    seed   : ignored

    Returns
    -------
    price : float
    grad  : ndarray, shape (3,)
    """
    theta = np.asarray(theta, dtype=float)

    def _price(th):
        a, b, c = _theta_to_abc(th)
        return exact_european_cir_put_price(x0, T, K, a, b, c)

    price = _price(theta)
    grad = _fd_grad(_price, theta)
    return price, grad


# ---------------------------------------------------------------------------
# Price + gradient for Asian put
# ---------------------------------------------------------------------------

def cir_asian_price_and_grad(theta, K, T, M_fixings, x0, n_paths=50000, seed=None):
    """
    Asian put price and gradient with respect to theta = (delta, b, c).

    Uses Monte Carlo (exact CIR sampling) for the price and central finite
    differences (re-seeded independently) for the gradient.

    Parameters
    ----------
    theta    : array-like (3,)   [delta, b, c]
    K, T     : float
    M_fixings: int
    x0       : float
    n_paths  : int   (default 50 000)
    seed     : int or None

    Returns
    -------
    price : float
    grad  : ndarray, shape (3,)
    """
    theta = np.asarray(theta, dtype=float)

    # Fixed seed for reproducible gradient (same paths for each perturbation)
    base_seed = seed if seed is not None else 0

    def _price(th):
        a, b, c = _theta_to_abc(th)
        p, _ = mc_asian_cir_put(x0, T, K, a, b, c, M_fixings, n_paths,
                                 method='exact', seed=base_seed)
        return p

    price = _price(theta)
    grad = _fd_grad(_price, theta)
    return price, grad


# ---------------------------------------------------------------------------
# Synthetic target generation
# ---------------------------------------------------------------------------

def generate_synthetic_targets(theta_true, strikes, maturities, x0,
                                option_type='european', M_fixings=10):
    """
    Generate exact synthetic put prices for a grid of (strike, maturity).

    Parameters
    ----------
    theta_true  : array-like (3,)   [delta, b, c]
    strikes     : array-like, shape (n_K,)
    maturities  : array-like, shape (n_T,)
    x0          : float
    option_type : {'european', 'asian'}
    M_fixings   : int   used only for 'asian'

    Returns
    -------
    prices : ndarray, shape (n_K * n_T,)
        Flattened row-major (strike varies fastest).
    strike_grid    : ndarray, shape (n_K * n_T,)
    maturity_grid  : ndarray, shape (n_K * n_T,)
    """
    theta_true = np.asarray(theta_true, dtype=float)
    a, b, c = _theta_to_abc(theta_true)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    n_K, n_T = len(strikes), len(maturities)
    prices = np.empty(n_K * n_T)
    sk_grid = np.empty(n_K * n_T)
    mat_grid = np.empty(n_K * n_T)

    idx = 0
    for T in maturities:
        for K in strikes:
            if option_type == 'european':
                prices[idx] = exact_european_cir_put_price(x0, T, K, a, b, c)
            elif option_type == 'asian':
                # Use large n_paths for near-exact prices
                p, _ = mc_asian_cir_put(x0, T, K, a, b, c, M_fixings,
                                         n_paths=200_000, method='exact', seed=42)
                prices[idx] = p
            else:
                raise ValueError(f"Unknown option_type '{option_type}'.")
            sk_grid[idx] = K
            mat_grid[idx] = T
            idx += 1

    return prices, sk_grid, mat_grid


# ---------------------------------------------------------------------------
# Levenberg-Marquardt calibration
# ---------------------------------------------------------------------------

def _price_vec(theta, strike_grid, maturity_grid, x0,
               option_type, M_fixings, n_paths, seed):
    """Compute price vector for all (strike, maturity) pairs at theta."""
    a, b, c = _theta_to_abc(theta)
    n = len(strike_grid)
    prices = np.empty(n)
    for i in range(n):
        K = strike_grid[i]
        T = maturity_grid[i]
        if option_type == 'european':
            prices[i] = exact_european_cir_put_price(x0, T, K, a, b, c)
        else:
            price_i, _ = mc_asian_cir_put(x0, T, K, a, b, c, M_fixings,
                                           n_paths, method='exact', seed=seed)
            prices[i] = price_i
    return prices


def _build_jacobian(theta, current_prices, strike_grid, maturity_grid, x0,
                    option_type, M_fixings, n_paths, seed, h=_DEFAULT_H):
    """
    Build Jacobian J via central finite differences.

    J_ij = d price_i / d theta_j   (not the residual Jacobian)

    Reuses current_prices (already evaluated at theta) to save computation.
    """
    n = len(current_prices)
    p = len(theta)
    J = np.empty((n, p))
    for j in range(p):
        th_p = theta.copy(); th_p[j] += h
        th_m = theta.copy(); th_m[j] -= h
        J[:, j] = (
            _price_vec(th_p, strike_grid, maturity_grid, x0,
                       option_type, M_fixings, n_paths, seed)
            - _price_vec(th_m, strike_grid, maturity_grid, x0,
                         option_type, M_fixings, n_paths, seed)
        ) / (2.0 * h)
    return J


def _project_theta(theta):
    """Enforce parameter constraints: delta > 0, b < 0, c > 0."""
    theta = theta.copy()
    theta[0] = max(theta[0], 1e-4)    # delta > 0
    theta[1] = min(theta[1], -1e-4)   # b < 0
    theta[2] = max(theta[2], 1e-4)    # c > 0
    return theta


def lm_calibrate(theta_init, target_prices, strikes, maturities, x0,
                 option_type='european', M_fixings=10,
                 max_iter=50, tol_r=1e-4, tol_step=1e-6,
                 mu_init=1e-2, n_paths=50_000, seed=42):
    """
    Levenberg-Marquardt calibration of CIR parameters.

    Minimises  sum_i (price_i(theta) - target_i)^2
    using the  J^T J + mu*I  normal-equation update.

    Efficiency note: the Jacobian J is recomputed only when a step is accepted.
    On rejection the damping factor mu is increased and J is reused, saving the
    expensive finite-difference pricing calls.

    Parameters
    ----------
    theta_init    : array-like (3,)   [delta, b, c] initial guess
    target_prices : array-like (n,)   target put prices
    strikes       : array-like (n_K,)
    maturities    : array-like (n_T,)
    x0            : float
    option_type   : {'european', 'asian'}
    M_fixings     : int
    max_iter      : int
    tol_r         : float   stop when ||r|| < tol_r
    tol_step      : float   stop when ||delta_theta|| < tol_step
    mu_init       : float   initial damping factor
    n_paths       : int
    seed          : int or None

    Returns
    -------
    result : dict with keys
        'theta'       : final parameter vector
        'residuals'   : history of ||r|| at each iteration
        'params'      : history of theta at each iteration
        'n_iter'      : number of iterations performed
        'converged'   : bool
        'final_r'     : final residual vector
    """
    theta = _project_theta(np.asarray(theta_init, dtype=float))
    target_prices = np.asarray(target_prices, dtype=float)
    strike_grid, maturity_grid = np.meshgrid(
        np.asarray(strikes), np.asarray(maturities))
    strike_grid = strike_grid.ravel()
    maturity_grid = maturity_grid.ravel()

    _pv_kw = dict(strike_grid=strike_grid, maturity_grid=maturity_grid,
                  x0=x0, option_type=option_type, M_fixings=M_fixings,
                  n_paths=n_paths, seed=seed)

    mu = float(mu_init)
    residual_history = []
    param_history = [theta.copy()]
    converged = False

    # Initial evaluation
    cur_prices = _price_vec(theta, **_pv_kw)
    r = cur_prices - target_prices
    J = _build_jacobian(theta, cur_prices, **_pv_kw)
    norm_r = np.linalg.norm(r)
    residual_history.append(norm_r)

    for k in range(max_iter):
        if norm_r < tol_r:
            converged = True
            break

        # LM normal equations: (J^T J + mu*I) delta = -J^T r
        JtJ = J.T @ J
        Jtr = J.T @ r
        A = JtJ + mu * np.eye(len(theta))
        try:
            delta_theta = np.linalg.solve(A, -Jtr)
        except np.linalg.LinAlgError:
            mu = min(mu * 10.0, 1e12)
            continue

        theta_new = _project_theta(theta + delta_theta)
        step_norm = np.linalg.norm(theta_new - theta)

        # Evaluate only the residual at theta_new (cheap: n_instruments calls)
        new_prices = _price_vec(theta_new, **_pv_kw)
        r_new = new_prices - target_prices
        norm_r_new = np.linalg.norm(r_new)

        if norm_r_new < norm_r:
            # Accept: update theta and recompute Jacobian at new point
            theta, r, cur_prices = theta_new, r_new, new_prices
            norm_r = norm_r_new
            mu = max(mu / 10.0, 1e-12)
            J = _build_jacobian(theta, cur_prices, **_pv_kw)
        else:
            # Reject: increase damping, reuse J (no extra evaluations)
            mu = min(mu * 10.0, 1e12)

        residual_history.append(norm_r)
        param_history.append(theta.copy())

        if step_norm < tol_step:
            converged = True
            break

    return {
        'theta': theta,
        'residuals': residual_history,
        'params': param_history,
        'n_iter': len(residual_history) - 1,
        'converged': converged,
        'final_r': r,
    }


# ---------------------------------------------------------------------------
# Multi-trial calibration study
# ---------------------------------------------------------------------------

def run_calibration_study(theta_true, strikes, maturities, x0,
                          n_trials=100, option_type='european', M_fixings=10,
                          perturb_frac=0.2, n_paths=50_000, seed=None):
    """
    Run multiple randomised LM calibration trials.

    Each trial starts from theta_true perturbed by Uniform(-perturb_frac,
    +perturb_frac) * |theta_true| in each component.

    Parameters
    ----------
    theta_true   : array-like (3,)
    strikes      : array-like
    maturities   : array-like
    x0           : float
    n_trials     : int
    option_type  : {'european', 'asian'}
    M_fixings    : int
    perturb_frac : float   fractional perturbation of theta_true
    n_paths      : int
    seed         : int or None

    Returns
    -------
    summary : dict with keys
        'theta_errors'   : ndarray (n_trials, 3) absolute parameter errors
        'residual_norms' : ndarray (n_trials,)   final ||r||
        'n_iters'        : ndarray (n_trials,)   iterations used
        'converged'      : ndarray (n_trials,)   bool
        'mean_error'     : ndarray (3,)
        'std_error'      : ndarray (3,)
        'mean_residual'  : float
        'mean_n_iter'    : float
        'conv_rate'      : float
    """
    theta_true = np.asarray(theta_true, dtype=float)
    rng = np.random.default_rng(seed)

    # Generate target prices once
    target_prices, _, _ = generate_synthetic_targets(
        theta_true, strikes, maturities, x0,
        option_type=option_type, M_fixings=M_fixings)

    theta_errors = np.empty((n_trials, 3))
    residual_norms = np.empty(n_trials)
    n_iters = np.empty(n_trials, dtype=int)
    conv_flags = np.empty(n_trials, dtype=bool)

    for trial in range(n_trials):
        # Random perturbation
        perturb = 1.0 + rng.uniform(-perturb_frac, perturb_frac, size=3)
        theta_init = _project_theta(theta_true * perturb)

        lm_seed = int(rng.integers(0, 2**31))
        result = lm_calibrate(
            theta_init, target_prices, strikes, maturities, x0,
            option_type=option_type, M_fixings=M_fixings,
            n_paths=n_paths, seed=lm_seed)

        theta_errors[trial] = np.abs(result['theta'] - theta_true)
        residual_norms[trial] = np.linalg.norm(result['final_r'])
        n_iters[trial] = result['n_iter']
        conv_flags[trial] = result['converged']

    return {
        'theta_errors': theta_errors,
        'residual_norms': residual_norms,
        'n_iters': n_iters,
        'converged': conv_flags,
        'mean_error': theta_errors.mean(axis=0),
        'std_error': theta_errors.std(axis=0),
        'mean_residual': float(residual_norms.mean()),
        'mean_n_iter': float(n_iters.mean()),
        'conv_rate': float(conv_flags.mean()),
    }
