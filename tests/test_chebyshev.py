"""
Unit tests for src/chebyshev.py.

Tests verify:
- Chebyshev polynomial recurrence correctness
- Clenshaw vs direct summation agreement to machine precision
- Coefficient symmetry and basic sanity checks
- Region transform monotonicity
- xi-to-x range validity
- Delta mapping correctness
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chebyshev import (
    U_MAX,
    chebyshev_T,
    clenshaw_1d,
    clenshaw_2d,
    clenshaw_2d_array,
    compute_chebyshev_coefficients,
    delta_to_y,
    derivative_coefficients_x,
    derivative_coefficients_y,
    direct_sum_2d,
    evaluate_inverse_chi2_batch,
    get_region_u_bounds,
    get_xi_bounds,
    load_coefficients,
    save_coefficients,
    u_to_xi,
    xi_to_x,
    y_to_delta,
)


# ---------------------------------------------------------------------------
# Chebyshev polynomial tests
# ---------------------------------------------------------------------------


def test_chebyshev_T0():
    """T_0(x) = 1 for all x."""
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(chebyshev_T(0, x), np.ones(5))


def test_chebyshev_T1():
    """T_1(x) = x."""
    x = np.linspace(-1, 1, 50)
    np.testing.assert_allclose(chebyshev_T(1, x), x)


def test_chebyshev_T2():
    """T_2(x) = 2x^2 - 1."""
    x = np.linspace(-1, 1, 50)
    expected = 2.0 * x**2 - 1.0
    np.testing.assert_allclose(chebyshev_T(2, x), expected, atol=1e-14)


def test_chebyshev_recurrence():
    """Verify T_n(cos(theta)) = cos(n*theta)."""
    thetas = np.linspace(0.01, np.pi - 0.01, 100)
    x = np.cos(thetas)
    for n in range(0, 8):
        Tn = chebyshev_T(n, x)
        expected = np.cos(n * thetas)
        np.testing.assert_allclose(Tn, expected, atol=1e-13,
                                   err_msg=f"T_{n} recurrence failed")


def test_chebyshev_extrema():
    """T_n(1) = 1 and T_n(-1) = (-1)^n."""
    for n in range(10):
        assert abs(chebyshev_T(n, np.array([1.0]))[0] - 1.0) < 1e-14
        assert abs(chebyshev_T(n, np.array([-1.0]))[0] - (-1.0)**n) < 1e-14


# ---------------------------------------------------------------------------
# Delta mapping tests
# ---------------------------------------------------------------------------


def test_delta_to_y_range():
    """y = delta_to_y(delta) should be in [-1, 1] for delta in [dmin, dmax]."""
    for delta_min, delta_max in [(0.1, 0.2), (0.01, 0.02), (0.001, 0.002)]:
        deltas = np.linspace(delta_min, delta_max, 100)
        y = delta_to_y(deltas, delta_min, delta_max)
        assert y.min() >= -1.0 - 1e-14
        assert y.max() <= 1.0 + 1e-14


def test_delta_to_y_endpoints():
    """y(delta_min) = -1 and y(delta_max) = +1."""
    delta_min, delta_max = 0.1, 0.2
    assert abs(delta_to_y(np.array([delta_min]), delta_min, delta_max)[0] - (-1.0)) < 1e-14
    assert abs(delta_to_y(np.array([delta_max]), delta_min, delta_max)[0] - 1.0) < 1e-14


def test_y_to_delta_inverse():
    """y_to_delta is the inverse of delta_to_y."""
    delta_min, delta_max = 0.1, 0.2
    deltas = np.linspace(delta_min, delta_max, 50)
    y = delta_to_y(deltas, delta_min, delta_max)
    recovered = y_to_delta(y, delta_min, delta_max)
    np.testing.assert_allclose(recovered, deltas, atol=1e-14)


# ---------------------------------------------------------------------------
# Region u-bounds and transform tests
# ---------------------------------------------------------------------------


def test_region_u_bounds_coverage():
    """The three regions partition [0, U_MAX]."""
    for delta_min, delta_max in [(0.1, 0.2), (0.01, 0.02)]:
        bounds = get_region_u_bounds(delta_min, delta_max)
        # Regions cover [0, U_MAX] without overlap
        assert bounds[0][0] == 0.0
        assert abs(bounds[0][1] - bounds[1][0]) < 1e-12  # R1 end = R2 start
        assert abs(bounds[1][1] - bounds[2][0]) < 1e-12  # R2 end = R3 start
        assert bounds[2][1] <= U_MAX + 1e-12


def test_u_to_xi_region1_identity():
    """Region 1 transform is the identity."""
    u = np.linspace(0.01, 0.79, 50)
    xi = u_to_xi(u, region=1, delta=0.15)
    np.testing.assert_allclose(xi, u, atol=1e-14)


def test_u_to_xi_region2_monotone():
    """Region 2 transform is monotone in u."""
    u = np.linspace(0.8, 0.94, 50)
    xi = u_to_xi(u, region=2, delta=0.15)
    # xi = log((1-u)*G) is decreasing in u (since 1-u decreases)
    assert np.all(np.diff(xi) < 0)


def test_u_to_xi_region3_monotone():
    """Region 3 transform is monotone in u (should be increasing)."""
    # Only valid when (1-u)*Gamma(delta/2) < 1
    delta = 0.15
    from scipy.special import gamma
    u_crit = 1.0 - 1.0 / gamma(0.5 * delta)
    u = np.linspace(max(0.95, u_crit + 0.01), U_MAX - 1e-5, 20)
    xi = u_to_xi(u, region=3, delta=delta)
    assert np.all(np.isfinite(xi))
    # xi increases with u in Region 3
    assert np.all(np.diff(xi) > 0)


def test_xi_to_x_range():
    """xi_to_x maps [xi_min, xi_max] to [-1, 1]."""
    xi_min, xi_max = -2.0, 3.0
    xi = np.linspace(xi_min, xi_max, 100)
    x = xi_to_x(xi, xi_min, xi_max)
    assert abs(x[0] - (-1.0)) < 1e-14
    assert abs(x[-1] - 1.0) < 1e-14
    assert x.min() >= -1.0 - 1e-14
    assert x.max() <= 1.0 + 1e-14


# ---------------------------------------------------------------------------
# Clenshaw recurrence tests
# ---------------------------------------------------------------------------


def test_clenshaw_1d_constant():
    """Clenshaw for constant series c=[c0] returns c0."""
    for c0 in [0.0, 1.0, -3.14]:
        result = clenshaw_1d(0.5, np.array([c0]))
        assert abs(result - c0) < 1e-14


def test_clenshaw_1d_linear():
    """Clenshaw for c=[0,1] evaluates T_1(x) = x."""
    for x in [-0.9, 0.0, 0.7]:
        result = clenshaw_1d(x, np.array([0.0, 1.0]))
        assert abs(result - x) < 1e-14


def test_clenshaw_vs_direct_small():
    """Clenshaw and direct summation agree to machine precision on a small matrix."""
    rng = np.random.default_rng(1)
    C = rng.standard_normal((4, 5))  # (M+1)x(N+1)
    xs = rng.uniform(-0.99, 0.99, 30)
    ys = rng.uniform(-0.99, 0.99, 30)
    for x, y in zip(xs, ys):
        val_c = clenshaw_2d(x, y, C)
        val_d = direct_sum_2d(x, y, C)
        assert abs(val_c - val_d) < 1e-12, (
            f"Clenshaw={val_c:.10e} direct={val_d:.10e} diff={abs(val_c-val_d):.2e}"
        )


def test_clenshaw_vs_direct_larger():
    """Clenshaw and direct summation agree on larger random matrix."""
    rng = np.random.default_rng(2)
    C = rng.standard_normal((10, 15))
    xs = rng.uniform(-0.95, 0.95, 20)
    ys = rng.uniform(-0.95, 0.95, 20)
    for x, y in zip(xs, ys):
        val_c = clenshaw_2d(x, y, C)
        val_d = direct_sum_2d(x, y, C)
        assert abs(val_c - val_d) < 1e-11


def test_clenshaw_2d_array_vs_scalar():
    """clenshaw_2d_array matches scalar clenshaw_2d on random inputs."""
    rng = np.random.default_rng(3)
    C = rng.standard_normal((5, 7))
    xs = rng.uniform(-0.99, 0.99, 15)
    ys = rng.uniform(-0.99, 0.99, 15)
    batch_results = clenshaw_2d_array(xs, ys, C)
    scalar_results = np.array([clenshaw_2d(x, y, C) for x, y in zip(xs, ys)])
    np.testing.assert_allclose(batch_results, scalar_results, atol=1e-12)


# ---------------------------------------------------------------------------
# Coefficient computation tests
# ---------------------------------------------------------------------------


def test_coefficients_small_m_n():
    """Coefficient computation runs without error for small M, N."""
    block = compute_chebyshev_coefficients(0.1, 0.2, region=1, M=3, N=5)
    C = block["coefficient_matrix"]
    assert C.shape == (4, 6)
    assert np.all(np.isfinite(C))


def test_coefficients_region2():
    """Region 2 coefficients computed without error."""
    block = compute_chebyshev_coefficients(0.1, 0.2, region=2, M=2, N=4)
    C = block["coefficient_matrix"]
    assert C.shape == (3, 5)
    assert np.all(np.isfinite(C))


def test_approximation_accuracy_region1():
    """Chebyshev approximation reproduces chi2.ppf to reasonable accuracy in Region 1."""
    delta_min, delta_max = 0.1, 0.2
    block = compute_chebyshev_coefficients(delta_min, delta_max, region=1, M=5, N=13)
    u_bounds = block["u_bounds"]
    region_u_bounds = get_region_u_bounds(delta_min, delta_max)

    delta_test = 0.15
    u_test = np.linspace(0.05, 0.75, 50)
    ref = stats.chi2.ppf(u_test, df=delta_test)
    approx = evaluate_inverse_chi2_batch(delta_test, u_test, [block, block, block],
                                         region_u_bounds)
    # Only check Region 1 range
    mask = (u_test >= region_u_bounds[0][0]) & (u_test < region_u_bounds[0][1])
    err = np.abs(approx[mask] - ref[mask])
    assert err.max() < 1.0, f"Max error {err.max():.3e} too large"


# ---------------------------------------------------------------------------
# Derivative coefficient tests
# ---------------------------------------------------------------------------


def test_derivative_y_zero_constant():
    """Derivative of a constant series (all c_m0 for m>0 zero) is zero."""
    C = np.zeros((5, 4))
    C[0, :] = [1.0, 2.0, 3.0, 4.0]
    dC = derivative_coefficients_y(C)
    assert np.allclose(dC, 0.0, atol=1e-14)


def test_derivative_x_zero_constant():
    """Derivative wrt x of a constant (in x) series is zero."""
    C = np.zeros((4, 5))
    C[:, 0] = [1.0, 2.0, 3.0, 4.0]
    dC = derivative_coefficients_x(C)
    assert np.allclose(dC, 0.0, atol=1e-14)


def test_derivative_y_linear():
    """d/dy of T_1(y) = 1. Coefficients c=[0,1] give derivative c=[1]."""
    C = np.zeros((2, 1))
    C[0, 0] = 0.0
    C[1, 0] = 1.0
    dC = derivative_coefficients_y(C)
    # d/dy [T_1(y)] = 1 = T_0(y), so dC[0,0] should be 1
    assert abs(dC[0, 0] - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# Save/load coefficient tests
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    """Save and load coefficient block preserves coefficient values."""
    block = compute_chebyshev_coefficients(0.1, 0.2, region=1, M=3, N=5)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name
    try:
        save_coefficients(fname, block)
        loaded = load_coefficients(fname)
        np.testing.assert_allclose(
            loaded["coefficient_matrix"],
            block["coefficient_matrix"],
            atol=1e-14,
        )
        assert loaded["M"] == block["M"]
        assert loaded["N"] == block["N"]
        assert loaded["region"] == block["region"]
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_chebyshev_T_high_degree():
    """T_20(x) at Chebyshev nodes is accurate."""
    n = 20
    thetas = np.linspace(0.1, np.pi - 0.1, 30)
    x = np.cos(thetas)
    Tn = chebyshev_T(n, x)
    expected = np.cos(n * thetas)
    np.testing.assert_allclose(Tn, expected, atol=1e-12)


def test_xi_bounds_finite():
    """get_xi_bounds returns finite values for all published intervals and regions."""
    for (delta_min, delta_max) in [(0.1, 0.2), (0.01, 0.02), (0.001, 0.002)]:
        region_u_bounds = get_region_u_bounds(delta_min, delta_max)
        for region, u_bounds in zip([1, 2, 3], region_u_bounds):
            xi_min, xi_max = get_xi_bounds(region, delta_min, delta_max, u_bounds)
            assert np.isfinite(xi_min), f"xi_min not finite for region={region}"
            assert np.isfinite(xi_max), f"xi_max not finite for region={region}"
            assert xi_max > xi_min, "xi_max must be > xi_min"
