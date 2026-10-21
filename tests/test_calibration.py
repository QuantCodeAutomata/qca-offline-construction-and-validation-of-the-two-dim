"""
Unit tests for src/calibration.py.

Run with:  python -m pytest tests/test_calibration.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from calibration import (
    cir_european_price_and_grad,
    cir_asian_price_and_grad,
    generate_synthetic_targets,
    lm_calibrate,
    _theta_to_abc,
)
from pricing import exact_european_cir_put_price

# ── shared parameters ──────────────────────────────────────────────────────────
THETA_TRUE = np.array([0.18, -0.50, 1.00])    # delta, b, c
X0 = 0.09
STRIKES    = np.array([0.07, 0.09, 0.11])
MATURITIES = np.array([5.0, 10.0])


class TestThetaToAbc:
    def test_a_formula(self):
        delta, b, c = 0.18, -0.5, 1.0
        a, b_out, c_out = _theta_to_abc(np.array([delta, b, c]))
        assert a == pytest.approx(delta * c**2 / 4)
        assert b_out == b
        assert c_out == c


class TestEuropeanPriceAndGrad:
    def test_returns_price_and_grad(self):
        price, grad = cir_european_price_and_grad(THETA_TRUE, K=0.09, T=10.0, x0=X0)
        assert isinstance(price, float)
        assert grad.shape == (3,)

    def test_price_positive(self):
        price, _ = cir_european_price_and_grad(THETA_TRUE, K=0.09, T=10.0, x0=X0)
        assert price > 0

    def test_price_matches_exact_formula(self):
        K, T = 0.09, 10.0
        price_cal, _ = cir_european_price_and_grad(THETA_TRUE, K=K, T=T, x0=X0)
        a, b, c = _theta_to_abc(THETA_TRUE)
        price_ref = exact_european_cir_put_price(X0, T, K, a, b, c)
        assert price_cal == pytest.approx(price_ref, rel=1e-6)

    def test_grad_not_all_zero(self):
        _, grad = cir_european_price_and_grad(THETA_TRUE, K=0.09, T=10.0, x0=X0)
        assert not np.allclose(grad, 0.0)


class TestAsianPriceAndGrad:
    def test_returns_price_and_grad(self):
        price, grad = cir_asian_price_and_grad(
            THETA_TRUE, K=0.09, T=10.0, M_fixings=5, x0=X0,
            n_paths=10_000, seed=0)
        assert isinstance(price, float)
        assert grad.shape == (3,)

    def test_price_non_negative(self):
        price, _ = cir_asian_price_and_grad(
            THETA_TRUE, K=0.09, T=10.0, M_fixings=5, x0=X0,
            n_paths=10_000, seed=0)
        assert price >= 0


class TestGenerateSyntheticTargets:
    def test_shape(self):
        prices, sk, mat = generate_synthetic_targets(
            THETA_TRUE, STRIKES, MATURITIES, X0, option_type='european')
        n_expected = len(STRIKES) * len(MATURITIES)
        assert prices.shape == (n_expected,)
        assert sk.shape    == (n_expected,)
        assert mat.shape   == (n_expected,)

    def test_all_positive(self):
        prices, _, _ = generate_synthetic_targets(
            THETA_TRUE, STRIKES, MATURITIES, X0, option_type='european')
        assert np.all(prices > 0)

    def test_monotone_in_strike(self):
        """Higher strike → higher put price (holding maturity fixed)."""
        prices, sk, mat = generate_synthetic_targets(
            THETA_TRUE, STRIKES, MATURITIES, X0, option_type='european')
        T_fixed = MATURITIES[0]
        mask = mat == T_fixed
        sub_prices = prices[mask]
        sub_sk = sk[mask]
        idx_sorted = np.argsort(sub_sk)
        for i in range(len(sub_prices) - 1):
            assert sub_prices[idx_sorted[i+1]] >= sub_prices[idx_sorted[i]] - 1e-12

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            generate_synthetic_targets(THETA_TRUE, STRIKES, MATURITIES, X0,
                                       option_type='bermuda')


class TestLmCalibrate:
    @classmethod
    def setup_class(cls):
        cls.targets, _, _ = generate_synthetic_targets(
            THETA_TRUE, STRIKES, MATURITIES, X0, option_type='european')

    def test_from_truth_gives_low_residual(self):
        """Starting at true theta: residual should be nearly zero."""
        res = lm_calibrate(
            THETA_TRUE, self.targets, STRIKES, MATURITIES, X0,
            option_type='european', max_iter=5, n_paths=10_000, seed=0)
        assert np.linalg.norm(res['final_r']) < 1e-6

    def test_returns_required_keys(self):
        res = lm_calibrate(
            THETA_TRUE, self.targets, STRIKES, MATURITIES, X0,
            option_type='european', max_iter=3, n_paths=5_000, seed=0)
        for key in ('theta', 'residuals', 'params', 'n_iter', 'converged', 'final_r'):
            assert key in res

    def test_parameter_recovery_close_init(self):
        """From a small perturbation of the truth, LM should recover parameters."""
        theta_init = THETA_TRUE * np.array([1.05, 0.95, 1.02])
        res = lm_calibrate(
            theta_init, self.targets, STRIKES, MATURITIES, X0,
            option_type='european', max_iter=30, tol_r=1e-5,
            n_paths=10_000, seed=0)
        # Check that the recovered parameters are close to truth
        assert np.allclose(res['theta'], THETA_TRUE, atol=0.1)

    def test_residual_history_non_increasing(self):
        """LM residual should (broadly) decrease."""
        res = lm_calibrate(
            THETA_TRUE * np.array([1.1, 0.9, 1.05]),
            self.targets, STRIKES, MATURITIES, X0,
            option_type='european', max_iter=10, n_paths=5_000, seed=0)
        history = res['residuals']
        # First residual should be >= last (LM only accepts improving steps)
        assert history[0] >= history[-1] - 1e-10

    def test_theta_constraints_respected(self):
        """Output theta must satisfy delta>0, b<0, c>0 after projection."""
        theta_bad = np.array([0.18, -0.5, 1.0])   # valid start
        res = lm_calibrate(
            theta_bad, self.targets, STRIKES, MATURITIES, X0,
            option_type='european', max_iter=5, n_paths=5_000, seed=0)
        delta, b, c = res['theta']
        assert delta > 0
        assert b < 0
        assert c > 0
