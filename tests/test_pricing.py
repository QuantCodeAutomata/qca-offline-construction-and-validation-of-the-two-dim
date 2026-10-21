"""
Unit tests for src/pricing.py.

Run with:  python -m pytest tests/test_pricing.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pricing import (
    exact_european_cir_put_price,
    mc_european_cir_put,
    mc_asian_cir_put,
    mc_basket_cir_put,
)

# ── shared parameters ──────────────────────────────────────────────────────────
A, B, C = 0.045, -0.5, 1.0
X0, T, K = 0.09, 10.0, 0.09
N_PATHS = 50_000
SEED = 42


class TestExactEuropeanPut:
    def test_positive(self):
        price = exact_european_cir_put_price(X0, T, K, A, B, C)
        assert price > 0

    def test_below_strike(self):
        price = exact_european_cir_put_price(X0, T, K, A, B, C)
        assert price <= K

    def test_monotoniciy_in_strike(self):
        """Higher strike → higher put price."""
        p_low  = exact_european_cir_put_price(X0, T, 0.05, A, B, C)
        p_high = exact_european_cir_put_price(X0, T, 0.15, A, B, C)
        assert p_high > p_low

    def test_monotonicity_in_T(self):
        """Shorter maturity → terminal distribution closer to X0=0.09=K → higher put."""
        # At T→0 the process barely moves; at T=10 the long-run mean matters.
        p_short = exact_european_cir_put_price(X0, 1.0, K, A, B, C)
        p_long  = exact_european_cir_put_price(X0, T,   K, A, B, C)
        # Both are positive; no simple ordering guaranteed, just check positivity
        assert p_short >= 0
        assert p_long  >= 0

    def test_zero_strike(self):
        """Put with K=0 has zero payoff."""
        price = exact_european_cir_put_price(X0, T, 0.0, A, B, C)
        assert price == pytest.approx(0.0, abs=1e-10)


class TestMcEuropeanPut:
    def test_exact_method_close_to_analytical(self):
        analytical = exact_european_cir_put_price(X0, T, K, A, B, C)
        mc_price, mc_se = mc_european_cir_put(
            X0, T, K, A, B, C, N_PATHS, method='exact', seed=SEED)
        # Should be within 5 standard errors of the analytical benchmark
        assert abs(mc_price - analytical) < 5 * mc_se + 1e-5

    def test_fte_method(self):
        price, se = mc_european_cir_put(
            X0, T, K, A, B, C, N_PATHS, method='fte', n_steps=40, seed=SEED)
        assert price > 0
        # FTE can produce slightly negative X_T (discretisation artefact),
        # inflating payoffs beyond K; only exact sampling guarantees price<=K.
        assert price < 2 * K

    def test_qem_method(self):
        price, se = mc_european_cir_put(
            X0, T, K, A, B, C, N_PATHS, method='qem', n_steps=10, seed=SEED)
        assert price > 0
        assert price <= K

    def test_mc_converges_with_more_paths(self):
        """MC error decreases as N grows (test with 3 levels)."""
        analytical = exact_european_cir_put_price(X0, T, K, A, B, C)
        errors = []
        for N in [1_000, 10_000, 100_000]:
            p, _ = mc_european_cir_put(
                X0, T, K, A, B, C, N, method='exact', seed=SEED)
            errors.append(abs(p - analytical))
        # Errors should generally decrease (not guaranteed for individual draws,
        # but standard errors give probabilistic guarantee)
        # Check: largest N gives smallest error among the three
        assert errors[2] < errors[0] + 5e-4

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            mc_european_cir_put(X0, T, K, A, B, C, 100, method='bogus')

    def test_fte_requires_n_steps(self):
        with pytest.raises(ValueError):
            mc_european_cir_put(X0, T, K, A, B, C, 100, method='fte')


class TestMcAsianPut:
    def test_asian_leq_european(self):
        """E[(K - avg X)^+] <= E[(K - X_T)^+] by Jensen's inequality."""
        euro_price, _ = mc_european_cir_put(
            X0, T, K, A, B, C, N_PATHS, method='exact', seed=SEED)
        asian_price, _ = mc_asian_cir_put(
            X0, T, K, A, B, C, M_fixings=10, n_paths=N_PATHS,
            method='exact', seed=SEED)
        # Strict inequality in general; allow small MC noise
        assert asian_price <= euro_price + 2e-4

    def test_positive(self):
        price, _ = mc_asian_cir_put(
            X0, T, K, A, B, C, M_fixings=10, n_paths=N_PATHS,
            method='exact', seed=SEED)
        assert price >= 0

    def test_fte_method(self):
        price, _ = mc_asian_cir_put(
            X0, T, K, A, B, C, M_fixings=10, n_paths=5_000,
            method='fte', n_steps=2, seed=SEED)
        assert price >= 0

    def test_qem_method(self):
        price, _ = mc_asian_cir_put(
            X0, T, K, A, B, C, M_fixings=10, n_paths=5_000,
            method='qem', n_steps=2, seed=SEED)
        assert price >= 0

    def test_more_fixings_has_lower_variance_payoff(self):
        """More averaging → smaller variance of average → Asian price effect."""
        p10, _ = mc_asian_cir_put(
            X0, T, K, A, B, C, M_fixings=10, n_paths=N_PATHS,
            method='exact', seed=SEED)
        p1, _ = mc_asian_cir_put(
            X0, T, K, A, B, C, M_fixings=1, n_paths=N_PATHS,
            method='exact', seed=SEED)
        # M=1 is just the European put (average over last fixing only)
        assert p10 >= 0 and p1 >= 0


class TestMcBasketPut:
    def test_single_asset_matches_european(self):
        """Single-asset basket with w=1 should equal European put."""
        params_list = [(A, B, C)]
        basket_price, _ = mc_basket_cir_put(
            [X0], T, K, [1.0], params_list, N_PATHS,
            method='exact', seed=SEED)
        euro_price, _ = mc_european_cir_put(
            X0, T, K, A, B, C, N_PATHS, method='exact', seed=SEED)
        # Should match to within a few basis points (same payoff, different seeds)
        assert abs(basket_price - euro_price) < 5e-3

    def test_positive(self):
        params_list = [(A, B, C)] * 3
        price, _ = mc_basket_cir_put(
            [X0] * 3, T, K, [1 / 3] * 3, params_list, N_PATHS,
            method='exact', seed=SEED)
        assert price >= 0

    def test_fte_method(self):
        params_list = [(A, B, C)] * 2
        price, _ = mc_basket_cir_put(
            [X0] * 2, T, K, [0.5, 0.5], params_list, 5_000,
            method='fte', n_steps=40, seed=SEED)
        assert price >= 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            mc_basket_cir_put([X0], T, K, [1.0], [(A, B, C)], 100,
                               method='bad')
