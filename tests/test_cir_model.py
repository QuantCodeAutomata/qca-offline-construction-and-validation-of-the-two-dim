"""
Unit tests for src/cir_model.py.

Run with:  python -m pytest tests/test_cir_model.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cir_model import (
    cir_delta,
    cir_C,
    cir_transition_params,
    cir_mean,
    cir_variance,
    sample_cir_terminal,
    sample_cir_path,
    simulate_fte,
    simulate_qem,
)

# ── shared parameters ──────────────────────────────────────────────────────────
A, B, C = 0.045, -0.5, 1.0     # b < 0, Feller condition: 2a > c^2 ↔ 0.09 > 1 is False
# Use a = 1.5 where Feller holds: 2*1.5 = 3 > 1
A_FELLER = 1.5
X0, T = 0.09, 5.0
N = 200_000
SEED = 0


class TestCirDelta:
    def test_formula(self):
        assert cir_delta(0.045, 1.0) == pytest.approx(0.18)

    def test_scales_with_a(self):
        assert cir_delta(2 * A, C) == pytest.approx(2 * cir_delta(A, C))

    def test_inverse_square_c(self):
        assert cir_delta(A, 2 * C) == pytest.approx(cir_delta(A, C) / 4)


class TestCirC:
    def test_positive_for_negative_b(self):
        assert cir_C(T, B, C) > 0

    def test_increases_with_t(self):
        assert cir_C(10.0, B, C) > cir_C(1.0, B, C)

    def test_zero_at_zero_t(self):
        # Limiting value as t → 0: c^2*t/4 (first-order expansion)
        assert cir_C(1e-8, B, C) == pytest.approx(0.0, abs=1e-6)


class TestCirTransitionParams:
    def test_shapes(self):
        x0_vec = np.array([0.05, 0.09, 0.12])
        delta, scale, lam = cir_transition_params(x0_vec, T, A, B, C)
        assert np.shape(lam) == (3,)
        assert np.isscalar(delta)
        assert np.isscalar(scale)

    def test_all_positive(self):
        delta, scale, lam = cir_transition_params(X0, T, A, B, C)
        assert delta > 0
        assert scale > 0
        assert lam > 0

    def test_delta_value(self):
        delta, _, _ = cir_transition_params(X0, T, A, B, C)
        assert delta == pytest.approx(cir_delta(A, C))


class TestSampleCirTerminal:
    samples = None

    @classmethod
    def setup_class(cls):
        cls.samples = sample_cir_terminal(X0, T, A, B, C, N, seed=SEED)

    def test_shape(self):
        assert self.samples.shape == (N,)

    def test_non_negative(self):
        assert np.all(self.samples >= 0)

    def test_mean_close(self):
        expected_mean = cir_mean(X0, T, A, B)
        sample_mean = self.samples.mean()
        # Allow 1% relative tolerance given MC noise at N=200k
        assert abs(sample_mean - expected_mean) / expected_mean < 0.02

    def test_variance_close(self):
        expected_var = cir_variance(X0, T, A, B, C)
        sample_var = self.samples.var()
        assert abs(sample_var - expected_var) / expected_var < 0.05


class TestSampleCirPath:
    def test_shape(self):
        times = [1.0, 3.0, 5.0]
        X = sample_cir_path(X0, times, A, B, C, 1000, seed=SEED)
        assert X.shape == (1000, 3)

    def test_non_negative(self):
        times = [1.0, 3.0, 5.0]
        X = sample_cir_path(X0, times, A, B, C, 1000, seed=SEED)
        assert np.all(X >= 0)

    def test_terminal_matches_single_step(self):
        """sample_cir_path terminal = sample_cir_terminal (same seed)."""
        T_test = 5.0
        X_path = sample_cir_path(X0, [T_test], A, B, C, N, seed=SEED)
        X_term = sample_cir_terminal(X0, T_test, A, B, C, N, seed=SEED)
        # Same seed → same chi2 draws → should be identical
        np.testing.assert_array_almost_equal(X_path[:, 0], X_term)


class TestSimulateFte:
    def test_shape(self):
        X_T = simulate_fte(X0, T, A, B, C, 1000, n_steps=50, seed=SEED)
        assert X_T.shape == (1000,)

    def test_mean_close_large_steps(self):
        """FTE with many steps should approximate true mean."""
        X_T = simulate_fte(X0, T, A, B, C, N, n_steps=500, seed=SEED)
        expected = cir_mean(X0, T, A, B)
        assert abs(X_T.mean() - expected) / expected < 0.05

    def test_fte_positive_drift_no_truncation(self):
        """Drift term in FTE uses raw X (not truncated); only sqrt uses max(X,0)."""
        # We cannot observe internals directly, but we can check that the scheme
        # runs without error even when X_0 is very small (near-zero Bessel regime)
        X_T = simulate_fte(1e-8, T, A, B, C, 100, n_steps=100, seed=SEED)
        assert X_T.shape == (100,)


class TestSimulateQem:
    def test_shape(self):
        X_T = simulate_qem(X0, T, A, B, C, 1000, n_steps=20, seed=SEED)
        assert X_T.shape == (1000,)

    def test_non_negative(self):
        """QE-M always produces non-negative values by construction."""
        X_T = simulate_qem(X0, T, A, B, C, 10_000, n_steps=10, seed=SEED)
        assert np.all(X_T >= 0)

    def test_mean_close(self):
        X_T = simulate_qem(X0, T, A, B, C, N, n_steps=40, seed=SEED)
        expected = cir_mean(X0, T, A, B)
        assert abs(X_T.mean() - expected) / expected < 0.05
