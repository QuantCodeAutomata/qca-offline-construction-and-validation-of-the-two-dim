"""
Unit tests for src/chi_square_sampler.py.

Tests verify:
- Sample positivity for all samplers
- Sample mean and variance close to theoretical values
- Non-central chi-square moment accuracy
- Ahrens-Dieter algorithm produces valid gamma variates
- Theoretical moment computation correctness
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy import special, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chi_square_sampler import (
    _ahrens_dieter_gamma,
    sample_ahrens_dieter,
    sample_central_chi2_scipy,
    sample_noncentral_chi2,
    theoretical_noncentral_moments,
)

# We use scipy-based sampling for most tests (no coefficient files needed)
N_SMALL = 50_000   # small sample for quick tests
N_LARGE = 200_000  # larger sample for moment accuracy tests
RTOL_MEAN = 0.05   # 5% relative tolerance for mean
RTOL_VAR = 0.10    # 10% relative tolerance for variance


# ---------------------------------------------------------------------------
# Ahrens-Dieter gamma sampler tests
# ---------------------------------------------------------------------------


def test_ahrens_dieter_gamma_positive():
    """Ahrens-Dieter gamma samples must all be positive."""
    rng = np.random.default_rng(0)
    for alpha in [0.05, 0.1, 0.5, 0.9]:
        samples = _ahrens_dieter_gamma(alpha, 1000, rng)
        assert np.all(samples > 0), f"Non-positive sample for alpha={alpha}"


def test_ahrens_dieter_gamma_mean():
    """E[Gamma(alpha,1)] = alpha."""
    rng = np.random.default_rng(1)
    for alpha in [0.05, 0.1, 0.5, 0.9]:
        samples = _ahrens_dieter_gamma(alpha, N_LARGE, rng)
        mean = samples.mean()
        assert abs(mean - alpha) / alpha < RTOL_MEAN, (
            f"alpha={alpha}: mean={mean:.5f}, expected={alpha:.5f}"
        )


def test_ahrens_dieter_gamma_variance():
    """Var[Gamma(alpha,1)] = alpha."""
    rng = np.random.default_rng(2)
    for alpha in [0.1, 0.5, 0.9]:
        samples = _ahrens_dieter_gamma(alpha, N_LARGE, rng)
        var = samples.var()
        assert abs(var - alpha) / alpha < RTOL_VAR, (
            f"alpha={alpha}: var={var:.5f}, expected={alpha:.5f}"
        )


def test_ahrens_dieter_invalid_alpha():
    """Ahrens-Dieter raises ValueError for alpha outside (0, 1)."""
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        _ahrens_dieter_gamma(0.0, 100, rng)
    with pytest.raises(ValueError):
        _ahrens_dieter_gamma(1.0, 100, rng)
    with pytest.raises(ValueError):
        _ahrens_dieter_gamma(-0.5, 100, rng)


# ---------------------------------------------------------------------------
# Central chi-square Ahrens-Dieter tests
# ---------------------------------------------------------------------------


def test_central_chi2_ad_positive():
    """Ahrens-Dieter central chi-square samples are non-negative.

    For very small delta (e.g. 0.001) the Gamma shape alpha=delta/2 is tiny and
    many samples underflow to exactly 0.0 in double precision — that is a valid
    representation of an extremely small positive chi-squared draw, not a bug.
    The support of chi2(delta) is [0, ∞), so we assert >= 0, not strictly > 0.
    """
    for delta in [0.1, 0.01, 0.001]:
        samples = sample_ahrens_dieter(delta, 1000, seed=4)
        assert np.all(samples >= 0)


def test_central_chi2_ad_mean():
    """E[chi2(delta)] = delta."""
    for delta in [0.1, 0.5, 1.0, 2.0]:
        samples = sample_ahrens_dieter(delta, N_LARGE, seed=5)
        mean = samples.mean()
        assert abs(mean - delta) / delta < RTOL_MEAN, (
            f"delta={delta}: mean={mean:.5f}"
        )


def test_central_chi2_ad_mean_very_small_delta():
    """Regression test: AD sampler must not over-estimate the mean for tiny delta.

    Before the fix, the vectorised acceptance guard '& (x > 0)' incorrectly
    rejected underflowed-to-zero small-branch samples for delta=0.001, causing
    the empirical mean to be ~3× the true value (observed 0.0032 vs 0.001).
    """
    delta = 0.001
    n = 500_000
    samples = sample_ahrens_dieter(delta, n, seed=99)
    mean = samples.mean()
    # Allow ±25 % relative error; the old (buggy) code gave ~220 % error
    assert abs(mean - delta) / delta < 0.25, (
        f"delta={delta}: mean={mean:.6f}, expected≈{delta}. "
        "Possible re-introduction of the small-branch rejection bug."
    )


def test_central_chi2_ad_variance():
    """Var[chi2(delta)] = 2*delta."""
    for delta in [0.1, 0.5, 1.5]:
        samples = sample_ahrens_dieter(delta, N_LARGE, seed=6)
        var = samples.var()
        assert abs(var - 2.0 * delta) / (2.0 * delta) < RTOL_VAR


def test_central_chi2_ad_large_delta():
    """For delta >= 2, Ahrens-Dieter falls back to numpy gamma."""
    samples = sample_ahrens_dieter(4.0, N_SMALL, seed=7)
    assert np.all(samples > 0)
    mean = samples.mean()
    assert abs(mean - 4.0) / 4.0 < RTOL_MEAN


# ---------------------------------------------------------------------------
# Scipy reference sampler tests
# ---------------------------------------------------------------------------


def test_scipy_sampler_positive():
    """scipy chi-square samples are positive."""
    samples = sample_central_chi2_scipy(0.1, 1000, seed=8)
    assert np.all(samples > 0)


def test_scipy_sampler_mean():
    """scipy chi-square mean is close to delta."""
    for delta in [0.1, 1.0, 5.0]:
        samples = sample_central_chi2_scipy(delta, N_LARGE, seed=9)
        assert abs(samples.mean() - delta) / delta < RTOL_MEAN


# ---------------------------------------------------------------------------
# Non-central chi-square sampler tests
# ---------------------------------------------------------------------------


def test_noncentral_positive():
    """Non-central chi-square samples are non-negative."""
    for delta, lam in [(0.1, 0.5), (0.1, 15.0), (0.01, 5.0)]:
        samples = sample_noncentral_chi2(delta, lam, N_SMALL, seed=10)
        assert np.all(samples >= 0.0)


def test_noncentral_mean():
    """E[ncx2(delta, lam)] = delta + lam."""
    for delta, lam in [(0.5, 2.0), (1.0, 5.0), (0.1, 0.5)]:
        samples = sample_noncentral_chi2(delta, lam, N_LARGE, seed=11)
        expected_mean = delta + lam
        assert abs(samples.mean() - expected_mean) / expected_mean < RTOL_MEAN, (
            f"delta={delta}, lam={lam}: mean={samples.mean():.4f}, expected={expected_mean:.4f}"
        )


def test_noncentral_variance():
    """Var[ncx2(delta, lam)] = 2*(delta + 2*lam)."""
    for delta, lam in [(0.5, 2.0), (1.0, 5.0)]:
        samples = sample_noncentral_chi2(delta, lam, N_LARGE, seed=12)
        expected_var = 2.0 * (delta + 2.0 * lam)
        assert abs(samples.var() - expected_var) / expected_var < RTOL_VAR, (
            f"delta={delta}, lam={lam}: var={samples.var():.4f}, expected={expected_var:.4f}"
        )


def test_noncentral_lambda_zero():
    """ncx2(delta, 0) should match central chi2(delta) in distribution."""
    delta = 1.0
    samples = sample_noncentral_chi2(delta, 0.0, N_LARGE, seed=13)
    assert abs(samples.mean() - delta) / delta < RTOL_MEAN
    assert abs(samples.var() - 2.0 * delta) / (2.0 * delta) < RTOL_VAR


def test_noncentral_high_lambda():
    """High non-centrality (lam=20) recursive decomposition produces valid samples."""
    delta, lam = 0.5, 20.0
    samples = sample_noncentral_chi2(delta, lam, N_LARGE, seed=14)
    expected_mean = delta + lam
    assert np.all(samples >= 0)
    assert abs(samples.mean() - expected_mean) / expected_mean < RTOL_MEAN


def test_noncentral_negative_lambda_raises():
    """Negative non-centrality raises ValueError."""
    with pytest.raises(ValueError):
        sample_noncentral_chi2(1.0, -1.0, 100, seed=15)


# ---------------------------------------------------------------------------
# Theoretical moment tests
# ---------------------------------------------------------------------------


def test_theoretical_moments_mean():
    """First theoretical moment = delta + lam."""
    for delta, lam in [(0.5, 2.0), (1.0, 0.0), (0.1, 5.0)]:
        moments = theoretical_noncentral_moments(delta, lam, max_order=1)
        expected = delta + lam
        assert abs(moments[0] - expected) / max(expected, 1e-10) < 1e-8, (
            f"delta={delta}, lam={lam}: moment_1={moments[0]:.6f}, expected={expected:.6f}"
        )


def test_theoretical_moments_variance():
    """Second cumulant (Var) = 2*(delta + 2*lam) from first two raw moments."""
    for delta, lam in [(0.5, 2.0), (1.0, 0.0), (0.1, 5.0)]:
        mu = theoretical_noncentral_moments(delta, lam, max_order=2)
        # Var = E[X^2] - E[X]^2
        var = mu[1] - mu[0] ** 2
        expected_var = 2.0 * (delta + 2.0 * lam)
        assert abs(var - expected_var) / max(expected_var, 1e-10) < 1e-6, (
            f"delta={delta}, lam={lam}: var={var:.6f}, expected={expected_var:.6f}"
        )


def test_theoretical_moments_count():
    """theoretical_noncentral_moments returns exactly max_order values."""
    for max_order in [1, 5, 10]:
        moments = theoretical_noncentral_moments(1.0, 2.0, max_order=max_order)
        assert len(moments) == max_order


def test_theoretical_moments_positive_moments():
    """All theoretical moments of a positive RV should be positive."""
    moments = theoretical_noncentral_moments(0.5, 1.5, max_order=10)
    assert np.all(moments > 0), "All moments of ncx2 should be positive"


def test_theoretical_moments_monotone():
    """For a positive RV with mean > 1, raw moments should be increasing."""
    # ncx2 with mean = delta + lam > 1
    delta, lam = 0.5, 2.0  # mean = 2.5
    moments = theoretical_noncentral_moments(delta, lam, max_order=5)
    assert np.all(np.diff(moments) > 0), "Raw moments should increase"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_scipy_single_sample():
    """Single-sample scipy sampling works."""
    samples = sample_central_chi2_scipy(0.5, 1, seed=20)
    assert len(samples) == 1
    assert samples[0] > 0


def test_ahrens_dieter_single_sample():
    """Single-sample Ahrens-Dieter sampling works."""
    samples = sample_ahrens_dieter(0.1, 1, seed=21)
    assert len(samples) == 1
    assert samples[0] > 0


def test_noncentral_single_sample():
    """Single-sample non-central sampling works."""
    samples = sample_noncentral_chi2(0.5, 1.0, 1, seed=22)
    assert len(samples) == 1
    assert samples[0] >= 0


def test_noncentral_reproducibility():
    """Same seed produces same samples."""
    s1 = sample_noncentral_chi2(0.5, 2.0, 100, seed=99)
    s2 = sample_noncentral_chi2(0.5, 2.0, 100, seed=99)
    np.testing.assert_array_equal(s1, s2)
