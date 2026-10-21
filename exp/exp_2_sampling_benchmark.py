"""
Experiment 2: Accuracy and efficiency of central and non-central chi-square
direct inversion sampling.

Benchmarks:
1. Central chi-square: direct inversion (Chebyshev) vs Ahrens-Dieter vs scipy.
   delta in {0.1, 0.01, 0.001}, n=10^6 samples each (paper uses 10^7; reduced
   to 10^6 for practical run time).
2. Non-central chi-square moment validation: 6 (delta, lambda) pairs,
   5*10^6 samples, moments 1..10 compared to theory.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chebyshev import (
    TRUNCATION_ORDERS,
    build_all_coefficient_blocks,
    get_region_u_bounds,
    load_all_coefficient_blocks,
)
from src.chi_square_sampler import (
    sample_ahrens_dieter,
    sample_central_chi2_direct,
    sample_central_chi2_scipy,
    sample_noncentral_chi2,
    theoretical_noncentral_moments,
    time_sampling,
)

COEFF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "coefficients")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paper benchmark settings (reduced from 10^7 to 10^6 for tractability)
N_CENTRAL = 1_000_000
N_NONCENTRAL = 2_000_000  # paper: 5*10^7; reduced to 2*10^6 for tractability

# delta values for central benchmark
CENTRAL_DELTAS = [0.1, 0.01, 0.001]

# (delta, lambda) pairs for non-central moment test
NONCENTRAL_PAIRS = [
    (0.1,   0.11517),
    (0.1,   15.9501),
    (0.01,  0.15505),
    (0.01,  15.995),
    (0.001, 0.1595),
    (0.001, 15.9995),
]


def _get_blocks_for_delta(delta: float, all_blocks: Dict):
    """Find the coefficient blocks covering the given delta value."""
    for (dmin, dmax) in TRUNCATION_ORDERS.keys():
        if dmin <= delta <= dmax:
            key = f"delta_{dmin}_{dmax}"
            if key in all_blocks:
                return all_blocks[key], get_region_u_bounds(dmin, dmax)
    raise ValueError(f"No coefficient block found for delta={delta}")


# ---------------------------------------------------------------------------
# Part 1: Central chi-square CPU benchmark
# ---------------------------------------------------------------------------

def run_central_benchmark(all_blocks: Dict) -> pd.DataFrame:
    """
    Benchmark direct inversion vs Ahrens-Dieter vs scipy for central chi-square.
    """
    print("\n" + "=" * 60)
    print("PART 1: Central chi-square CPU benchmark")
    print(f"(n={N_CENTRAL:,} samples per method per delta)")
    print("=" * 60)

    rows = []
    for delta in CENTRAL_DELTAS:
        blocks, region_u_bounds = _get_blocks_for_delta(delta, all_blocks)

        for method in ["direct", "ahrens_dieter", "scipy"]:
            samples, elapsed = time_sampling(
                method=method,
                delta=delta,
                n_samples=N_CENTRAL,
                coeff_blocks=blocks,
                region_u_bounds=region_u_bounds,
                seed=42,
            )
            # Basic sanity checks
            mean_s = float(samples.mean())
            var_s = float(samples.var())
            rows.append({
                "delta": delta,
                "method": method,
                "n_samples": N_CENTRAL,
                "time_s": round(elapsed, 4),
                "sample_mean": round(mean_s, 6),
                "theoretical_mean": round(delta, 6),
                "sample_var": round(var_s, 6),
                "theoretical_var": round(2 * delta, 6),
            })
            print(
                f"  delta={delta:.3f}  {method:<16}  "
                f"time={elapsed:.3f}s  mean={mean_s:.4f} (th={delta})  "
                f"var={var_s:.4f} (th={2*delta:.4f})"
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp2_central_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Speed comparison table
    print("\nSpeed comparison (time_ahrens_dieter / time_direct):")
    for delta in CENTRAL_DELTAS:
        sub = df[df["delta"] == delta]
        t_direct = sub[sub["method"] == "direct"]["time_s"].values
        t_ad = sub[sub["method"] == "ahrens_dieter"]["time_s"].values
        if len(t_direct) > 0 and len(t_ad) > 0 and t_direct[0] > 0:
            ratio = t_ad[0] / t_direct[0]
            print(f"  delta={delta}: AD/direct speed ratio = {ratio:.2f}x")

    return df


# ---------------------------------------------------------------------------
# Part 2: Non-central chi-square moment validation
# ---------------------------------------------------------------------------

def run_noncentral_moments(all_blocks: Dict) -> pd.DataFrame:
    """
    Validate non-central chi-square sampler by comparing empirical moments 1..10
    to theoretical values.
    """
    print("\n" + "=" * 60)
    print("PART 2: Non-central chi-square moment validation")
    print(f"(n={N_NONCENTRAL:,} samples per parameter pair)")
    print("=" * 60)

    rows = []
    all_rel_errors = {}

    for delta, lam in NONCENTRAL_PAIRS:
        blocks, region_u_bounds = _get_blocks_for_delta(delta, all_blocks)
        theory_moments = theoretical_noncentral_moments(delta, lam, max_order=10)

        t0 = time.perf_counter()
        samples = sample_noncentral_chi2(
            delta=delta,
            lam=lam,
            n_samples=N_NONCENTRAL,
            coeff_blocks=blocks,
            region_u_bounds=region_u_bounds,
            seed=123,
        )
        elapsed = time.perf_counter() - t0

        # Empirical moments
        emp_moments = np.array([np.mean(samples ** k) for k in range(1, 11)])
        rel_errors = np.abs(emp_moments - theory_moments) / (np.abs(theory_moments) + 1e-300)

        pair_key = f"(delta={delta},lam={lam})"
        all_rel_errors[pair_key] = rel_errors.tolist()
        print(f"  {pair_key}  max_rel_err={rel_errors.max():.2e}  time={elapsed:.2f}s")

        for k, (emp, th, rel) in enumerate(
            zip(emp_moments, theory_moments, rel_errors), start=1
        ):
            rows.append({
                "delta": delta,
                "lam": lam,
                "moment_order": k,
                "empirical_moment": float(emp),
                "theoretical_moment": float(th),
                "relative_error": float(rel),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp2_noncentral_moments.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Produce Figure 1-style plot
    _plot_moment_errors(all_rel_errors)

    return df


def _plot_moment_errors(all_rel_errors: Dict) -> None:
    """Plot relative moment errors vs order for each (delta, lam) pair."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(
        f"Non-central chi-square moment relative errors\n(n={N_NONCENTRAL:,} samples)",
        fontsize=13,
    )

    for ax, (pair_key, rel_errs) in zip(axes, all_rel_errors.items()):
        orders = np.arange(1, len(rel_errs) + 1)
        ax.semilogy(orders, np.array(rel_errs) + 1e-20, "o-", markersize=5)
        ax.axhline(1.0 / N_NONCENTRAL**0.5, color="r", ls="--", label="MC noise floor")
        ax.set_title(pair_key, fontsize=9)
        ax.set_xlabel("Moment order k")
        ax.set_ylabel("Relative error |mu_k - theory| / |theory|")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xticks(orders)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp2_moment_errors.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Experiment 2: chi-square Sampler Accuracy & Efficiency")
    print("=" * 60)

    # Load or build coefficient blocks
    try:
        all_blocks = load_all_coefficient_blocks(COEFF_DIR)
        print("Loaded precomputed coefficient blocks from", COEFF_DIR)
    except Exception as e:
        print(f"Could not load blocks ({e}). Building now...")
        all_blocks = build_all_coefficient_blocks(COEFF_DIR)

    df_central = run_central_benchmark(all_blocks)
    df_noncentral = run_noncentral_moments(all_blocks)

    _append_to_results_md(df_central, df_noncentral)
    print("\nExperiment 2 complete.")


def _append_to_results_md(df_central: pd.DataFrame, df_noncentral: pd.DataFrame):
    results_md = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(results_md, "a") as f:
        f.write("\n\n## Experiment 2: chi-square Sampler Accuracy & Efficiency\n\n")
        f.write(f"### Central chi-square CPU Benchmark (n={N_CENTRAL:,})\n\n")
        f.write("| delta | Method | Time (s) | Sample Mean | Theoretical Mean |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df_central.iterrows():
            f.write(
                f"| {row['delta']} | {row['method']} | "
                f"{row['time_s']:.4f} | {row['sample_mean']:.6f} | "
                f"{row['theoretical_mean']:.6f} |\n"
            )

        f.write(f"\n### Non-central Moment Validation (n={N_NONCENTRAL:,})\n\n")
        # Report max relative error per pair
        f.write("| delta | lambda | Max Rel Err (k=1..10) |\n")
        f.write("|---|---|---|\n")
        for (delta, lam), sub in df_noncentral.groupby(["delta", "lam"]):
            max_err = sub["relative_error"].max()
            f.write(f"| {delta} | {lam} | {max_err:.2e} |\n")


if __name__ == "__main__":
    main()
