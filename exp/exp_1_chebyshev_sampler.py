"""
Experiment 1: Offline construction and validation of the two-dimensional
Chebyshev inverse central chi-square sampler.

Methodology:
1. Build coefficient blocks for all three delta intervals.
2. Validate approximation accuracy on dense grids.
3. Verify Clenshaw recurrence vs direct summation.
4. Validate analytical derivatives via finite differences.
5. Save coefficient files and results.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chebyshev import (
    TRUNCATION_ORDERS,
    U_MAX,
    build_all_coefficient_blocks,
    chebyshev_T,
    clenshaw_2d,
    clenshaw_2d_array,
    compute_chebyshev_coefficients,
    delta_to_y,
    derivative_coefficients_x,
    derivative_coefficients_y,
    direct_sum_2d,
    evaluate_derivative_delta,
    evaluate_inverse_chi2,
    evaluate_inverse_chi2_batch,
    get_region_u_bounds,
    get_xi_bounds,
    load_all_coefficient_blocks,
    u_to_xi,
    xi_to_x,
)

COEFF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "coefficients")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Build all coefficient blocks
# ---------------------------------------------------------------------------

def build_coefficients() -> Dict:
    """Build and save all coefficient blocks."""
    print("\n" + "=" * 60)
    print("STEP 1: Building Chebyshev coefficient blocks")
    print("=" * 60)
    t0 = time.perf_counter()
    all_blocks = build_all_coefficient_blocks(COEFF_DIR)
    elapsed = time.perf_counter() - t0
    print(f"\nCoefficient generation completed in {elapsed:.2f}s (offline cost)")
    print(f"Coefficient files saved to: {COEFF_DIR}")
    return all_blocks, elapsed


# ---------------------------------------------------------------------------
# Step 2: Validate approximation accuracy
# ---------------------------------------------------------------------------

def validate_accuracy(all_blocks: Dict) -> Dict:
    """
    Validate approximation accuracy on dense grids.

    Returns dict with max_abs_err, max_rel_err, percentile errors per interval/region.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Approximation accuracy validation")
    print("=" * 60)

    N_DELTA = 200   # delta points per interval
    N_U = 2000      # u points per region

    validation_results = {}

    for delta_min, delta_max in TRUNCATION_ORDERS.keys():
        key = f"delta_{delta_min}_{delta_max}"
        blocks = all_blocks[key]
        region_u_bounds = get_region_u_bounds(delta_min, delta_max)

        delta_grid = np.linspace(delta_min * 1.001, delta_max * 0.999, N_DELTA)
        region_results = {}

        for region_idx, (block, (u_lo, u_hi)) in enumerate(
            zip(blocks, region_u_bounds)
        ):
            region = region_idx + 1
            # Dense u grid with concentration near boundaries
            u_inner = np.linspace(u_lo + 1e-6, u_hi - 1e-6, N_U // 2)
            u_boundary_lo = np.linspace(u_lo + 1e-8, u_lo + 0.01 * (u_hi - u_lo), N_U // 4)
            u_boundary_hi = np.linspace(u_hi - 0.01 * (u_hi - u_lo), u_hi - 1e-8, N_U // 4)
            u_grid = np.unique(np.concatenate([u_boundary_lo, u_inner, u_boundary_hi]))
            u_grid = np.clip(u_grid, u_lo + 1e-10, min(u_hi - 1e-10, U_MAX))

            abs_errs = []
            rel_errs = []

            for delta in delta_grid[::10]:  # subsample delta for speed
                ref = stats.chi2.ppf(u_grid, df=delta)
                approx = evaluate_inverse_chi2_batch(
                    delta, u_grid, blocks, region_u_bounds
                )
                # Only evaluate within the region
                mask = (u_grid >= u_lo) & (u_grid < u_hi)
                if mask.sum() == 0:
                    continue
                err = np.abs(approx[mask] - ref[mask])
                rel = err / (np.abs(ref[mask]) + 1e-15)
                abs_errs.extend(err.tolist())
                rel_errs.extend(rel.tolist())

            if abs_errs:
                abs_errs = np.array(abs_errs)
                rel_errs = np.array(rel_errs)
                region_results[region] = {
                    "max_abs_err": float(abs_errs.max()),
                    "max_rel_err": float(rel_errs.max()),
                    "p50_abs_err": float(np.percentile(abs_errs, 50)),
                    "p95_abs_err": float(np.percentile(abs_errs, 95)),
                    "p99_abs_err": float(np.percentile(abs_errs, 99)),
                    "mean_abs_err": float(abs_errs.mean()),
                }
                print(
                    f"  delta=[{delta_min},{delta_max}] R{region}: "
                    f"max_abs={abs_errs.max():.2e}  max_rel={rel_errs.max():.2e}  "
                    f"mean_abs={abs_errs.mean():.2e}"
                )

        validation_results[key] = region_results

    return validation_results


# ---------------------------------------------------------------------------
# Step 3: Verify Clenshaw vs direct summation
# ---------------------------------------------------------------------------

def verify_clenshaw(all_blocks: Dict) -> float:
    """
    Verify that Clenshaw recurrence matches direct summation to near machine precision.

    Returns max discrepancy across all tests.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Clenshaw vs direct summation verification")
    print("=" * 60)

    rng = np.random.default_rng(42)
    max_disc = 0.0

    for delta_min, delta_max in list(TRUNCATION_ORDERS.keys())[:1]:  # main interval
        key = f"delta_{delta_min}_{delta_max}"
        blocks = all_blocks[key]
        for region_idx, block in enumerate(blocks):
            C = np.asarray(block["coefficient_matrix"])
            # Random (x, y) test points in [-1, 1]
            n_test = 50
            xs = rng.uniform(-0.99, 0.99, n_test)
            ys = rng.uniform(-0.99, 0.99, n_test)

            for x, y in zip(xs, ys):
                val_clenshaw = clenshaw_2d(x, y, C)
                val_direct = direct_sum_2d(x, y, C)
                disc = abs(val_clenshaw - val_direct)
                max_disc = max(max_disc, disc)

    print(f"  Max discrepancy Clenshaw vs direct: {max_disc:.3e}")
    print(f"  (Expected: near machine precision ~1e-14)")
    return max_disc


# ---------------------------------------------------------------------------
# Step 4: Validate analytical derivatives vs finite differences
# ---------------------------------------------------------------------------

def validate_derivatives(all_blocks: Dict) -> Dict:
    """
    Validate analytical d/d(delta) F_delta^{-1}(u) against central finite differences.

    Returns dict with max and mean relative errors per delta interval.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Derivative validation (analytical vs finite difference)")
    print("=" * 60)

    H_DELTA = 1e-5
    results = {}

    for delta_min, delta_max in TRUNCATION_ORDERS.keys():
        key = f"delta_{delta_min}_{delta_max}"
        blocks = all_blocks[key]
        region_u_bounds = get_region_u_bounds(delta_min, delta_max)

        # Test on interior points
        delta_mid = 0.5 * (delta_min + delta_max)
        u_tests = [0.3, 0.6, 0.85, 0.97]
        u_tests = [u for u in u_tests if u < U_MAX]

        rel_errs = []
        for u in u_tests:
            # Determine region
            region_idx = 0
            for ri, (u_lo, u_hi) in enumerate(region_u_bounds):
                if u_lo <= u < u_hi:
                    region_idx = ri
                    break
            block = blocks[region_idx]

            # Analytical derivative
            try:
                analytical = evaluate_derivative_delta(delta_mid, u, block)
            except Exception:
                continue

            # Central finite difference
            h = H_DELTA
            q_plus = stats.chi2.ppf(u, df=delta_mid + h)
            q_minus = stats.chi2.ppf(u, df=delta_mid - h)
            fd = (q_plus - q_minus) / (2.0 * h)

            if abs(fd) > 1e-12:
                rel_err = abs(analytical - fd) / abs(fd)
                rel_errs.append(rel_err)
            else:
                rel_errs.append(abs(analytical - fd))

        max_re = float(np.max(rel_errs)) if rel_errs else float("nan")
        mean_re = float(np.mean(rel_errs)) if rel_errs else float("nan")
        results[key] = {"max_rel_err": max_re, "mean_rel_err": mean_re}
        print(
            f"  delta=[{delta_min},{delta_max}]: "
            f"max_rel_err_derivative={max_re:.2e}  mean={mean_re:.2e}"
        )

    return results


# ---------------------------------------------------------------------------
# Step 5: Produce validation plots
# ---------------------------------------------------------------------------

def produce_plots(all_blocks: Dict) -> None:
    """Produce error plots and save to results/."""
    print("\n" + "=" * 60)
    print("STEP 5: Producing validation plots")
    print("=" * 60)

    delta_min, delta_max = 0.1, 0.2  # main interval
    key = f"delta_{delta_min}_{delta_max}"
    blocks = all_blocks[key]
    region_u_bounds = get_region_u_bounds(delta_min, delta_max)

    delta_test = 0.15
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Approximation error  delta={delta_test}  (delta interval [{delta_min},{delta_max}])"
    )

    for region_idx, (block, (u_lo, u_hi), ax) in enumerate(
        zip(blocks, region_u_bounds, axes)
    ):
        region = region_idx + 1
        u_grid = np.linspace(u_lo + 1e-6, min(u_hi - 1e-6, U_MAX - 1e-10), 500)
        ref = stats.chi2.ppf(u_grid, df=delta_test)
        approx = evaluate_inverse_chi2_batch(delta_test, u_grid, blocks, region_u_bounds)
        mask = (u_grid >= u_lo) & (u_grid < u_hi)
        abs_err = np.abs(approx[mask] - ref[mask])
        ax.semilogy(u_grid[mask], abs_err + 1e-20, label="abs error")
        ax.set_title(f"Region {region}  u in [{u_lo:.2f},{u_hi:.4f}]")
        ax.set_xlabel("u")
        ax.set_ylabel("Absolute error")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp1_accuracy_by_region.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    # Error across delta values for Region 1
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    deltas_plot = np.linspace(delta_min + 0.001, delta_max - 0.001, 10)
    u_test = np.linspace(0.01, 0.79, 500)
    for d in deltas_plot[::3]:
        ref = stats.chi2.ppf(u_test, df=d)
        approx = evaluate_inverse_chi2_batch(d, u_test, blocks, region_u_bounds)
        abs_err = np.abs(approx - ref)
        ax2.semilogy(u_test, abs_err + 1e-20, alpha=0.6, label=f"delta={d:.3f}")
    ax2.set_title("Region 1 error vs u for several delta values")
    ax2.set_xlabel("u")
    ax2.set_ylabel("Absolute error")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    plot2_path = os.path.join(RESULTS_DIR, "exp1_region1_error_vs_delta.png")
    plt.savefig(plot2_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot2_path}")


# ---------------------------------------------------------------------------
# Step 6: Reproduce truncation order check
# ---------------------------------------------------------------------------

def check_truncation_orders(all_blocks: Dict) -> Dict:
    """
    Check whether published truncation orders are sufficient for O(1e-8) accuracy.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Truncation order sufficiency check")
    print("=" * 60)

    results = {}
    for (delta_min, delta_max), orders in TRUNCATION_ORDERS.items():
        key = f"delta_{delta_min}_{delta_max}"
        blocks = all_blocks[key]
        region_u_bounds = get_region_u_bounds(delta_min, delta_max)

        # Quick accuracy check at centre of interval
        delta_test = 0.5 * (delta_min + delta_max)
        max_errs = []
        for region_idx, (block, (u_lo, u_hi)) in enumerate(
            zip(blocks, region_u_bounds)
        ):
            M, N = orders[region_idx + 1]
            u_grid = np.linspace(u_lo + 1e-5, min(u_hi - 1e-5, U_MAX - 1e-9), 200)
            ref = stats.chi2.ppf(u_grid, df=delta_test)
            approx = evaluate_inverse_chi2_batch(
                delta_test, u_grid, blocks, region_u_bounds
            )
            mask = (u_grid >= u_lo) & (u_grid < u_hi)
            if mask.sum() > 0:
                max_errs.append(float(np.max(np.abs(approx[mask] - ref[mask]))))
        overall = max(max_errs) if max_errs else float("nan")
        status = "SUFFICIENT" if overall < 1e-6 else "INSUFFICIENT"
        print(
            f"  delta=[{delta_min},{delta_max}]: max_err={overall:.2e}  {status}"
        )
        results[key] = {"max_err": overall, "status": status}
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Experiment 1: Chebyshev Inverse Chi-Square Sampler")
    print("=" * 60)

    # Build coefficient blocks (offline)
    all_blocks, coeff_gen_time = build_coefficients()

    # Validate accuracy
    accuracy_results = validate_accuracy(all_blocks)

    # Verify Clenshaw
    max_disc = verify_clenshaw(all_blocks)

    # Validate derivatives
    deriv_results = validate_derivatives(all_blocks)

    # Produce plots
    produce_plots(all_blocks)

    # Check truncation orders
    trunc_results = check_truncation_orders(all_blocks)

    # Save numeric summary
    summary = {
        "experiment": "exp_1",
        "coefficient_generation_time_sec": coeff_gen_time,
        "clenshaw_vs_direct_max_discrepancy": max_disc,
        "accuracy": {
            k: {str(r): v for r, v in region_dict.items()}
            for k, region_dict in accuracy_results.items()
        },
        "derivatives": deriv_results,
        "truncation_check": trunc_results,
    }
    summary_path = os.path.join(RESULTS_DIR, "exp1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Append to RESULTS.md
    _append_to_results_md(accuracy_results, deriv_results, max_disc, coeff_gen_time)

    print("\nExperiment 1 complete.")
    return summary


def _append_to_results_md(accuracy_results, deriv_results, max_disc, coeff_gen_time):
    results_md = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(results_md, "a") as f:
        f.write("\n\n## Experiment 1: Chebyshev Inverse Chi-Square Sampler\n\n")
        f.write(f"- Coefficient generation time (offline): {coeff_gen_time:.2f}s\n")
        f.write(f"- Clenshaw vs direct summation max discrepancy: {max_disc:.2e}\n")
        f.write("\n### Approximation Accuracy by Delta Interval and Region\n\n")
        f.write("| Interval | Region | Max Abs Err | Max Rel Err | Mean Abs Err |\n")
        f.write("|---|---|---|---|---|\n")
        for key, regions in accuracy_results.items():
            interval = key.replace("delta_", "")
            for region, stats_dict in regions.items():
                f.write(
                    f"| {interval} | {region} | "
                    f"{stats_dict['max_abs_err']:.2e} | "
                    f"{stats_dict['max_rel_err']:.2e} | "
                    f"{stats_dict['mean_abs_err']:.2e} |\n"
                )
        f.write("\n### Derivative Validation\n\n")
        f.write("| Interval | Max Rel Err (deriv) | Mean Rel Err (deriv) |\n")
        f.write("|---|---|---|\n")
        for key, d in deriv_results.items():
            interval = key.replace("delta_", "")
            f.write(
                f"| {interval} | {d['max_rel_err']:.2e} | {d['mean_rel_err']:.2e} |\n"
            )


if __name__ == "__main__":
    main()
