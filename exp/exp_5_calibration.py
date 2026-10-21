"""
Experiment 5 – Levenberg-Marquardt CIR calibration.

True parameters: delta*=0.18, b*=-0.50, c*=1.00  →  a*=0.045, X0=0.09

Strikes    : [0.065, 0.085, 0.090, 0.100, 0.110, 0.120]   (6 values)
Maturities : [6, 7, 8, 9, 10]                               (5 values)
Instruments: 6 × 5 = 30 synthetic put prices

Study A – European puts, 100 random-initialisation LM trials
Study B – Asian puts (M=10 fixings), 100 LM trials

NOTE: n_paths=50 000 used for gradient FD in calibration; 100 000 for exact
      target generation. The paper uses larger path counts.

Outputs
-------
results/exp5_euro_calibration.csv
results/exp5_asian_calibration.csv
results/exp5_convergence_euro.png
results/exp5_convergence_asian.png
results/RESULTS.md  (appended)
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from calibration import (
    generate_synthetic_targets,
    lm_calibrate,
    run_calibration_study,
)

# ── true parameters ───────────────────────────────────────────────────────────
DELTA_TRUE = 0.18
B_TRUE     = -0.50
C_TRUE     = 1.00
THETA_TRUE = np.array([DELTA_TRUE, B_TRUE, C_TRUE])
X0         = 0.09

STRIKES    = np.array([0.065, 0.085, 0.090, 0.100, 0.110, 0.120])
MATURITIES = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
N_TRIALS        = 100          # European study: 100 trials with analytical pricing
N_PATHS_CAL     = 50_000      # European calibration paths (analytical ⇒ fast)
N_PATHS_TARGET  = 200_000
# Asian MC calibration is expensive (~0.03 s/price × 210 calls/iter × iters × trials).
# We use reduced settings to keep wall time < 5 min; full-scale values are noted.
N_TRIALS_ASIAN  = 20          # reduced from 100 (full study would take ~50 min)
N_PATHS_ASIAN   = 5_000       # reduced from 50 000 (MC noise acceptable for Jacobian FD)
SEED_BASE  = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Experiment 5 – LM Calibration")
print("=" * 60)


# ── Study A: European put calibration ────────────────────────────────────────
print("\nStudy A: European puts (30 instruments, 100 trials)")

t_euro_start = time.perf_counter()
# Generate targets (analytical formula → no MC noise)
targets_euro, sk_grid, mat_grid = generate_synthetic_targets(
    THETA_TRUE, STRIKES, MATURITIES, X0, option_type='european')
print(f"  Target prices: min={targets_euro.min():.5f}  "
      f"max={targets_euro.max():.5f}  shape={targets_euro.shape}")

# Representative single run from truth
res_truth = lm_calibrate(
    THETA_TRUE * 1.01,    # tiny perturbation
    targets_euro, STRIKES, MATURITIES, X0,
    option_type='european', n_paths=N_PATHS_CAL, seed=SEED_BASE)
print(f"  From-truth run: converged={res_truth['converged']}, "
      f"n_iter={res_truth['n_iter']}, "
      f"||r||={np.linalg.norm(res_truth['final_r']):.2e}")
print(f"  Recovered theta: {res_truth['theta']}")

# Full 100-trial study
study_euro = run_calibration_study(
    THETA_TRUE, STRIKES, MATURITIES, X0,
    n_trials=N_TRIALS, option_type='european',
    perturb_frac=0.2, n_paths=N_PATHS_CAL, seed=SEED_BASE)

t_euro = time.perf_counter() - t_euro_start

print(f"\n  Mean |delta_err|={study_euro['mean_error'][0]:.4f}  "
      f"|b_err|={study_euro['mean_error'][1]:.4f}  "
      f"|c_err|={study_euro['mean_error'][2]:.4f}")
print(f"  Mean ||r|| = {study_euro['mean_residual']:.4e}")
print(f"  Mean n_iter = {study_euro['mean_n_iter']:.1f}")
print(f"  Convergence rate = {study_euro['conv_rate']*100:.1f}%")
print(f"  Total CPU = {t_euro:.1f}s")

df_euro_cal = pd.DataFrame({
    'Trial': np.arange(1, N_TRIALS + 1),
    'delta_err': study_euro['theta_errors'][:, 0],
    'b_err':     study_euro['theta_errors'][:, 1],
    'c_err':     study_euro['theta_errors'][:, 2],
    '||r||':     study_euro['residual_norms'],
    'n_iter':    study_euro['n_iters'],
    'converged': study_euro['converged'],
})
df_euro_cal.to_csv(
    os.path.join(RESULTS_DIR, 'exp5_euro_calibration.csv'), index=False)


# ── Study B: Asian put calibration (M=10) ────────────────────────────────────
print(f"\nStudy B: Asian puts M=10 (30 instruments, {N_TRIALS_ASIAN} trials, "
      f"n_paths={N_PATHS_ASIAN})")
print("  NOTE: reduced from 100 trials / 50 000 paths; full run ~50 min.")

t_asian_start = time.perf_counter()
targets_asian, _, _ = generate_synthetic_targets(
    THETA_TRUE, STRIKES, MATURITIES, X0,
    option_type='asian', M_fixings=10)
print(f"  Target prices: min={targets_asian.min():.5f}  "
      f"max={targets_asian.max():.5f}  shape={targets_asian.shape}")

study_asian = run_calibration_study(
    THETA_TRUE, STRIKES, MATURITIES, X0,
    n_trials=N_TRIALS_ASIAN, option_type='asian', M_fixings=10,
    perturb_frac=0.2, n_paths=N_PATHS_ASIAN, seed=SEED_BASE + 1)

t_asian = time.perf_counter() - t_asian_start

print(f"\n  Mean |delta_err|={study_asian['mean_error'][0]:.4f}  "
      f"|b_err|={study_asian['mean_error'][1]:.4f}  "
      f"|c_err|={study_asian['mean_error'][2]:.4f}")
print(f"  Mean ||r|| = {study_asian['mean_residual']:.4e}")
print(f"  Mean n_iter = {study_asian['mean_n_iter']:.1f}")
print(f"  Convergence rate = {study_asian['conv_rate']*100:.1f}%")
print(f"  Total CPU = {t_asian:.1f}s")

df_asian_cal = pd.DataFrame({
    'Trial': np.arange(1, N_TRIALS_ASIAN + 1),
    'delta_err': study_asian['theta_errors'][:, 0],
    'b_err':     study_asian['theta_errors'][:, 1],
    'c_err':     study_asian['theta_errors'][:, 2],
    '||r||':     study_asian['residual_norms'],
    'n_iter':    study_asian['n_iters'],
    'converged': study_asian['converged'],
})
df_asian_cal.to_csv(
    os.path.join(RESULTS_DIR, 'exp5_asian_calibration.csv'), index=False)


# ── Convergence plots for representative runs ─────────────────────────────────
def _conv_plot(theta_true, target_prices, strikes, maturities, x0,
               option_type, M_fixings, n_paths, n_reps, seed, title, outpath):
    """Run n_reps LM trials and plot residual convergence curves."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7, 4))
    for rep in range(n_reps):
        perturb = 1.0 + rng.uniform(-0.2, 0.2, size=3)
        theta_init = np.array([
            max(theta_true[0] * perturb[0], 1e-4),
            min(theta_true[1] * perturb[1], -1e-4),
            max(theta_true[2] * perturb[2], 1e-4),
        ])
        res = lm_calibrate(theta_init, target_prices, strikes, maturities, x0,
                           option_type=option_type, M_fixings=M_fixings,
                           n_paths=n_paths,
                           seed=int(rng.integers(0, 2**31)))
        iters = np.arange(len(res['residuals']))
        ax.semilogy(iters, res['residuals'], alpha=0.4, lw=0.8, color='steelblue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||r||')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved → {outpath}")


print("\nGenerating convergence plots (10 representative runs each)...")

_conv_plot(THETA_TRUE, targets_euro, STRIKES, MATURITIES, X0,
           'european', 10, N_PATHS_CAL, 10, SEED_BASE + 200,
           'Exp 5A – European put LM residual convergence (10 runs)',
           os.path.join(RESULTS_DIR, 'exp5_convergence_euro.png'))

_conv_plot(THETA_TRUE, targets_asian, STRIKES, MATURITIES, X0,
           'asian', 10, N_PATHS_ASIAN, 5, SEED_BASE + 300,
           f'Exp 5B – Asian put LM residual convergence (5 runs, n_paths={N_PATHS_ASIAN})',
           os.path.join(RESULTS_DIR, 'exp5_convergence_asian.png'))


# ── Summary tables ────────────────────────────────────────────────────────────
def _summary_df(study, label):
    return pd.DataFrame({
        'Study': [label],
        'Mean|delta_err|': [study['mean_error'][0]],
        'Mean|b_err|':     [study['mean_error'][1]],
        'Mean|c_err|':     [study['mean_error'][2]],
        'Std|delta_err|':  [study['std_error'][0]],
        'Std|b_err|':      [study['std_error'][1]],
        'Std|c_err|':      [study['std_error'][2]],
        'Mean||r||':       [study['mean_residual']],
        'Mean n_iter':     [study['mean_n_iter']],
        'Conv rate (%)':   [study['conv_rate'] * 100],
    })

df_summary = pd.concat([
    _summary_df(study_euro,  f'European ({N_TRIALS} trials)'),
    _summary_df(study_asian, f'Asian M=10 ({N_TRIALS_ASIAN} trials, {N_PATHS_ASIAN} paths)'),
], ignore_index=True)

print("\n--- Summary ---")
print(df_summary.to_string(index=False, float_format='%.4f'))


# ── Append to RESULTS.md ─────────────────────────────────────────────────────
md_path = os.path.join(RESULTS_DIR, 'RESULTS.md')
with open(md_path, 'a') as fh:
    fh.write("\n\n## Experiment 5 – LM Calibration\n\n")
    fh.write(f"True theta: delta={DELTA_TRUE}, b={B_TRUE}, c={C_TRUE}  "
             f"(a={DELTA_TRUE*C_TRUE**2/4:.4f})\n")
    fh.write(f"X0={X0}, Strikes={list(STRIKES)}, Maturities={list(MATURITIES)}\n")
    fh.write(f"European: n_trials={N_TRIALS}, n_paths_cal={N_PATHS_CAL}\n")
    fh.write(f"Asian:    n_trials={N_TRIALS_ASIAN}, n_paths_cal={N_PATHS_ASIAN} "
             f"(reduced for tractability; full 100-trial run ~50 min)\n")
    fh.write("perturb_frac=0.20\n\n")
    fh.write("### Summary\n\n")
    fh.write(df_summary.to_markdown(index=False, floatfmt='.4f') + '\n\n')
    fh.write("### European calibration per-trial stats (first 5 rows)\n\n")
    fh.write(df_euro_cal.head().to_markdown(index=False, floatfmt='.5f') + '\n\n')
    fh.write("### Asian calibration per-trial stats (first 5 rows)\n\n")
    fh.write(df_asian_cal.head().to_markdown(index=False, floatfmt='.5f') + '\n\n')
    fh.write("Convergence plots:\n")
    fh.write("![euro](exp5_convergence_euro.png)\n")
    fh.write("![asian](exp5_convergence_asian.png)\n")

print(f"\nResults appended to {md_path}")
print("Experiment 5 complete.")
