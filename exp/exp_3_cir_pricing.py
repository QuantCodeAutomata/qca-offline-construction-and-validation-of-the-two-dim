"""
Experiment 3 – CIR European and Asian put pricing.

Reproduces the convergence tables and error plots comparing:
  • Exact:  exact noncentral chi-square sampling (direct inversion baseline)
  • FTE:    Full Truncation Euler with step sizes h = 1, 1/2, 1/4
  • QE-M:   Quadratic-Exponential scheme with step sizes h = 1, 1/2, 1/4

CIR parameters (matching paper Table 3 / Figure 3 settings)
------------------------------------------------------------
X0 = 0.09,  a = 0.045,  b = -0.5,  c = 1.0
  → delta = 4*a/c^2 = 0.18,  T = 10,  K = 0.09

NOTE: The paper uses N = 10^6 paths; we use N = 100 000 for tractability.
Results will have larger MC noise but the same qualitative behaviour.

Outputs
-------
results/exp3_european_table.csv
results/exp3_asian_M10_table.csv
results/exp3_asian_M40_table.csv
results/exp3_convergence.png
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

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pricing import (
    exact_european_cir_put_price,
    mc_european_cir_put,
    mc_asian_cir_put,
)

# ── constants ────────────────────────────────────────────────────────────────
X0, A, B, C = 0.09, 0.045, -0.5, 1.0
T, K = 10.0, 0.09
N_PATHS = 100_000          # paper uses 1 000 000
SEED_BASE = 2024
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def timed(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ── European pricing table ────────────────────────────────────────────────────
print("=" * 60)
print("Experiment 3 – CIR Pricing")
print("=" * 60)

exact_price = exact_european_cir_put_price(X0, T, K, A, B, C)
print(f"\nExact European put price = {exact_price:.6f}")

euro_rows = []

# Exact MC
(price_e, std_e), t_e = timed(mc_european_cir_put, X0, T, K, A, B, C,
                               N_PATHS, method='exact', seed=SEED_BASE)
euro_rows.append({
    'Method': 'Exact (ncx2)',
    'h': '—',
    'n_steps': '—',
    'Price': price_e,
    'StdErr': std_e,
    'RelErr': abs(price_e - exact_price) / exact_price,
    'CPU (s)': t_e,
})

# FTE with h = 1, 1/2, 1/4
for h_inv in [1, 2, 4]:
    h = 1.0 / h_inv if h_inv > 1 else 1.0
    n_steps = int(T / h)
    (price_f, std_f), t_f = timed(mc_european_cir_put, X0, T, K, A, B, C,
                                    N_PATHS, method='fte',
                                    n_steps=n_steps, seed=SEED_BASE)
    euro_rows.append({
        'Method': 'FTE',
        'h': f'1/{h_inv}' if h_inv > 1 else '1',
        'n_steps': n_steps,
        'Price': price_f,
        'StdErr': std_f,
        'RelErr': abs(price_f - exact_price) / exact_price,
        'CPU (s)': t_f,
    })

# QE-M with h = 1, 1/2, 1/4
for h_inv in [1, 2, 4]:
    h = 1.0 / h_inv if h_inv > 1 else 1.0
    n_steps = int(T / h)
    (price_q, std_q), t_q = timed(mc_european_cir_put, X0, T, K, A, B, C,
                                    N_PATHS, method='qem',
                                    n_steps=n_steps, seed=SEED_BASE)
    euro_rows.append({
        'Method': 'QE-M',
        'h': f'1/{h_inv}' if h_inv > 1 else '1',
        'n_steps': n_steps,
        'Price': price_q,
        'StdErr': std_q,
        'RelErr': abs(price_q - exact_price) / exact_price,
        'CPU (s)': t_q,
    })

df_euro = pd.DataFrame(euro_rows)
df_euro.to_csv(os.path.join(RESULTS_DIR, 'exp3_european_table.csv'), index=False)
print("\nEuropean put pricing table:")
print(df_euro.to_string(index=False, float_format='%.6f'))


# ── Asian pricing – M = 10 (yearly fixings) ──────────────────────────────────
print("\n" + "-" * 60)
print("Asian put  M=10 (yearly fixings, T=10)")
print("-" * 60)

M10 = 10

(price_ae, std_ae), t_ae = timed(mc_asian_cir_put, X0, T, K, A, B, C, M10,
                                   N_PATHS, method='exact', seed=SEED_BASE)
asian10_rows = [{'Method': 'Exact', 'h': '—', 'n_steps_per_int': '—',
                 'Price': price_ae, 'StdErr': std_ae, 'CPU (s)': t_ae}]

# Each monitoring interval has length T/M=1 year.
# h = 1 → 1 step/interval; h=1/2 → 2; h=1/4 → 4
for h_inv in [1, 2, 4]:
    n_sub = h_inv   # steps per interval (interval length = 1yr, h = 1/h_inv)
    for meth, mname in [('fte', 'FTE'), ('qem', 'QE-M')]:
        (price_m, std_m), t_m = timed(mc_asian_cir_put, X0, T, K, A, B, C,
                                       M10, N_PATHS, method=meth,
                                       n_steps=n_sub, seed=SEED_BASE)
        asian10_rows.append({
            'Method': mname,
            'h': f'1/{h_inv}' if h_inv > 1 else '1',
            'n_steps_per_int': n_sub,
            'Price': price_m,
            'StdErr': std_m,
            'CPU (s)': t_m,
        })

df_a10 = pd.DataFrame(asian10_rows)
df_a10.to_csv(os.path.join(RESULTS_DIR, 'exp3_asian_M10_table.csv'), index=False)
print(df_a10.to_string(index=False, float_format='%.6f'))


# ── Asian pricing – M = 40 (quarterly fixings) ───────────────────────────────
print("\n" + "-" * 60)
print("Asian put  M=40 (quarterly fixings, T=10)")
print("-" * 60)

M40 = 40

(price_ae40, std_ae40), t_ae40 = timed(mc_asian_cir_put, X0, T, K, A, B, C,
                                        M40, N_PATHS, method='exact',
                                        seed=SEED_BASE)
asian40_rows = [{'Method': 'Exact', 'h': '—', 'n_steps_per_int': '—',
                 'Price': price_ae40, 'StdErr': std_ae40, 'CPU (s)': t_ae40}]

# Each monitoring interval has length T/M = 0.25 yr (= 1 quarter).
# h=1/4 → 1 step/interval; h=1/8 → 2; h=1/16 → 4
for h_inv in [4, 8, 16]:
    n_sub = h_inv // 4   # steps per interval
    for meth, mname in [('fte', 'FTE'), ('qem', 'QE-M')]:
        (price_m, std_m), t_m = timed(mc_asian_cir_put, X0, T, K, A, B, C,
                                       M40, N_PATHS, method=meth,
                                       n_steps=n_sub, seed=SEED_BASE)
        asian40_rows.append({
            'Method': mname,
            'h': f'1/{h_inv}',
            'n_steps_per_int': n_sub,
            'Price': price_m,
            'StdErr': std_m,
            'CPU (s)': t_m,
        })

df_a40 = pd.DataFrame(asian40_rows)
df_a40.to_csv(os.path.join(RESULTS_DIR, 'exp3_asian_M40_table.csv'), index=False)
print(df_a40.to_string(index=False, float_format='%.6f'))


# ── Figure 3 style: convergence of European pricing error vs path count ───────
print("\n" + "-" * 60)
print("Convergence plot: European put error vs N_paths")
print("-" * 60)

path_counts = [1_000, 3_000, 10_000, 30_000, 100_000]
methods_conv = [
    ('exact', None, 'Exact (ncx2)', 'o-'),
    ('fte',  int(T / 1.0),  'FTE h=1',    's--'),
    ('fte',  int(T / 0.25), 'FTE h=1/4',  '^--'),
    ('qem',  int(T / 1.0),  'QE-M h=1',   'D:'),
    ('qem',  int(T / 0.25), 'QE-M h=1/4', 'x:'),
]

fig, ax = plt.subplots(figsize=(8, 5))
for meth, n_steps, label, fmt in methods_conv:
    errs = []
    for N in path_counts:
        kw = dict(n_steps=n_steps) if n_steps is not None else {}
        price_n, _ = mc_european_cir_put(X0, T, K, A, B, C, N,
                                          method=meth, seed=SEED_BASE, **kw)
        errs.append(abs(price_n - exact_price) / exact_price)
    ax.loglog(path_counts, errs, fmt, label=label, markersize=5)

ax.set_xlabel('Number of paths N')
ax.set_ylabel('Relative error')
ax.set_title('Experiment 3 – European put convergence vs N (CIR, K=0.09, T=10)')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
conv_path = os.path.join(RESULTS_DIR, 'exp3_convergence.png')
fig.savefig(conv_path, dpi=150)
plt.close(fig)
print(f"Saved convergence plot → {conv_path}")


# ── Append to RESULTS.md ─────────────────────────────────────────────────────
md_path = os.path.join(RESULTS_DIR, 'RESULTS.md')
with open(md_path, 'a') as fh:
    fh.write("\n\n## Experiment 3 – CIR Pricing\n\n")
    fh.write(f"Parameters: X0={X0}, a={A}, b={B}, c={C}, "
             f"delta={4*A/C**2}, T={T}, K={K}, N={N_PATHS}\n\n")
    fh.write(f"Analytical (benchmark) European put = {exact_price:.8f}\n\n")
    fh.write("### European put (N=100 000)\n\n")
    fh.write(df_euro.to_markdown(index=False, floatfmt='.6f') + '\n\n')
    fh.write("### Asian put M=10 (yearly)\n\n")
    fh.write(df_a10.to_markdown(index=False, floatfmt='.6f') + '\n\n')
    fh.write("### Asian put M=40 (quarterly)\n\n")
    fh.write(df_a40.to_markdown(index=False, floatfmt='.6f') + '\n\n')
    fh.write(f"Convergence plot: ![exp3_convergence](exp3_convergence.png)\n")

print(f"\nResults appended to {md_path}")
print("Experiment 3 complete.")
