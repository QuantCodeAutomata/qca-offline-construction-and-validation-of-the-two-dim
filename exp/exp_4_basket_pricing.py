"""
Experiment 4 – Basket CIR put pricing.

Prices E[(K - sum_i w_i X_T^i)^+] for a basket of d=5 independent CIR assets.

NOTE: Table 5 from the paper is not reproduced verbatim (exact parameters not
available). The four parameter cases below are representative and chosen to
span different delta / mean-reversion / volatility regimes.

Parameter cases
---------------
Case 1: all assets a=0.045, b=-0.5, c=1.0  (delta=0.18)
Case 2: all assets a=0.045, b=-0.3, c=1.0  (delta=0.18, slower reversion)
Case 3: all assets a=0.045, b=-0.5, c=1.2  (delta≈0.125, higher vol)
Case 4: mixed – 3 assets (a=0.045,b=-0.5,c=1.0), 2 assets (a=0.045,b=-0.3,c=1.2)

Common settings
---------------
d=5, T=10, K=0.09, w_i=1/5, X0_i=0.09, N_paths=100 000

NOTE: Paper uses N=10^6 paths; N=100 000 used here for tractability.

Outputs
-------
results/exp4_basket_table.csv
results/exp4_ks_test.csv
results/exp4_marginal_check.png
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
from scipy.stats import ncx2, ks_2samp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cir_model import cir_transition_params, sample_cir_terminal
from pricing import mc_basket_cir_put

# ── constants ────────────────────────────────────────────────────────────────
D = 5
T, K = 10.0, 0.09
X0 = 0.09
WEIGHTS = np.full(D, 1.0 / D)
N_PATHS = 100_000
SEED_BASE = 2024
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── parameter cases ───────────────────────────────────────────────────────────
CASES = {
    'Case 1 (baseline)': [(0.045, -0.5, 1.0)] * D,
    'Case 2 (slow rev)': [(0.045, -0.3, 1.0)] * D,
    'Case 3 (high vol)': [(0.045, -0.5, 1.2)] * D,
    'Case 4 (mixed)':    ([(0.045, -0.5, 1.0)] * 3
                          + [(0.045, -0.3, 1.2)] * 2),
}
X0_VEC = np.full(D, X0)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ── basket pricing table ──────────────────────────────────────────────────────
print("=" * 60)
print("Experiment 4 – Basket CIR put pricing")
print("=" * 60)

rows = []
for case_name, params_list in CASES.items():
    print(f"\n{case_name}")

    # Exact MC (direct inversion via ncx2)
    (p_e, s_e), t_e = timed(mc_basket_cir_put, X0_VEC, T, K, WEIGHTS,
                              params_list, N_PATHS, method='exact',
                              seed=SEED_BASE)
    rows.append({'Case': case_name, 'Method': 'Exact', 'n_steps': '—',
                 'Price': p_e, 'StdErr': s_e, 'CPU (s)': t_e})
    print(f"  Exact:     price={p_e:.6f}  se={s_e:.6f}  t={t_e:.2f}s")

    # FTE benchmark h=1/10  (100 steps total)
    n_fte = int(T / 0.1)
    (p_f, s_f), t_f = timed(mc_basket_cir_put, X0_VEC, T, K, WEIGHTS,
                              params_list, N_PATHS, method='fte',
                              n_steps=n_fte, seed=SEED_BASE)
    rows.append({'Case': case_name, 'Method': 'FTE h=1/10', 'n_steps': n_fte,
                 'Price': p_f, 'StdErr': s_f, 'CPU (s)': t_f})
    print(f"  FTE h=1/10: price={p_f:.6f}  se={s_f:.6f}  t={t_f:.2f}s")

    # QE-M h=1/4  (40 steps total)
    n_qem = int(T / 0.25)
    (p_q, s_q), t_q = timed(mc_basket_cir_put, X0_VEC, T, K, WEIGHTS,
                              params_list, N_PATHS, method='qem',
                              n_steps=n_qem, seed=SEED_BASE)
    rows.append({'Case': case_name, 'Method': 'QE-M h=1/4', 'n_steps': n_qem,
                 'Price': p_q, 'StdErr': s_q, 'CPU (s)': t_q})
    print(f"  QE-M h=1/4: price={p_q:.6f}  se={s_q:.6f}  t={t_q:.2f}s")

df_basket = pd.DataFrame(rows)
df_basket.to_csv(os.path.join(RESULTS_DIR, 'exp4_basket_table.csv'), index=False)
print("\nBasket pricing table saved.")


# ── KS test: validate marginals vs exact ncx2 ────────────────────────────────
print("\n" + "-" * 60)
print("KS test: marginals of FTE and QE-M vs exact ncx2")
print("-" * 60)

rng_ks = np.random.default_rng(SEED_BASE + 1)
case1_params = CASES['Case 1 (baseline)']
ks_rows = []

for i, (a_i, b_i, c_i) in enumerate(case1_params):
    # Exact samples for asset i
    exact_samples = sample_cir_terminal(X0, T, a_i, b_i, c_i, N_PATHS,
                                         seed=int(rng_ks.integers(0, 2**31)))

    # FTE samples (h=1/10)
    from cir_model import simulate_fte
    fte_samples = simulate_fte(X0, T, a_i, b_i, c_i, N_PATHS,
                                n_steps=int(T / 0.1),
                                seed=int(rng_ks.integers(0, 2**31)))

    # QE-M samples (h=1/4)
    from cir_model import simulate_qem
    qem_samples = simulate_qem(X0, T, a_i, b_i, c_i, N_PATHS,
                                n_steps=int(T / 0.25),
                                seed=int(rng_ks.integers(0, 2**31)))

    ks_fte, p_fte = ks_2samp(exact_samples, fte_samples)
    ks_qem, p_qem = ks_2samp(exact_samples, qem_samples)
    ks_rows.append({'Asset': i + 1,
                    'KS_FTE': ks_fte, 'p_FTE': p_fte,
                    'KS_QEM': ks_qem, 'p_QEM': p_qem})
    print(f"  Asset {i+1}: KS_FTE={ks_fte:.4f}(p={p_fte:.3f})  "
          f"KS_QEM={ks_qem:.4f}(p={p_qem:.3f})")

df_ks = pd.DataFrame(ks_rows)
df_ks.to_csv(os.path.join(RESULTS_DIR, 'exp4_ks_test.csv'), index=False)


# ── Marginal density plot (Case 1, Asset 1) ───────────────────────────────────
a0, b0, c0 = case1_params[0]
delta0, scale0, lam0 = cir_transition_params(X0, T, a0, b0, c0)

rng_plot = np.random.default_rng(SEED_BASE + 99)
exact_plot = sample_cir_terminal(X0, T, a0, b0, c0, N_PATHS,
                                  seed=int(rng_plot.integers(0, 2**31)))
fte_plot   = simulate_fte(X0, T, a0, b0, c0, N_PATHS, n_steps=int(T / 0.1),
                           seed=int(rng_plot.integers(0, 2**31)))
qem_plot   = simulate_qem(X0, T, a0, b0, c0, N_PATHS, n_steps=int(T / 0.25),
                           seed=int(rng_plot.integers(0, 2**31)))

x_grid = np.linspace(0.0, np.percentile(exact_plot, 99.5), 300)
pdf_exact = ncx2.pdf(x_grid / scale0, delta0, float(lam0)) / scale0

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, np.percentile(exact_plot, 99.5), 80)
ax.hist(exact_plot, bins=bins, density=True, alpha=0.35,
        label='Exact (ncx2)', color='steelblue')
ax.hist(fte_plot,   bins=bins, density=True, alpha=0.35,
        label='FTE h=1/10', color='orange')
ax.hist(qem_plot,   bins=bins, density=True, alpha=0.35,
        label='QE-M h=1/4', color='green')
ax.plot(x_grid, pdf_exact, 'k-', lw=2, label='Exact PDF')
ax.set_xlabel('X_T')
ax.set_ylabel('Density')
ax.set_title('Experiment 4 – Marginal distribution check (Case 1, Asset 1, T=10)')
ax.legend(fontsize=8)
ax.set_xlim(left=0)
fig.tight_layout()
marg_path = os.path.join(RESULTS_DIR, 'exp4_marginal_check.png')
fig.savefig(marg_path, dpi=150)
plt.close(fig)
print(f"\nMarginal density plot saved → {marg_path}")


# ── Sensitivity: Case 1 without coupling (independent assets) ─────────────────
# In our implementation each asset is already independently sampled.
# We demonstrate by confirming the basket price is stable under different seeds.
print("\n" + "-" * 60)
print("Sensitivity: basket price stability (Case 1, exact, different seeds)")
print("-" * 60)
sens_prices = []
for s in range(5):
    p_s, _ = mc_basket_cir_put(X0_VEC, T, K, WEIGHTS,
                                 case1_params, N_PATHS, method='exact',
                                 seed=SEED_BASE + s * 100)
    sens_prices.append(p_s)
    print(f"  seed={SEED_BASE + s * 100}: price={p_s:.6f}")
print(f"  Mean={np.mean(sens_prices):.6f}  Std={np.std(sens_prices):.6f}")


# ── Append to RESULTS.md ─────────────────────────────────────────────────────
md_path = os.path.join(RESULTS_DIR, 'RESULTS.md')
with open(md_path, 'a') as fh:
    fh.write("\n\n## Experiment 4 – Basket CIR Put Pricing\n\n")
    fh.write(f"d={D}, T={T}, K={K}, w_i=1/{D}, X0={X0}, N={N_PATHS}\n\n")
    fh.write("*Note: representative parameter cases (Table 5 not reproduced verbatim).*\n\n")
    fh.write("### Basket prices\n\n")
    fh.write(df_basket.to_markdown(index=False, floatfmt='.6f') + '\n\n')
    fh.write("### KS tests – Case 1 marginals\n\n")
    fh.write(df_ks.to_markdown(index=False, floatfmt='.4f') + '\n\n')
    fh.write(f"Marginal plot: ![exp4_marginal](exp4_marginal_check.png)\n")

print(f"\nResults appended to {md_path}")
print("Experiment 4 complete.")
