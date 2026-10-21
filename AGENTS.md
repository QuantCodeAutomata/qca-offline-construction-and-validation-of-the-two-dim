# AGENTS.md – CIR Pricing & Calibration Project

Persistent notes for AI agents working on this repository.

---

## Project Layout

```
src/
  cir_model.py        CIR SDE: exact transition (ncx2), FTE, QE-M simulation
  pricing.py          exact European put (closed form), MC European/Asian/basket
  calibration.py      LM calibration with FD Jacobian; synthetic target generation
exp/
  exp_3_cir_pricing.py   European & Asian put convergence tables
  exp_4_basket_pricing.py  basket put across 4 parameter cases
  exp_5_calibration.py    100-trial European LM study; 20-trial Asian LM study
tests/
  test_cir_model.py   22 tests – delta, C, transition, FTE, QE-M
  test_pricing.py     20 tests – exact put, MC European/Asian/basket
  test_calibration.py 16 tests – gradient, synthetic targets, LM convergence
results/              CSV, PNG, RESULTS.md (appended by each exp script)
```

## CIR Convention

`dX = (a + b*X) dt + c*sqrt(X) dW`  with **b < 0** (mean-reverting).

Transition X_s → X_t (tau = t - s):
```
scale = c^2 * (1 - exp(b*tau)) / (-4*b)   # > 0 since b < 0
lam   = X_s * 4*(-b)*exp(b*tau) / (c^2*(1 - exp(b*tau)))
delta = 4*a / c^2

X_t ~ scale * ncx2(df=delta, nc=lam)
```

Long-run mean: `-a/b`.  Feller condition: `2a > c^2` (delta > 2).

## Exact European Put – Closed Form

Using the ncx2 moment identity `E[Y * 1_{Y≤t}] = delta * F(t; delta+2, lam) + lam * F(t; delta+4, lam)`:

```python
price = K * ncx2.cdf(K/scale, delta, lam)
      - scale * (delta * ncx2.cdf(K/scale, delta+2, lam)
               + lam   * ncx2.cdf(K/scale, delta+4, lam))
```

Three CDF calls only (~0.25 ms). **Do not use scipy.integrate.quad** (was 30 ms/call
– 120× slower and no longer used).

## LM Calibration Design

- `theta = (delta, b, c)`,  constraints: delta > 0, b < 0, c > 0.
- `_price_vec` evaluates all (K, T) instruments for a given theta.
- `_build_jacobian` takes `current_prices` (already computed) so no re-evaluation
  of the base point; only 2n_params price-vector calls for central FD.
- **Jacobian is reused on rejected LM steps** – only recomputed on acceptance.
  This avoids 6×n_instruments pricing calls per rejection.

## Experiment Settings (Tractability Notes)

| Experiment | n_paths | Note |
|---|---|---|
| exp_3 European MC | 100 000 | Paper uses 1M |
| exp_3 Asian MC    | 100 000 | Paper uses 1M |
| exp_4 basket      | 100 000 | Paper uses 1M |
| exp_5 European LM | 50 000 per FD eval | 100 trials, ~25 s total |
| exp_5 Asian LM    | 5 000 per FD eval  | 20 trials, ~105 s; full 100-trial run ≈50 min |

## Test Invariants

- FTE can produce slightly negative X_T (sub-Feller, delta=0.18 < 2).  MC European
  price can slightly exceed K in FTE; test uses `price < 2*K` for that method only.
- QE-M always produces non-negative samples; price ≤ K holds.
- Exact formula always gives price ∈ [0, K].

## Key Parameters Used in Experiments

```
X0 = 0.09,  a = 0.045,  b = -0.50,  c = 1.0
delta = 0.18,  T = 10,  K = 0.09
Strikes    = [0.065, 0.085, 0.090, 0.100, 0.110, 0.120]
Maturities = [6, 7, 8, 9, 10]
```

## Dependencies

Python 3.12, numpy 1.26, scipy 1.14, pandas 2.2, matplotlib 3.11.

`src/chi_square_sampler.py` is **not yet present** (reserved for a Chebyshev
sampler by another agent). Pricing currently uses `scipy.stats.ncx2` and
`numpy.random.Generator.noncentral_chisquare` directly (the "direct inversion"
method). When the Chebyshev sampler is added, hook it into `sample_cir_terminal`
and `sample_cir_path` as an alternative `method='chebyshev'`.

## Running

```bash
# Tests (fast – 3–4 s)
python -m pytest tests/ -v

# Experiments
python exp/exp_3_cir_pricing.py   # ~15 s
python exp/exp_4_basket_pricing.py # ~30 s
python exp/exp_5_calibration.py    # ~130 s
```
