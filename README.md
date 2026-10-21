# Chebyshev Inverse Chi-Square Sampler & CIR Pricing

A Python implementation of the two-dimensional Chebyshev approximation of the
inverse central chi-square CDF (`F_δ⁻¹(u)`) and its application to exact-
transition Monte Carlo pricing and Levenberg–Marquardt calibration of
Cox–Ingersoll–Ross (CIR) interest-rate models.

## Repository Structure

```
├── src/                       # Core library
│   ├── chebyshev.py           # 2D Chebyshev approximation & coefficient generation
│   ├── chi_square_sampler.py  # Central / non-central chi-square samplers
│   ├── cir_model.py           # CIR SDE: exact, FTE, QE-M transition
│   ├── pricing.py             # European & Asian put pricing (exact + MC)
│   ├── calibration.py         # Levenberg–Marquardt calibration utilities
│   └── coefficients/          # Auto-generated JSON coefficient files
│
├── exp/                       # Experiment scripts
│   ├── exp_1_chebyshev_sampler.py    # Build & validate Chebyshev sampler
│   ├── exp_2_sampling_benchmark.py   # CPU benchmark & moment validation
│   ├── exp_3_cir_pricing.py          # European & Asian put pricing tables
│   ├── exp_4_basket_pricing.py       # Multi-asset basket option pricing
│   └── exp_5_calibration.py          # CIR calibration study
│
├── tests/                     # pytest test suite
│   ├── test_chebyshev.py
│   ├── test_chi_square_sampler.py
│   ├── test_cir_model.py
│   ├── test_pricing.py
│   └── test_calibration.py
│
├── data/                      # Data directory (no empirical data required)
├── results/                   # Experiment outputs (CSV, PNG, RESULTS.md)
└── AGENTS.md                  # Persistent AI agent memory
```

## CIR Model Convention

```
dX = (a + b·X) dt + c·√X dW,   b < 0 (mean-reverting)
```

Exact transition: `X_{t+τ} ~ scale · χ²_δ(λ)` where

```
scale = c²(1 - e^{bτ}) / (4|b|)
λ     = X_t · 4|b|·e^{bτ} / (c²(1 - e^{bτ}))
δ     = 4a / c²
```

## Experiments

| Experiment | Description |
|---|---|
| **exp_1** | Build Chebyshev coefficient tables; validate accuracy O(10⁻⁸) and derivatives |
| **exp_2** | CPU benchmark: direct inversion vs Ahrens-Dieter; non-central moment validation |
| **exp_3** | European & Asian CIR put pricing: Exact vs FTE vs QE-M |
| **exp_4** | Five-asset basket put pricing with coupling strategy |
| **exp_5** | LM calibration to synthetic European & Asian prices (100 trials each) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run all experiments (in order; exp_1 must run first to generate coefficients)
python exp/exp_1_chebyshev_sampler.py
python exp/exp_2_sampling_benchmark.py
python exp/exp_3_cir_pricing.py
python exp/exp_4_basket_pricing.py
python exp/exp_5_calibration.py
```

Results are saved in `results/`. See `results/RESULTS.md` for a summary.

## Dependencies

- numpy, scipy, matplotlib, pandas
- mpmath (optional, for high-precision validation)

## Key References

- Chebyshev approximation: three-region piecewise inverse chi-square sampler
  with region boundaries u ∈ {[0,0.8], [0.8,0.95], [0.95,1-10⁻⁸]}
- Region transforms: identity / log-Gamma / log-log-Gamma
- Truncation orders (O(10⁻⁸)): see `src/chebyshev.py::TRUNCATION_ORDERS`
- Non-central decomposition: Poisson splitting at λ=10 (Malham & Wiese)
- CIR exact pricing: ncx2 scaling from squared-Bessel representation
