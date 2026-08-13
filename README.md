# FunctionalCurves

Research code for simulating dependent (mixing) time series and estimating **Tukey's halfspace depth** and the associated minimal direction at a point, both empirically (via simulation) and analytically (for Gaussian / VAR(1) processes). Used to study how fast empirical depth/direction estimates converge to their theoretical values as sample size grows.

This is research/exploration code, not a published package. `pyproject.toml` exists only to make `functionalcurves` importable in an editable local install and to run the test suite — there is no intent to publish this to PyPI.

## Structure

```
FunctionalCurves/
├── pyproject.toml                 # Local dependency/test config (editable install only)
├── functionalcurves/               # Core research code, importable locally
│   ├── __init__.py
│   ├── mixing_models.py            # Simulate mixing (weakly dependent) bivariate processes
│   └── depth.py                    # Compute Tukey depth / minimal direction, empirically and analytically
├── notebooks/                      # Interactive demos
│   ├── example_var1.ipynb          # Demo: VAR(1) process + empirical depth estimation & convergence
│   ├── example_polynomial.ipynb    # Demo: MixingLinearModel process + Gaussian analytic depth comparison
│   └── scratch_interactive.ipynb   # Scratch/interactive notebook, mostly duplicated commented-out code
├── scripts/
│   └── donsker_cdf_scratch.py      # Scratch script: empirical CDF convergence (Donsker) plots
└── tests/                          # pytest unit tests for the package's public functions
```

## Modules

### `mixing_models.py`
Generates synthetic bivariate sample paths with controllable dependence ("mixing rate").

- `transition_markov(X, p, e)` / `transition_diff(X, p, e)` — transition functions defining how the next state depends on the previous state and an innovation `e`.
- `MixingMarkovModel` — simple Markov-chain-style simulator: repeatedly applies a transition function to Gaussian innovations.
- `MixingLinearModel` — simulates a linearly-weighted mixing process where weights decay as `k^-mixing_rate`; includes an error-correction term (`zeta`/`zeta_k`, via `scipy.special.zeta`) to control approximation error from truncating an infinite sum, and a `distribution()` method giving the theoretical mean/covariance of the limiting process.
- `CovHC_` — placeholder for a heteroscedasticity-consistent covariance estimator (unimplemented).
- `if __name__ == '__main__'` block — example script simulating a process and plotting estimated depth/direction vs. sample size (uses `depth.py`).

### `depth.py`
Core depth/direction estimation logic.

- `rad(v1, v2, ...)` — signed angle in `[0, 2π)` between vectors (used throughout for directional comparisons).
- `rad_wu`, `rad_dec` — related/deprecated angle helpers.
- `GaussianDepth(mean, cov, X0)` — closed-form Tukey depth and minimal direction for a bivariate normal distribution.
- `Estimator` — empirical depth/direction estimator given a sample `X` and a point `X0`. Two methods:
  - `'deg'` — scans candidate directions uniformly over `[-π, π]`.
  - `'point_wise'` — uses each sample point's own direction as a candidate.
  Returns the minimal depth, the corresponding direction, and the full arrays of angles/depths/directions considered.
- `TDE` — deprecated older counting-based depth estimator, superseded by `Estimator`.
- `Analytic_Depth` — computes the exact Tukey depth and minimal direction for a stationary VAR(1) process (`X_t = A0 + A1 X_{t-1} + noise`), by deriving the process's stationary mean/covariance and applying the Gaussian depth formula (`TD_analytic`).
- `Depth` class — thin wrapper stub intended to expose a `halfspace` depth method (currently unimplemented / placeholder).
- `if __name__ == '__main__'` block — simulates a VAR(1) process via `statsmodels`, computes the true analytic depth/direction, and compares against empirical estimates (`'deg'` vs `'point_wise'`) as sample size increases, plotting convergence.

### `scripts/donsker_cdf_scratch.py`
Standalone scratch script unrelated to the depth machinery: plots the empirical CDF process `√N (F_N(x) - F(x))` for uniform samples of increasing size, illustrating Donsker's theorem.

## Notebooks

- **`notebooks/example_var1.ipynb`** — Simulates a VAR(1) process, computes the analytic Tukey depth/direction at a chosen point `X0`, then compares against empirical estimates (`Estimator`, both methods) across increasing sample sizes, with convergence and directional-vector plots. Also has a section using `mixing_models.MixingModel` (transition-based simulation).
- **`notebooks/example_polynomial.ipynb`** — Same style of analysis but driving the sample from `MixingLinearModel` instead of a VAR process, comparing empirical estimates against the Gaussian analytic depth (`GaussianDepth`) using the model's theoretical stationary distribution.
- **`notebooks/scratch_interactive.ipynb`** — Exploratory/scratch notebook; most cells are commented-out duplicates of the empirical-CDF script from `scripts/donsker_cdf_scratch.py`, plus an interactive version. Not core to the library.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `statsmodels` (for `VARProcess`), `tqdm`. A commented-out import suggests intended (but currently unused/unavailable) integration with a `depth` package (`depth.multivariate`) for halfspace depth via compiled libraries.

## Local setup

This is not published anywhere; install it locally in editable mode to make `functionalcurves` importable and run the tests:

```bash
pip install -e ".[dev]"
```

## Testing

Focused unit tests for the core public functions (`rad`, `GaussianDepth`,
`Analytic_Depth`, `Estimator` in `depth.py`; the transition functions,
`MixingMarkovModel`, and `MixingLinearModel` in `mixing_models.py`) live in
`tests/`. Run them with:

```bash
pytest
```

## Usage

Both `functionalcurves/mixing_models.py` and `functionalcurves/depth.py` are runnable as scripts (`python -m functionalcurves.depth`, `python -m functionalcurves.mixing_models`) and contain end-to-end examples: simulate a process, estimate depth/direction over growing sample sizes, and plot convergence against the true/analytic values. The notebooks in `notebooks/` walk through the same workflow interactively.
