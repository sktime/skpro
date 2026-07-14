# Bayesian Modelling API — Proof of Concept

This PoC demonstrates the core architecture for a Bayesian modelling API layer
in skpro, as proposed for GSoC 2026.

## What this PoC contains

| File | Purpose |
|------|---------|
| `_base.py` | `BaseBayesianRegressor` — mixin that adds prior/posterior/update interfaces to `BaseProbaRegressor` |
| `_prior.py` | `Prior` — wraps skpro distributions as parameter priors (backend-agnostic) |
| `_ridge.py` | `BayesianRidgeRegressor` — working estimator with evidence maximization, posterior access, and sequential updating |
| `tests/test_bayesian_ridge.py` | Tests covering fit, predict, posterior API, sequential updates, and Prior class |

## Core idea

The central mechanism is a **`BaseBayesianRegressor`** mixin that extends skpro's
existing `BaseProbaRegressor` with a standardized Bayesian interface:

- `get_prior()` / `set_prior()` — inspect and configure prior distributions
- `get_posterior()` / `get_posterior_summary()` — access fitted posterior
- `sample_posterior(n)` — draw from the parameter posterior
- `update(X, y)` — sequential Bayesian update (posterior becomes new prior)

Priors are specified using skpro's own 68+ distribution classes (via the `Prior`
wrapper), so the same `Normal`, `HalfCauchy`, `InverseGamma`, etc. work as both
priors and predictive distributions — no separate prior DSL needed.

The `BayesianRidgeRegressor` demonstrates this in action: it performs Type-II
maximum likelihood (evidence maximization) to optimize hyperparameters, computes
the closed-form posterior, and returns standard `BaseDistribution` objects from
`predict_proba` — fully compatible with skpro's metrics, pipelines, and model
selection tools.

## How to run locally

```bash
# From the repository root
pip install -e .

# Run the tests
pytest skpro/regression/bayesian/tests/test_bayesian_ridge.py -v -o "addopts="

# Quick smoke test
python3 -c "
from skpro.regression.bayesian import BayesianRidgeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = BayesianRidgeRegressor()
reg.fit(X_train, y_train)

# Full predictive distribution
dist = reg.predict_proba(X_test)
print('Predictive mean shape:', dist.mean().shape)
print('Predictive std sample:', dist.var().values[:3, 0] ** 0.5)

# Bayesian-specific API
print(reg.get_posterior_summary())
"
```

## What this does NOT include

- No UI, no notebooks, no full pipeline
- No MCMC/VI inference (the full project adds these)
- No survival models (planned for Phase 3)
- No new distribution classes (skpro's existing 68+ are sufficient)
