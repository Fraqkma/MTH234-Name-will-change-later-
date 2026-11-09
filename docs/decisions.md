# Phase 0 Interface and Method Decisions

**Status: architecture approved by the requester; scientific interface/method sign-off remains pending from P2-P5.** These notes preserve the Phase 0 contract while recording the requested frontend migration. Finalize shared solver and experiment APIs before parallel implementation.

## Architectural migration

On 2026-11-09, the project requester replaced the planned Streamlit UI with a responsive vanilla HTML/CSS/JavaScript site served by FastAPI. Plotly.js is bundled locally for offline charts. The API is a thin adapter over `ridge_lab`; the numerical package remains independent of FastAPI and browser code. This changes the application architecture, not the project's mathematical question, scientific safeguards, six-member allocation, or deadline.

## Data and split contract

- `make_dataset(seed, noise_std, n_train, n_val, n_test)` generates independent samples from `sin(2*pi*x) + Gaussian noise` on `[0, 1]`.
- The returned bundle contains one-dimensional `x`, `y`, and stable split-prefixed sample IDs for `train`, `validation`, and `test`; arrays are read-only.
- Default proposal: seed `42`, noise standard deviation `0.20`, counts `24/24/120`. `config/experiment.yaml` marks these values provisional until the first official evaluation is frozen.
- Fit coefficients on train; use validation for model and lambda selection; reserve test for the final frozen evaluation.

## Feature and solver contract

- `make_A(x, degree)` is proposed to return an `(n_samples, degree + 1)` matrix with an all-ones intercept in column zero and powers of `x` in the remaining columns.
- Any polynomial scaling is fit on training features only and then reused unchanged for validation and test. Ridge and OLS use the same transformed features.
- Production OLS uses `numpy.linalg.lstsq` or QR; the normal equations are for mathematical explanation and guarded demonstrations, not explicit inversion.
- Ridge uses `||y - A w||_2^2 + lambda * sum(w[j]**2 for j > 0)`, so the intercept is unpenalized and `lambda >= 0`. `lambda=0` is OLS on the same features.
- MSE is the mean squared residual. Loss normalization must be stated if any experiment uses a different convention.

## Dependency and ownership boundaries

- `src/ridge_lab/` owns data, features, solvers, metrics, diagnostics, experiments, and evaluation; it must not import FastAPI, Plotly, or frontend code.
- `api/` validates requests and serializes results; `frontend/` renders reusable outputs and must not fit models or construct the scientific design matrix.
- Proposed API names and signatures remain subject to P2-P5 review before parallel implementation.

Design guidance adapted from Unslop by Damir Sokolovsky under CC BY 4.0.
