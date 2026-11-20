> Current owner decision (2026-11-19): Nipitpon is the sole human reviewer. This supersedes project-managed requests below for separate Sitthawee, Sirapobe, or independent reviewer sign-offs. G0 acknowledgments remain unrecorded unless actually supplied. Protocol freeze and official holdout evaluation remain pending Nipitpon's consolidated final review.

# Phase 0 Interface and Method Decisions

**Status: architecture approved by the requester; scientific interface/method sign-off remains pending from P2-P5.** These notes preserve the Phase 0 contract while recording the requested frontend migration. Finalize shared solver and experiment APIs before parallel implementation.

## Confirmed Battle A and presentation decisions (2026-11-18)

- **Project title:** *The Least Squares Showdown*. Continue in this existing repository.
- **Real-world case:** UCI Capital Bike Sharing `hour.csv`, selected by the project owner after [the three-candidate audit](datasets_review.md). Its hourly bike-rental question is stronger for the presentation; the audit's simpler Airfoil recommendation remains recorded as history. This choice did not use model scores.
- **Prediction basis:** `cnt` rentals/hour from categorical `hr`, binary `workingday`, both main effects, and their interaction. One intercept, 23 hour indicators with hour 0 as the reference, one working-day effect, and 23 interaction columns. No weather, rental component counts, IDs, or other raw variables enter the design.
- **Evaluation:** fixed final 146-date future holdout; five expanding-window development folds on earlier dates, all hours of a date together. See [the exact split contract and development-only smoke metrics](bike_split_contract.md). No official holdout result has been opened. Synthetic degree selection and ten seeded runs remain separate.
- **Battle A ranking:** mean validation RMSE (rentals/hour) first, with MAE, R², and variation across windows beside it. User-adjusted tier weights are allowed only for clearly labeled exploratory rankings. The separate Battle B solver ranking remains independent.
- **Presentation:** open with “Can we predict how many bikes will be rented in a given hour?” Explain the interaction as the change in the hourly pattern between working and nonworking days. Keep a genuine two-parameter 3D loss surface as the main Battle B demonstration and preserve six approximately equal English-speaking segments.
- **Protocol-freeze status:** [Sitthawee and Sirapobe's review checklist](bike_split_contract.md#protocol-freeze-review--pending) is pending; no sign-off or official holdout score is recorded. The ten-seed requirement belongs to the synthetic experiment, not Battle A's chronological validation.

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

## Consolidated review implementation choices (2026-11-19)

- Normal bike loading now replaces reserved counts with NaN without converting their target text. Dates and predictor metadata remain available for split verification. The API computes actual development results and refit; the exploratory ranking now uses full precision metrics.
- No existing bounded Battle B configuration was found in the checkout. Added a proposed bounded benchmark: seed 42, shapes 100x8 and 500x16, prescribed condition numbers 10 and 1e6, identical matrices/targets across methods, three timed repeats, median wall time. Input bytes are reported, not peak memory. Gradient descent starts at zero, uses step 1/||A||^2 for half-SSE, gradient tolerance 1e-8, maximum 10,000 iterations. Iteration limits are retained as outcomes.
- Synthetic proposed sensitivity configuration: seeds 42 through 51, noise standard deviations 0.1/0.2/0.4, existing 24/24/120 counts, degrees 1 through 15, train-only scaling, Ridge 0 plus twelve log-spaced values 1e-8 through 1e3, Lasso 0.001/0.01/0.1/1. Select by validation MSE, then lower degree and penalty on exact ties. Nonconverged candidates are reported and excluded, never silently counted as successful. Synthetic test remains reserved because config/experiment.yaml is explicitly provisional; final review must freeze it before official evaluation.
- The loss surface is the full intercept/slope objective SSE + lambda*slope^2 on a deterministic 30-point illustrative linear dataset. It is separate from both bike prediction and the polynomial study.

## Lasso numerical resolution (2026-11-19)

Preserved the seeded data, 24/24/120 partitions, training-only scaling, polynomial and alpha grids, SSE/(2n)+alpha L1 objective with an unpenalized intercept, and unrounded validation-MSE selection rule. Baseline cyclic coordinate descent used 10,000 iterations and tolerance 1e-8; 365/1,800 configurations raised RuntimeError. Increasing only the cap to 50,000 still failed in two representative high-degree fits. The final solver retains the same tolerance and cyclic update, maintains the residual gradient incrementally, and checks the active-set least-squares solution against the full Lasso KKT conditions every 500 sweeps. If a coefficient would change sign, it steps to that boundary and resumes cyclic descent. The candidate cap is 50,000; all 1,800 final fits converged, maximum 30,500 iterations and maximum KKT residual 9.61e-9. A separate 1e-10 stricter fit agrees on validation predictions and MSE in representative low/high degrees. The full ten-seed run changed 21 Lasso degree selections but no aggregate model order. This is a numerical solver setting, not a protocol change.
