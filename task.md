> Current owner decision (2026-11-19): Nipitpon is the sole human reviewer. This supersedes project-managed requests below for separate Sitthawee, Sirapobe, or independent reviewer sign-offs. G0 acknowledgments remain unrecorded unless actually supplied. Protocol freeze and official holdout evaluation remain pending Nipitpon's consolidated final review.

# task.md — The Least Squares Showdown

> **Status:** Phase 0 scaffold and Battle A bike development path exist / Frontend uses HTML-CSS-JS + FastAPI / Official holdout, named ownership acknowledgments, and method sign-off pending
> **Updated:** 2026-11-18 (Asia/Bangkok)
> **Course:** Linear Algebra group project (6 members)  
> **Final submission:** **2026-12-01** — report (any format) + **10-minute video; six members speak equally**  
> **Source of truth:** the group's supplied README and Thai/English plans, supplemented by the corrections explicitly marked below.  
> **Confirmed language:** **English** for report, website, slides, narration, code, UI labels, comments, and documentation. Internal team chat may use Thai.
> **Repository decision:** Continue in this existing repository; no remote URL, GitHub owner, or visibility has been provided.
> **Leadership confirmed:** **Nipitpon Choojit — Project Manager**; **Sirapobe Parinyarat (Sirapob) — Technical Advisor / Research Guide**. Both have hands-on implementation responsibilities.
> **Other role assignments:** Working proposal from the six-person roster, to be acknowledged by members at kickoff; not independently confirmed by each member.

## 0. Mission and submission contract

**Confirmed title:** *The Least Squares Showdown*

**Research question:** Does the model with the smallest training error generalize best to new data, and how does L2 regularization change this trade-off?

Build a **locally runnable, interactive web demonstration** of polynomial regression using Least Squares and Ridge Regression. Show why high-degree polynomial fitting can overfit noisy data, select model complexity/regularization **without test-set leakage**, and compare against a genuinely untouched holdout set. Explain the links to Section 6.5 (Least Squares, projection, normal equations, QR) and Section 6.6 (Machine Learning and Linear Models) of the course textbook.

**Required finished deliverables**
- [ ] Reproducible Python implementation, packaged as `ridge_lab` and covered by automated tests.
- [ ] Working vanilla HTML/CSS/JavaScript + FastAPI web demo with adjustable degree and ridge `lambda`, data/fit plots, error curves, and explanations.
- [ ] Experimental results, figures and machine-readable configuration/metric tables.
- [ ] Coherent English academic report with theory, method, actual measured results, limitations, sources, and all six members' contributions.
- [ ] Final slides + **one 10:00-minute video**; six speakers approximately **1:40 each**, none omitted.
- [ ] Clear README setup and demonstration instructions; successful clean-machine installation/demo.

**Non-goals:** building a production ML platform, neural networks, paid datasets, public deployment before the local app passes checks, pretending Ridge always wins, or inventing results. **Use free/existing tooling.**

## 1. Corrections to the original group plan (decisions)

1. **Train / validation / test:** Old documents use train/test and choose `lambda` by test MSE. Replace with independent `train`, `validation`, and sealed `test`: train coefficients on train, choose degree/`lambda` on validation, use test once to report final held-out performance. Never pick a model by test performance.
2. **Mathematical objective:** Fix a single convention: `J(w)=||y-Aw||_2^2 + lambda * sum_{j=1}^{p-1} w_j^2`, with first coefficient as unpenalized intercept. Accordingly, use `P=diag(0,1,...,1)` and `(A.T @ A + lambda*P) w = A.T @ y`. Explain that `lambda*I` in the original plan corresponds to a different convention that also penalizes the intercept. `lambda=0` is OLS.
3. **Avoid direct inverse:** Present normal equations for theory but use `numpy.linalg.lstsq` (SVD-based) or QR for production OLS; guard ill-conditioned / singular normal-equation demos instead of treating floating-point failure as meaningful overfitting evidence. For ridge, use a numerically stable augmented least-squares formulation where feasible, and cross-check against the regularized normal equation on well-conditioned cases.
4. **Feature scaling:** Standardize non-intercept polynomial features using *training-only* statistics, reuse identical transforms on validation/test; compare OLS/Ridge on the same features. Record the basis and scaler in every experiment.
5. **Numerical instability is not overfitting:** Show `cond(A)` and optionally `cond(A.T @ A)` as **diagnostics**. A huge condition number is not by itself proof of poor generalization, and the train/validation MSE curves are not guaranteed to have a specific shape for every dataset.
6. **Ridge is conditional, not magic:** Show its bias–variance trade-off, an illustrative high-degree use case, and cases where too much `lambda` underfits. Report cases in which it does **not** improve holdout error as well.
7. **Team workflow:** Keep the six original subject areas, now with a **named six-person provisional allocation**, explicit code/report/demo ownership, interfaces, reviewers, dependencies and acceptance checks. **Nipitpon (P6)** coordinates delivery *and* implements the web shell/integration; **Sirapobe (P5)** guides technical direction *and* implements Ridge/validation logic. Neither is a supervisory-only role; other members own and deliver their work independently.
8. **Web scope:** The supplied documents assumed static Matplotlib graphs. The current frontend is vanilla HTML/CSS/JavaScript served by FastAPI, with locally bundled Plotly.js for interactive charts. Keep math/data/experiment modules independent of FastAPI and browser code.

### 1.1 Requester-approved frontend architecture migration (2026-11-09)

This migration replaces the originally proposed Streamlit application with a responsive single-page HTML/CSS/ES-module frontend served by FastAPI. The API is a thin adapter over `src/ridge_lab`; the numerical core imports neither FastAPI nor frontend code. Plotly.js is bundled locally so charts work without a CDN connection. The migration preserves the research question, scientific safeguards, original README, six-member allocation, and deadline. The currently exposed fit is an illustrative seeded example using train and validation data only; it is not a final experiment result. The demo does not return test observations or scores.

## 2. Recommended technology and conventions

- **Python:** prefer the team's existing Python 3.13 environment; target a broadly supported `>=3.11` release in `pyproject.toml` after dependency compatibility checks.
- **Runtime:** NumPy (math), FastAPI (local HTTP/static-file service), vanilla HTML/CSS/JavaScript, and locally bundled Plotly.js for interactive charts; Matplotlib for report exports if useful.
- **Development:** `pytest` and `ruff`; `scikit-learn` may be used as an **independent verification baseline**, not as the replacement for the hand-written OLS/Ridge methods required by this course.
- **Packaging:** `pyproject.toml` with `src/` layout and optional `dev` dependencies. Lock/record versions before final handoff.
- **Determinism:** default `seed=42`. Never resample data on UI widget reruns unless the user explicitly requests a new experiment.
- **Naming:** public math APIs preserve familiar names `make_A`, `ridge`, and `mse`; recommend one general `evaluate_split` rather than hard-wiring evaluation behavior into `train_mse` and `test_mse`.
- **Branches:** `feature/p1-theory`, `feature/p2-qr`, `feature/p3-dataset`, `feature/p4-overfitting`, `feature/p5-ridge`, `feature/p6-web-integration`; review through pull requests, no direct main commits.
- **Language:** English-only externally visible deliverables (report, website, slides, 10-minute video, README, figure labels and captions); no mixing Thai into final project artifacts.

### Current repository layout

```text
ridge-regression-lab/
??? AGENTS.md
??? task.md
??? README.md
??? pyproject.toml
??? .gitignore
??? .github/workflows/ci.yml
??? api/main.py                 # thin FastAPI adapter and static site server
??? config/experiment.yaml
??? frontend/
?   ??? index.html
?   ??? css/                    # design tokens, base, layout, components
?   ??? js/                     # API, state, page glue and chart modules
?   ??? vendor/                 # locally bundled Plotly.js and license
??? src/ridge_lab/              # framework-independent data and math core
??? tests/                      # core, API, integration and browser smoke checks
??? docs/
??? report/
??? slides/
```

**Architecture dependency direction (mandatory):** `datasets/features -> solvers -> experiments/evaluation -> api -> frontend -> report figures`. The core under `src/ridge_lab/` must **never import** FastAPI, frontend JavaScript, or display code. The API calls reusable core functions; browser code must not assemble polynomial design matrices or solve linear systems. Notebooks are scratchpads only.

## 3. Reproducible experiment protocol

### Primary illustrative dataset — fixed before looking at results
- True signal: `f(x) = sin(2*pi*x)`, x sampled within `[0,1]`.
- Observation: `y = f(x) + Gaussian noise`, initial `sigma = 0.20`, generator seed `42`.
- Proposed sizes (may be amended *before* first official evaluation): **24 train, 24 validation, 120 sealed test**. These are independently generated samples with explicit split identifiers, not separately shuffled copies of the same sample.
- Use a separate dense display grid for plotting the known true function, **not** for selecting hyperparameters.
- Candidate polynomial degrees: `1..15`; reference plots: `1, 3, 9, 15`.
- Ridge `lambda` candidates: `{0}` plus a documented log-spaced range (start proposal `10^-8..10^3`) on consistently standardized features.
- Record dataset seed, sigma, sample sizes, feature transformation, solver, degree, lambda, train/validation/test metrics, version and run timestamp. Serialize as JSON/CSV.

### Model selection and comparisons
A. **Fixed-complexity teaching demo:** compare high-degree (`degree=15`) OLS versus `degree=15` Ridge, selecting `lambda` by validation MSE. Shows how Ridge changes high-degree behavior.

B. **Fair model-selection baseline:** compare validation-selected OLS degree against validation-selected Ridge (degree and lambda). Note that Ridge need not win every run.

C. **Numerical methods demo:** a small hand-computable case, normal equation and QR/SVD; differentiate unstable computation from overfitting.

D. **Generalization check:** once A and B are fixed using validation only, report sealed test scores once; preserve a snapshot of selections. Do not revise choices after observing test outcomes. If exploratory UI shows test for teaching, mark it "illustrative only / no longer unbiased model selection" and keep official evaluation unchanged.

### Required analyses and visuals
- [ ] Sample scatter (train/validation), ground-truth sine curve and selected fitted polynomial(s).
- [ ] Train/validation MSE against degree, with understandable annotations.
- [ ] OLS versus Ridge curve overlay and coefficient magnitude comparison.
- [ ] Validation MSE versus `log10(lambda)` (show 0 separately), plus too-strong-regularization case.
- [ ] Condition-number diagnostic with a clear numerical-precision warning.
- [ ] Final held-out results table (MSE + RMSE; no fabricated performance claims).
- [ ] **Interactive controls:** polynomial degree and lambda; reset-to-default; consistent state when switching pages.
- [ ] **Battle B's main demonstration:** a genuinely two-parameter 3D loss surface of a simple linear model, with OLS and Ridge minima. A high-degree polynomial has more than two parameters: label any 3D depiction of it as a **two-parameter slice** rather than "the full loss surface." No need for a gradient-descent animation unless time remains.
- [ ] Every figure has title, axes, units/metric definitions, legends, dataset split labels, and an actual/illustrative marker.

**Predefined guardrail:** If the default synthetic dataset does not show visually strong overfitting, document the observed outcome first. Any *additional* exploratory demonstration scenario must be versioned and described as exploratory, never silently swapped in after viewing sealed test results.

## 4. Six-member ownership and handoffs

### Confirmed leadership and proposed person-to-role allocation

**Leadership positions are confirmed by the requester.** The proposed allocation of the other members to P1–P4 and the allocation of the two leaders to technical tracks are recommended working assignments, pending a quick team acknowledgment. The six student names are taken from the group's supplied README; no skill or preference has been inferred from anyone's name.

| ID | Student | Role / personal deliverable | Module / artifacts owned | Integration dependency | Mandatory independent PR review |
|---|---|---|---|---|---|
| **P1** | **Nassasi Treesuptavee** | Least Squares theory specialist; explain projection, uniqueness and normal equations | `report/sections/01_least_squares.md`, projection diagram, explanations and content for Least Squares web page, own slide/script | Align terminology with P2; math display conventions from P6 | **P2 Thanatorn** |
| **P2** | **Thanatorn Charoensuk** | Numerical methods developer; hand-worked example, QR and numerical stability | OLS/QR part of `solvers.py`, `tests/test_solvers.py` for OLS, `report/sections/02_hand_examples_qr.md`, QR explanation, own slide/script | Agree design-matrix and scaling contracts with P3; coordinate with P5 on shared solver file | **P5 Sirapobe** |
| **P3** | **Parama Polsen** | Data and polynomial modeling developer; **early shared-API handoff** | `datasets.py`, `features.py`, `config.py`, corresponding tests, `report/sections/03_data_models.md`, curve overlay data/figures, own slide/script | Provide deterministic split, `make_A` and scaling API to P2/P4/P5 immediately after G0 | **P4 Sitthawee** |
| **P4** | **Sitthawee Whangsawangwong** | Overfitting experiment and diagnostics developer; degree sweep and evaluation metrics | `metrics.py`, `diagnostics.py`, degree-sweep portion of `experiments.py`, tests, `report/sections/04_overfitting.md`, error/conditioning charts, own slide/script | Depends on P3's frozen train/validation splits and P2's solver; coordinate experiment interfaces with P5 | **P3 Parama** |
| **P5** | **Sirapobe Parinyarat (Sirapob)** | **Technical Advisor / Research Guide + hands-on Ridge developer**; review mathematical decisions and experimental design | Ridge portion of `solvers.py`, lambda sweep/validation selection of `experiments.py`, Ridge tests, `report/sections/05_ridge.md`, Ridge page/chart data, own slide/script | P3 fixed features and P4 metrics; coordinate solver edits with P2; review science gates without overriding independent review | **P2 Thanatorn** |
| **P6** | **Nipitpon Choojit** | **Project Manager + hands-on web/integration developer**; schedule, milestones, issue/PR coordination, UI experience | `api/main.py`, `frontend/`, browser smoke/E2E tests, `report/sections/06_ml_conclusions.md`, final report/slide/video assembly, own slide/script | Integrate published APIs/assets from P1–P5; delegate each page's science content to its subject owner | **P1 Nassasi** for written/UI changes, **P4 Sitthawee** for result displays |

### Project management and technical advisory authority

- **Nipitpon (PM):** owns timeline, issue board, scope control, handoffs, CI/release readiness, demo coherence and final submission checklist. **Must write/review substantive web code, tests and ML/report content**; management work does not replace an equal academic or speaking contribution. PM may coordinate merges after approvals but cannot unilaterally waive a failing quality/science gate.
- **Sirapobe (Technical Advisor):** sets up short design reviews, provides scientific direction on Ridge, conditioning, validation/test leakage and numerical correctness, and assists P2–P5 when blocked. **Must implement Ridge features, tests, analysis and the Ridge presentation segment**; advisory work is not a passive title or exemption from delivery. Major methodological changes require documented team review, not unilateral override.
- **Everyone:** owns one report section, one ~100-second spoken segment, their section's UI explanation/figure input, code and tests where applicable, and **at least one review of another person's PR**. P6 is the integrator, not the only author of the website or the only video editor.
- **Integration ownership by page:** P1 supplies Least Squares content, P3 supplies dataset/model controls and curve data, P4 supplies overfitting visuals, P5 supplies Ridge visuals, P6 writes shared frontend components/navigation and integrates overview/results; P2 verifies calculations and supplies a QR comparison illustration. Plotly.js rendering stays in `frontend/`; scientific calculations stay in `src/ridge_lab`.
- **Review rule:** no self-approval or direct-to-main changes. Assigned reviewers above are defaults; when unavailable, choose another member not involved in that PR. Changes to shared APIs must tag affected owners; P5 advises but still needs an independent reviewer for P5's own PRs.
- **Role changes:** the team may swap P1–P4 or a technical track after kickoff, but record the final name/ID mapping in `task.md` and GitHub issues before Phase 1. PM and Technical Advisor titles stay as confirmed unless explicitly changed by the requester.

### First handoffs (avoid blocked parallel work)

1. **P6 Nipitpon + P3 Parama:** initialize the new local repo and CI/app shell, document package/data interfaces by Nov 11.
2. **P5 Sirapobe + P2 Thanatorn + P4 Sitthawee:** sign off on lambda/scaler/loss and solver/test-data rules before independent numerical implementations.
3. **P3 Parama:** first stable dataset/feature API handoff to P2/P4/P5; avoid creating four incompatible datasets.
4. **P1 Nassasi:** deliver validated theory copy and diagram directly to the web/report section without waiting on experimental results.
5. **P6 Nipitpon:** track integration and capture missing exports in issues; **all owners** verify their final UI charts/text rather than outsourcing QA to P6.

### Speaker schedule — exactly 10 minutes target

| Window | Speaker | Focus |
|---|---|---|
| 00:00–01:40 | P1 — Nassasi | Open with “Can we predict how many bikes will be rented in a given hour?”; Least Squares intuition |
| 01:40–03:20 | P2 — Thanatorn | Normal equations, one small numeric/QR illustration |
| 03:20–05:00 | P3 — Parama | Bike hour × working-day design in plain English; separate synthetic degree study |
| 05:00–06:40 | P4 — Sitthawee | Bike chronological validation and synthetic overfitting curves |
| 06:40–08:20 | P5 — Sirapobe | Ridge/Lasso comparison and Battle B 3D loss surface |
| 08:20–10:00 | P6 — Nipitpon | RMSE ranking, supporting metrics, held-out evaluation after freeze, limitations |

Each member records their own ~100-second segment or participates in one synchronized recording; reserve transitions *inside* their allocated segment rather than adding an extra narrator.

## 5. Implementation phases and checkpoints

### Battle A — Audit and select one real-world dataset

**Decision owner:** Nipitpon, with the team if needed. **Status:** Capital Bike Sharing and the chronological evaluation are confirmed; official holdout scoring awaits protocol freeze. The real-world OLS/Ridge/Lasso comparison is a separate Battle A evaluation. The controlled synthetic degree/ten-seed experiment remains separate.

- [x] Compare three public regression datasets using original publisher records and locally inspected raw files. Record source, CC BY 4.0 terms, target and units, schema, row counts, missingness, duplicates, repeated settings or dates, leakage risks, and feasible polynomial complexity in [the audit](docs/datasets_review.md).
- [x] Recommend one dataset and a fallback using data quality, split validity, licensing, and workload. Keep candidate selection independent of model scores.
- [x] **Project owner:** select UCI Capital Bike Sharing `hour.csv` for its clear hourly rental question. Accept the more demanding chronological split and narrower fixed interaction model rather than the audit's simpler Airfoil recommendation. This choice was made on presentation and question fit, not candidate model scores; preserve the audit's original recommendation as history.
- [x] Use `cnt` (rentals/hour) as the target and a single-intercept design with categorical `hr`, binary `workingday`, both main effects, and their interaction. Exclude weather, `casual`, `registered`, IDs, and all other variables from predictors. The target components reconstruct `cnt` exactly.
- [x] Reserve the last 146 of 731 distinct dates as one fixed, sealed future holdout. Use five expanding-window development folds on the preceding 585 dates; keep each date intact. Record exact dates and row counts in [the split contract](docs/bike_split_contract.md).
- [x] Implement the loader, source checksum, stable date-hour row IDs, shared features/folds, and tests. Development-only OLS/Ridge/Lasso smoke metrics are documented separately from official holdout results.
- [ ] **P3 → Sitthawee and Sirapobe:** complete the [protocol-freeze review checklist](docs/bike_split_contract.md#protocol-freeze-review--pending) and record dated technical sign-offs. Arrange the separate independent PR review; Sirapobe cannot self-approve Ridge code. Any preprocessing learned later must be fit within each training window. Agree on the final refit and reporting protocol before holdout scoring.
- [ ] Freeze feature, preprocessing, model selection, and evaluation protocol before the single official holdout scoring stage. Do not use holdout outcomes to choose the dataset, split, degree, lambda, or Lasso penalty.

**Battle A ranking:** primary mean validation RMSE in rentals/hour; show MAE, R², and RMSE variation across the five future windows beside it. User-adjusted tier weights may produce an explicitly exploratory ranking; do not substitute that ranking for the fixed primary result. **Battle B solver ranking remains separate.** Polynomial degree selection and the ten-seed sensitivity study belong to the synthetic experiment; bike validation is chronological and unseeded.

**Freeze status:** pending Sitthawee and Sirapobe sign-off, independent code review, and G0 role/interface acknowledgments. The research decisions and bike split contract resolve the ten-seed scope; the historical dataset audit's earlier airfoil-repeat proposal is superseded. No official bike holdout scoring is authorized by this review packet.

### Phase 0 — Initialization, contracts, and feasibility (Nov 9–11)
- [x] Local scaffold prepared for review: `pyproject.toml`, `.gitignore`, CI, package skeleton, FastAPI service and browser UI; original group README preserved. Named ownership/acknowledgment is still pending, and no remote GitHub repo was created.
- [ ] **All:** Acknowledge proposed six-person technical assignments (P1–P6) and their reviewers; create six corresponding issues/branches and a PR template after team agreement.
- [ ] **P5 Sirapobe + P2 Thanatorn + P3 Parama + P4 Sitthawee:** Agree signatures, data shapes, intercept convention, feature scaling, loss normalization and split policy in `docs/decisions.md` or README.
- [x] Local dataset metadata/spec, tiny sample fixture and provisional experiment configuration prepared for P3 review; owner approval is pending.
- [x] One-page English storyboard and initial shell prepared for P1/P6 review; owner approval is pending.
- [x] **Agent:** Inspect available MCP servers and installed skills; use only relevant connected tools.

**Gate G0:** fresh install works, `python -m pytest` works, `python -m uvicorn api.main:app` serves the site and API; paths/package imports resolve; the six **named** contributors acknowledge their proposed work allocations and interfaces. Local technical checks are now passing; role acknowledgments and method-owner sign-off remain open, so G0 is not formally closed.

### Phase 1 — Math/data/model baseline (Nov 12–16)
- [ ] **P1:** Write least squares proof/diagram and textbook mapping; validate notation.
- [ ] **P2:** Implement and test normal equation demonstration, `lstsq` / QR reference (including near-singular case).
- [ ] **P3:** Implement generator/splits, scaling, `make_A`, baseline degree sweep; serialize reproducibility config.
- [ ] **P4:** Implement MSE and diagnostics, independent split evaluation and validations.
- [ ] **P5:** Implement ridge solver and compare against a trusted independent implementation on well-conditioned cases.
- [ ] **P6:** Complete shared page shell/controls and app smoke test; hook to mock typed results only until core is ready.

**Gate G1:** `pytest` passes, fixed-seed outputs reproducible, no test leakage, OLS/Ridge math unit tests pass, no model logic hidden in web.

### Phase 2 — Full experiments and interactive visualizations (Nov 17–21)
- [ ] Degree vs train/validation MSE, curve overlays, coefficient norms and condition diagnostics.
- [ ] Ridge `lambda` sweep, fixed-degree comparison, validation-selected fair comparison.
- [ ] Functional controls + 2D Plotly charts, explanation text for each section.
- [ ] Preliminary reproducible CSV/JSON results with unambiguous provenance.
- [ ] Add 3D **two-parameter** landscape if G1 and the core graphs are already complete.

**Gate G2:** app works end-to-end using the core calculation package; each figure matches corresponding exported results; no runtime exceptions for edge values (including `lambda=0` and degree 15).

### Phase 3 — Report, validation, recording (Nov 22–26)
- [ ] Freeze model selection on validation; perform the one-time official held-out test.
- [ ] Each member submits their section, figures, concise script and citation notes.
- [ ] Merge report/slides; show actual results, limitations and potential bias–variance tradeoff.
- [ ] Perform math peer review and usability test with a non-contributor.
- [ ] Record a draft 10-minute talk; check approximately equal speaking duration.

**Gate G3:** complete draft of all assignment artifacts; numerical results independently reproducible; all citations accurate; six members contribute.

### Phase 4 — Freeze and submit (Nov 27–Dec 1)
- [ ] Nov 27–29: Fix critical bugs, run full tests on clean install, rehearse, record final video.
- [ ] Nov 30: Export final report PDF and slide PDF/presentation, verify video length/audio/all six voices, freeze Git commit and submit-ready folder.
- [ ] **Dec 1:** Submit the report and the 10-minute video on the official course platform; verify upload/confirmation.

**Gate G4 / Definition of Done:** report + video both exist, run instructions verified, tests pass, exact captured commit/config recorded, six speaking segments approximately equal, and no unsubstantiated claims (especially "Ridge always improves").

## 6. API contracts for phase 0 (draft; finalize before parallel implementation)

```python
# src/ridge_lab/datasets.py
make_dataset(seed: int, noise_std: float, n_train: int, n_val: int, n_test: int)
# -> typed immutable/frozen split bundle with x, y, sample IDs and provenance

# src/ridge_lab/features.py
make_A(x: np.ndarray, degree: int) -> np.ndarray
# shape (n_samples, degree+1), intercept in column zero
# separate train-only transformer fit/transform for polynomial feature scaling

# src/ridge_lab/solvers.py
fit_ols(A_train: np.ndarray, y_train: np.ndarray, method: str = "lstsq")
ridge(A_train: np.ndarray, y_train: np.ndarray, lam: float, penalize_intercept: bool = False)
# -> coefficients / structured diagnostics, with 0 <= lam and finite input validation

# src/ridge_lab/metrics.py
mse(y_true: np.ndarray, y_pred: np.ndarray) -> float
rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float

# src/ridge_lab/experiments.py
run_degree_sweep(dataset, degrees, config)
run_lambda_sweep(dataset, degree, lambdas, config)
select_hyperparameters(validation_metrics)

# src/ridge_lab/evaluation.py
finalize_and_evaluate(dataset, frozen_selection, config)
# official test results only after recorded freeze
```

Design nuances: immutable results are preferred; function arguments explicit; input array shapes checked; all functions deterministic; no writing files in core functions. Avoid making the names above authoritative **until G0 API agreement**.

## 7. Testing and scientific integrity checklist

- [ ] 2D design matrix shape, intercept values, degree bounds and non-finite input checks.
- [ ] Fixed-seed deterministic dataset and no overlap between split IDs.
- [ ] Preprocessing learned from training only; trained transform reused on validation and test.
- [ ] `lambda=0` Ridge agrees with stable OLS within numerical tolerance (same feature matrix).
- [ ] Ridge penalty excludes intercept in declared convention; coefficients and outputs finite for a sensible test matrix.
- [ ] Small hand-solvable LS case agrees with calculation; QR/SVD agree with stable reference.
- [ ] Ill-conditioned normal-equation examples produce labeled warnings/fallbacks, not silently bogus outputs.
- [ ] MSE nonnegative and correct on known vectors; selection code never accesses test targets.
- [ ] Final metrics and chart CSVs reproduce from a clean run with same config.
- [ ] Browser site loads; degree/lambda controls update the API-backed plots and metrics coherently; mobile navigation and errors are accessible.
- [ ] README quickstart, sources, no paid dependencies, and reproducibility commands work.

## 8. Codex + MCP + skills working agreement

- Before substantial implementation, inspect repository, available **Codex skills** and connected MCP servers. Do **not assume** every tool remembered from other projects is available in this session.
- **Context7 MCP** (if available): verify current package docs for FastAPI static serving and validation, Plotly.js, and NumPy/scikit-learn APIs before implementing unfamiliar calls.
- **GitHub MCP** (if available and authorized): manage issues, branch/PR reviews, and CI status. Never push or merge to a shared branch without the team's expected workflow.
- **Playwright MCP** (if available): end-to-end checks for navigation, sliders, plots, export and responsive layout. Record failures and screenshots only when useful.
- **Other skills/MCPs:** invoke narrowly for their actual purpose (e.g., architecture skill for dependency review); do not add unnecessary dependencies or run unrelated agent tooling.
- Coding agent should complete **Phase 0 only** on first initialization pass, with small safe scaffolding, initial tests, and clear TODOs. Do not fabricate data/results or make up screenshots/recorded submissions.

## 9. Confirmed configuration and remaining decisions

**Confirmed by the requester:**
- **Deliverable language:** English for the website, report, presentation/slides, 10-minute spoken video and project docs/code.
- **Repository:** continue in the existing The Least Squares Showdown repository.
- **Battle A:** UCI Capital Bike Sharing `hour.csv`, `cnt` target, hour × working-day main effects and interaction, chronological validation, RMSE primary with MAE/R²/window variation beside it.
- **Presentation:** open with the hourly bike-rental prediction question; keep Battle B's 3D loss surface as its main demonstration.
- **Leadership:** Nipitpon Choojit is Project Manager; Sirapobe Parinyarat (Sirapob) is Technical Advisor/Research Guide. **Both remain active implementers and speakers.**

**Still pending; use safe non-blocking defaults:**
- **Team acknowledgment of proposed P1–P6 work allocation:** roster-based work proposal is above. Do not misrepresent it as six individually confirmed assignments.
- **GitHub owner/organization, repository visibility and collaborator handles:** not provided. Initialize locally; ask for details before creating a remote or inviting people.
- **Hosting:** local `python -m uvicorn api.main:app` by default; public sharing optional after testing.
- **UI style:** clean light theme, math-forward, graph-first, responsive laptop layout.
- **Recording:** six ~100-second English-speaking segments, with transitions within assigned time.
- **Administrative actions:** remote repo creation, public release and final course submission require explicit authorization; Codex can safely scaffold local files immediately.

## 10. First Codex instruction (copy/paste)

> Historical Phase 0 bootstrap instruction, superseded by the requester-approved architecture migration in section 1.1. The active local interface is vanilla HTML/CSS/JavaScript + FastAPI; it does not authorize later-phase experiments or remote changes.

---

**Academic source:** Textbook Sections 6.5 and 6.6 as provided by the course, plus the team's original README and bilingual group plans. Cite relevant textbook sections and any externally consulted references in the final report.

A1 approved by Nipitpon: approval supplied in the current conversation for the preceding working-tree final-refit fix. No unique committed revision exists for that fix; historical base is bda781cc1dae8bd0493de3f3979e351a05cd6bd7.

## Consolidated owner handoff (2026-11-19)

A1 approved by Nipitpon. Sole human review is now Nipitpon; prior project-managed separate sign-off requests are superseded. Development table/API and masked reserved targets are implemented. New reproducible studies, Run All, and full two-parameter surface are available. See BATTLE_A_PROTOCOL_REVIEW.md for measured checks and report/draft.md for actual results. Synthetic convergence limitations and provisional configuration require final review. G0 acknowledgments remain open; official Bike Sharing holdout stays sealed. No PDF/video/submission is claimed.
