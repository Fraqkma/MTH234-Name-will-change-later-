> Current owner decision (2026-11-19): Nipitpon is the sole human reviewer. This supersedes project-managed requests below for separate Sitthawee, Sirapobe, or independent reviewer sign-offs. G0 acknowledgments remain unrecorded unless actually supplied. Protocol freeze and official holdout evaluation remain pending Nipitpon's consolidated final review.

# Battle A — Bike Sharing Protocol Review

**Project** The Least Squares Showdown  
**Technical reviewers** Sitthawee and Sirapobe  
**Review status** Pending human sign-off  
**Review scope** Development evidence, implementation checks, and readiness for protocol freeze

## 1. Task for Codex

Execute the technical checks in this document for both reviewers in one pass. Read the current checkout, trace the Bike Sharing experiment end to end, run the relevant development-only tests, and append one evidence report to this document. Give both reviewers enough information to verify or challenge the conclusions from the same report.

Use the existing protocol as the baseline. Preserve the team's edits and keep code and protocol files read-only for this review. If you find a defect, cite it, explain its impact, and propose the smallest correction. Continue all review work that does not depend on the correction or on a human answer.

Keep the official Bike Sharing holdout sealed. Development runs may use the intended development entry points after you verify their call paths. Inspect final evaluation code statically or exercise its boundaries with a synthetic fixture. Do not run the official holdout evaluator, generate holdout predictions, inspect holdout outcomes, or use holdout information for method selection.

## 2. Starting points and reported baseline

Start with `AGENTS.md` if present, `task.md`, `docs/bike_split_contract.md`, `docs/decisions.md`, and `src/ridge_lab/bike.py`. Find the relevant tests, experiment runner, and site data consumer from the repository. Record the reviewed commit and whether relevant working-tree edits were present. If you lack the only copy of required files or data, name the exact paths needed and proceed with checks you can perform.

Treat these values as the team's reported baseline. Confirm them against the current contract and checkout before marking them verified.

| Item | Reported baseline |
| --- | --- |
| Dataset and target | UCI Capital Bike Sharing `hour.csv`, with `cnt` as rentals per hour |
| Shared design matrix | 48 columns total, including one intercept, hour and working-day main effects, and their interaction |
| Development validation | Five chronological expanding windows and 13,915 rows in the full development set |
| Reserved holdout | 2012-08-08 through 2012-12-31 inclusive, with 146 dates and 3,464 rows |
| Ten-seed study | Assigned to the synthetic experiment. Bike Sharing uses unseeded chronological windows |
| Development selection | Ridge selected λ = 0 and Lasso selected α = 0.001 |
| Reported mean validation RMSE | OLS 117.460, Ridge 117.460, and Lasso 117.474 rentals per hour |

## 3. Data, feature basis, and chronological split

### Trace the data into the site

Trace the path from the `hour.csv` loader through field validation, split construction, feature construction, model selection, metrics, the relevant API or artifact, and site rendering. Give file and line references for each transition. Identify how the site distinguishes the development benchmark from its adjustable exploratory tier ordering and from Battle B's solver ranking. Check that units, split labels, and rounding match the contract.

### Verify the shared 48-column basis

- Derive the column count from the actual reference coding. Identify the reference hour and working-day level, count main effects and interaction columns, and show where the single intercept enters. For reference coding with 24 hours and two working-day levels, the expected count is 1 + 23 + 1 + 23.
- Verify that OLS, Ridge, and Lasso use the same feature mapping and column order. Check for a second estimator intercept, duplicated columns, inconsistent category ordering across folds, and any handling of rank-deficient folds. Confirm dimension and rank on development splits if the safe entry point exposes the design matrix.
- Verify that only approved predictors reach the matrix. In particular, show where `casual` and `registered` are excluded because they sum to `cnt`. Check that weather, IDs, raw date keys, and target-derived or future information stay out of the fit. Dates can serve as split keys. If any preprocessing learns parameters, show that it fits on each training window.

### Verify the five windows

- Report each window's train and validation date bounds and row counts. Check that training ends before validation starts, train and validation dates do not overlap within each window, and training history expands as documented. Check the expected behavior for observations on cutoff dates and for multiple observations from the same date.
- Confirm that all development training, validation, selection, and metrics stay on the development side of the cutoff. Verify the final development row count without generating any holdout outputs. Use split metadata, contract values, or fixture-based checks for the holdout range and boundary.
- Confirm the ten-seed study's synthetic-only scope against the current documents. Raise a decision only if newer source material genuinely conflicts with that documented scope.

## 4. Objectives, selection, and final refit

### Reconcile the penalty objectives

Write the OLS, Ridge, and Lasso objectives as implemented. Include any factor of `n` or `1/2` in the residual term, the precise penalty term, and the mapping from the reported λ or α to the estimator parameter. Check the mapping across training windows and at refit if it depends on sample count.

Show how each model leaves the intercept unpenalized while penalizing the intended coefficients. If centering or scaling is involved, trace when and where it is fitted and how predictions recover the intercept. Review any explicit zero-penalty branch and verify what Ridge λ = 0 means for the reported OLS comparison under this implementation.

### Reproduce development selection and results

- Verify the documented Ridge and Lasso grids, the common windows for all candidates, and which metric picks the candidate. Show whether reported mean RMSE is the mean of window RMSEs or the RMSE of pooled validation residuals, and verify any weights and tie tolerance or ordering directly from the implementation.
- Check that ties are deterministic and follow the documented rule. Verify reported selected values and the RMSE, MAE, R², and variation statistics shown on the site. If the rounded RMSE values disagree with the reported baseline, trace the discrepancy and record where it originates.
- Confirm that adjustable presentation weights and Battle B's solver ranking do not feed Battle A hyperparameter selection or refit.

### Inspect the final-refit path

Verify that the final refit receives the fixed development-selected parameters and uses all 13,915 development rows. Check that it rebuilds any fitted preprocessing from the full development set and neither reselects hyperparameters nor includes validation rows twice. Locate the official scoring entry point and inspect its call boundaries without invoking it. If a final refit entry point is absent or incomplete, distinguish a planned step from implemented behavior and assess whether the missing piece blocks freeze under the current contract.

## 5. Verification and report

Inspect test fixtures and setup before running the relevant tests. Prefer the existing Bike Sharing, model-selection, and site/API tests that exercise only development data or synthetic fixtures. Run the development command if its entry point stays within scope. Record each exact command, pass/fail and skip counts, relevant warnings, and any check you could only perform statically. Run `git diff --check` and confirm the review preserved existing team edits. A previous test run can be noted as historical context.

Append the findings below the report heading in this file. List freeze blockers first. For each finding, give the observed behavior, the expected contract behavior, the consequence, and an actionable correction. Provide actual file and line references from the checkout. If there are no technical blockers in the evidence you inspected, make that explicit and list the limits of the verification.

| ID | Freeze impact | Finding and consequence | Evidence | Proposed correction or decision |
| --- | --- | --- | --- | --- |
| — | Blocker / Non-blocking | — | `path:line` | — |

The handoff should also include the reviewed commit and working-tree scope, the data-to-site trace, the five-window summary, the penalty and selection conclusions, test results, and whether official holdout scoring remained untouched.

## 6. Human decisions and sign-offs

Ask Nipitpon before changing an owner-level choice such as the feature set, split boundaries, penalty definition, primary metric, or refit policy. Ask only when the checkout and existing decisions leave a genuine choice or conflict. Present the conflicting evidence, feasible options, your recommendation, and the effect on freeze. Record the question and keep moving on checks that are independent of the answer. A clear implementation mismatch should be reported as a finding with a proposed correction.

Bring the same report to Sitthawee and Sirapobe for their own judgments. Record approvals only when actually supplied and retain the reviewer's name, reviewed revision, decision, and any conditions. Use the repository's existing policy for the separate independent code review and G0 role/interface acknowledgments. The technical evidence here supports review and does not silently satisfy a separate approval gate. Leave freeze pending whenever a required approval or technical blocker remains open. Official holdout evaluation is a later task.

## 7. Review report

| Item | Current status |
| --- | --- |
| Combined technical evidence | A1 fixed in the working tree; awaiting Sitthawee and Sirapobe verification in the dated follow-up below |
| Sitthawee technical sign-off | Pending according to the last team update |
| Sirapobe technical sign-off | Pending according to the last team update |
| Separate independent code review | Pending according to the last team update |
| G0 role and interface acknowledgments | Open according to the last team update |
| Protocol freeze | Pending |
| Official holdout evaluation | Reserved for later evaluation |

### Combined technical evidence — 2026-11-19

**Review status:** evidence collected for both reviewers; this is not Sitthawee's or Sirapobe's approval. **Reviewed commit:** `bda781cc1dae8bd0493de3f3979e351a05cd6bd7`. The working tree was already dirty: relevant edits in `src/ridge_lab/features.py`, `src/ridge_lab/solvers.py`, `frontend/index.html`, `frontend/js/app.js`, `task.md`, and the research documents; `data/bike/hour.csv`, `src/ridge_lab/bike.py`, `src/ridge_lab/bike_experiment.py`, `tests/test_bike.py`, and this review file were untracked. This report reviews that checkout, not the committed tree. No implementation or protocol file was changed during this review.

#### Findings, ordered by freeze impact

| ID | Freeze impact | Observed behavior, expected contract, and consequence | File:line evidence | Smallest correction or decision |
| --- | --- | --- | --- | --- |
| A1 | **Blocker** | The implemented bike path ends with fold evaluation and per-method setting selection. There is no final-refit function or official scoring entry point to verify that the selected settings fit **all 13,915 development rows exactly once** before the later holdout evaluation. The contract explicitly requires this refit after sign-off. Thus the final training boundary cannot yet be certified. | `src/ridge_lab/bike_experiment.py:37-99`; `src/ridge_lab/bike.py:131-136`; `docs/bike_split_contract.md:65`; `api/main.py:20-28` | After reviewer approval, add a dedicated refit that takes the fixed selected settings and `split.development`, builds the existing 48-column basis once, fits each method once, records the 13,915-row membership and settings, and has a development-only or synthetic-fixture test. Do not invoke official scoring in this correction. No owner-level protocol choice is needed. |
| A2 | **Non-blocking for protocol freeze; publication risk** | The bike table is a manually copied development snapshot, not a runtime result from the CSV or an API/artifact. Recalculation in this review matches every displayed three-decimal summary, so there is no current numeric discrepancy. A later code or grid change could leave the site stale without a failed test. | `frontend/index.html:208-231`; `api/main.py:20-28`; `frontend/js/bike-rank.js:22-28`; `src/ridge_lab/bike_experiment.py:77-99` | Preserve a versioned development result snapshot or add a check that regenerates and compares the table before publication. Keep the exploratory tier order separate from primary RMSE. |
| A3 | **Non-blocking under the current development path** | The loader materializes `cnt` for every row, including reserved dates, in `BikeData.counts`; sealing is enforced by the evaluator's index boundary rather than by a separate target-access interface. The evaluator uses only fold train/validation indices, and the holdout-NaN test passes, so no holdout score or target-driven choice occurred here. The open interface remains an accidental-use risk for later code. | `src/ridge_lab/bike.py:70-95`; `src/ridge_lab/bike_experiment.py:47-70`; `tests/test_bike.py:90-100` | Keep official evaluation behind a separate later entry point and retain the holdout-exclusion test. Consider restricting development-facing target access if the final evaluator is added. |
| A4 | **Non-blocking display clarity** | The selector's exact-tie rule is implemented within each method, but there is no separate primary cross-method ranking function or explicit primary-table tie marker. The site shows equal OLS/Ridge metrics and says their predictions match; its exploratory ranking groups equal scores. This is consistent numerically but less explicit than the contract's instruction to report equal cross-method RMSE as a tie. | `src/ridge_lab/bike_experiment.py:89-99`; `frontend/index.html:214-231`; `frontend/js/bike-rank.js:40-47`; `docs/bike_split_contract.md:64` | Label OLS and Ridge tied in the primary development table or caption before publication; do not use exploratory weights to break that tie. |

**Governance gate:** `docs/bike_split_contract.md:58-74` still shows Sitthawee and Sirapobe pending, as well as the separate independent code review and G0 role/interface acknowledgments. This evidence report does not satisfy those human gates. Protocol freeze remains pending even after A1 is corrected.

#### Data-to-site trace and verified boundaries

1. `data/bike/hour.csv:1` supplies the raw schema. `src/ridge_lab/bike.py:14-19,58-98` verifies its SHA-256 (`e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`), column order, date/hour/status/count types and ranges, and unique date-hour IDs. The returned object contains only `dates`, `hours`, `working_days`, `counts`, IDs, and hash. `cnt` is the target; `casual` and `registered`, weather, `instant`, and other raw fields never enter the design. The prior audit reports `casual + registered = cnt`; this review did not reread reserved target outcomes to repeat that full-file claim.
2. `src/ridge_lab/bike.py:101-136` sorts distinct dates, reserves the last 146 dates, and divides the previous 585 dates into six consecutive blocks for five expanding folds. Date-index expansion at lines 112-125 keeps all hours of a cutoff date on the same side. The development run reported 13,915 development rows; split metadata reported 3,464 reserved rows from 2012-08-08 through 2012-12-31. No holdout predictions or metrics were generated.
3. `src/ridge_lab/features.py:25-40` constructs `1 + 23 + 1 + 23 = 48` columns: one intercept; hour 1–23 indicators with hour 0 as reference; one `workingday` main effect with status 0 as reference; and 23 products. Category order is fixed by `np.arange(1, 24)`, with no fold-fitted encoder or second estimator intercept. The review command found numerical rank 48 for **each** training window. All three methods call this same function for train and validation at `src/ridge_lab/bike_experiment.py:56-69`.
4. `src/ridge_lab/bike_experiment.py:70-85` computes RMSE, MAE, and R² in each validation window, then uses the **unweighted mean of five window RMSEs** (not pooled-residual RMSE) as primary value. `np.std` defaults to population SD. `src/ridge_lab/bike_experiment.py:89-99` selects within each method by unrounded mean RMSE, then lower penalty on an exact tie; no tolerance is applied. Ridge `λ=0` calls OLS exactly. The primary table shows equal OLS/Ridge values, while the exploratory ordering explicitly groups their equal scores; no separate primary tie marker exists (A4). Exploratory weights read the static HTML rows in `frontend/js/bike-rank.js:22-47`; they do not enter the Python selector. Battle B has no path into these calculations.
5. `frontend/index.html:208-231` labels the table **development-only**, gives error units as rentals/hour, shows RMSE 117.460/117.460/117.474, MAE 83.835/83.835/83.838, R² 0.471 for each, and window RMSE SD 36.719/36.719/36.728, all rounded to three decimals. It separately labels the tier-weight order exploratory. `api/main.py:20-28` exposes only the synthetic demo; the bike values were copied into the site rather than fetched or generated there.

| Fold | Train dates and rows | Validation dates and rows | Training design rank |
| --- | --- | --- | ---: |
| 1 | 2011-01-01–2011-04-08; 2,259 | 2011-04-09–2011-07-15; 2,351 | 48 |
| 2 | 2011-01-01–2011-07-15; 4,610 | 2011-07-16–2011-10-21; 2,335 | 48 |
| 3 | 2011-01-01–2011-10-21; 6,945 | 2011-10-22–2012-01-26; 2,321 | 48 |
| 4 | 2011-01-01–2012-01-26; 9,266 | 2012-01-27–2012-05-02; 2,321 | 48 |
| 5 | 2011-01-01–2012-05-02; 11,587 | 2012-05-03–2012-08-07; 2,328 | 48 |

**Objectives and scaling:** OLS minimizes `SSE` via `numpy.linalg.lstsq` (`src/ridge_lab/solvers.py:23-26`). Ridge minimizes `SSE + λ Σ(j>0) β_j²`: the augmented rows use `sqrt(λ)` and set the intercept penalty to zero (`src/ridge_lab/solvers.py:29-43`). Ridge's reported λ is passed unchanged in every fold; on the `SSE/(2n)` scale its quadratic coefficient is `λ/(2n)`, which varies with training-window size. Lasso minimizes `SSE/(2n) + α Σ(j>0)|β_j|`: it centers the training features/target, divides its Gram and covariance by that fold's `n`, soft-thresholds at the reported α, and reconstructs the unpenalized intercept (`src/ridge_lab/solvers.py:46-87`). α is passed unchanged across folds. No additional learned preprocessing is used. The documented grids at `src/ridge_lab/bike_experiment.py:41-42` are Ridge `{0,1,10,100}` and Lasso `{0.001,0.01,0.1,1}`. The development run selected Ridge `λ=0` and Lasso `α=0.001`; Ridge `λ=0` equals this implementation's OLS on the same matrix. At final refit, Ridge's `λ/(2n)` mapping would use `n=13,915` if the same sum-SSE objective is retained, but no refit path exists yet.

**Ten-seed scope:** `task.md:165,176-178`, `docs/decisions.md:10,13`, and `docs/bike_split_contract.md:20,67` assign degree selection and ten seeded runs to the synthetic experiment. Bike validation is fixed and chronological. The older airfoil-repeat idea in `docs/datasets_review.md` is historical; it does not create a current owner-level conflict. No Nipitpon decision was required for this review.

#### Commands and outcomes

Run in `C:\Users\Non\Documents\GitHub\MTH234-Name-will-change-later-` on 2026-11-19:

```powershell
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src/ridge_lab/bike.py src/ridge_lab/bike_experiment.py src/ridge_lab/features.py src/ridge_lab/solvers.py src/ridge_lab/metrics.py tests/test_bike.py
git diff --check
```

- Pytest: **23 passed, 0 failed, 0 skipped**, in 0.95 s. One `StarletteDeprecationWarning` from `fastapi.testclient`/`httpx`; it did not affect the bike checks. `tests/test_bike.py:20-106` covers source hash/schema, distinct date-hour IDs, split membership and order, a 48-column full-rank first fold, Lasso intercept/soft-threshold behavior, and development evaluation with holdout targets replaced by NaN.
- Targeted Ruff: **all checks passed**. `git diff --check`: **passed** (no output).
- Development-only reproduction: the following exact command invoked only `evaluate_bike_development` and `select_bike_settings`, whose inspected call path reads fold train/validation targets only (`src/ridge_lab/bike_experiment.py:47-70`). It printed the five windows in the table above and the source hash above; all five training matrices had rank 48.

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np; from ridge_lab.bike import load_bike_hourly, make_bike_split; from ridge_lab.features import make_bike_design; from ridge_lab.bike_experiment import evaluate_bike_development, select_bike_settings; d=load_bike_hourly(); s=make_bike_split(d); print('source',d.sha256); print('development_rows',len(s.development),'holdout_rows',len(s.holdout),'holdout_dates',s.holdout_start,s.holdout_end); print('folds',[(str(f.train_start),str(f.train_end),len(f.train),str(f.validation_start),str(f.validation_end),len(f.validation),int(np.linalg.matrix_rank(make_bike_design(d.hours[list(f.train)],d.working_days[list(f.train)])))) for f in s.folds]); r=select_bike_settings(evaluate_bike_development(d,s)); print('selected',[(k,v.penalty,round(v.mean_rmse,3),round(v.mean_mae,3),round(v.mean_r2,3),round(v.rmse_std,3),[round(w.rmse,3) for w in v.windows]) for k,v in r.items()])"
```

Reproduced summaries: OLS `(RMSE 117.460, MAE 83.835, R² 0.471, RMSE SD 36.719)`; Ridge `λ=0` identical; Lasso `α=0.001` `(117.474, 83.838, 0.471, 36.728)`. Window RMSEs were OLS/Ridge `145.914, 88.495, 64.822, 122.708, 165.359` and Lasso `145.936, 88.512, 64.815, 122.724, 165.382`, matching `docs/bike_split_contract.md:44-54` and `frontend/index.html:215-217`. No official holdout scorer was invoked, no holdout prediction or metric was computed, and no holdout outcome was inspected in this review.

**Handoff:** A1 must be corrected and rechecked before technical freeze. Sitthawee and Sirapobe must give their own dated decisions on the reviewed revision; separate independent code review and G0 acknowledgments remain pending. Official holdout evaluation remains a later, distinct task.

### A1 final-refit correction — 2026-11-19

**Current A1 status: fixed in the working tree; awaiting Sitthawee and Sirapobe's verification.** This section supersedes only the historical A1 status above. It is an implementation and test record, not either reviewer's sign-off or a protocol freeze. No owner-level protocol choice was needed: the five-window selection yielded the already reviewed Ridge `λ=0` and Lasso `α=0.001`.

The new `refit_bike_development` accepts the existing selection output and does no selection or scoring (`src/ridge_lab/bike_experiment.py:129-134`). It checks the source hash, 2012-08-08 holdout cutoff, exactly 13,915 pre-cutoff development indices, complementary holdout indices, and all three reviewed settings *before* fitting (`src/ridge_lab/bike_experiment.py:135-153`). It builds the existing 48-column basis and target vector from only those development indices, then calls the existing OLS, Ridge, and Lasso helpers once each with penalties `0`, `0`, and `0.001` (`src/ridge_lab/bike_experiment.py:155-171`). The result records source hash, exact training row IDs, dates, settings, and read-only coefficient vectors; it has no holdout predictions or scores (`src/ridge_lab/bike_experiment.py:45-61,172-178`). The public package export is at `src/ridge_lab/__init__.py:4-8,18-20,37`.

The targeted regression test derives the reviewed settings from the actual five-window development selector, replaces every reserved target with `NaN`, and spies on each estimator. It verifies the *actual arrays passed to fit* are the same `(13915, 48)` development design and development targets, with calls `OLS()`, `Ridge(0.0)`, and `Lasso(0.001)`; it also checks recorded row IDs and 2011-01-01 through 2012-08-07 bounds (`tests/test_bike.py:114-149`). A second test rejects a holdout row added to development or a changed Ridge setting before any solver runs (`tests/test_bike.py:152-168`). These checks address A1's training boundary and setting risk. They do not count as independent reviewer approval.

Commands run in `C:\Users\Non\Documents\GitHub\MTH234-Name-will-change-later-`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_bike.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -c "from ridge_lab.bike import load_bike_hourly, make_bike_split; from ridge_lab.bike_experiment import evaluate_bike_development, select_bike_settings, refit_bike_development; d=load_bike_hourly(); s=make_bike_split(d); selected=select_bike_settings(evaluate_bike_development(d,s)); fitted=refit_bike_development(d,s,selected); print('source',fitted.source_sha256); print('training',len(fitted.training_row_ids),fitted.training_start,fitted.training_end); print('models',[(m.method,m.penalty,m.coefficients.shape) for m in fitted.models])"
git diff --check
```

Results: targeted pytest **12 passed** in 2.21 s; full pytest **25 passed** in 2.66 s with one `StarletteDeprecationWarning` from `fastapi.testclient`/`httpx`; full Ruff **all checks passed**; `git diff --check` **passed** with no output. The development-only smoke printed source SHA-256 `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`, `training 13915 2011-01-01 2012-08-07`, and `models [('OLS', 0.0, (48,)), ('Ridge', 0.0, (48,)), ('Lasso', 0.001, (48,))]`. It did not compute holdout predictions or scores.

**Remaining findings and gates:** A2 (manually copied site development table), A3 (loader materializes reserved targets), and A4 (primary tie display) retain their prior follow-up status. Sitthawee and Sirapobe's dated sign-offs, a separate independent code review, G0 role/interface acknowledgments, and protocol freeze remain pending. Official holdout evaluation stays reserved for the later frozen-protocol stage.


A1 approved by Nipitpon: approval supplied in the current conversation for the preceding working-tree final-refit fix. No unique committed revision exists for that fix; historical base is bda781cc1dae8bd0493de3f3979e351a05cd6bd7.

## Consolidated owner review package - 2026-11-19

**Review decision needed:** approve the provisional synthetic and bounded benchmark configuration in docs/decisions.md, and decide how to treat 365 Lasso iteration-limited candidates. Recommendation: retain this run as development evidence, improve/check convergence before making a final synthetic model ranking, and keep synthetic test evaluation reserved until that policy is frozen. Bike holdout remains sealed regardless. No explicit separate-reviewer course requirement was found in the available original group README; unavailable course materials were not certified.

**A1 approved by Nipitpon.** The owner approved the prior working-tree final-refit implementation. Base commit is `bda781cc1dae8bd0493de3f3979e351a05cd6bd7`; the approved change and this package are uncommitted working-tree revisions, so that commit alone does not identify them. The owner now supplies the sole human review. Historical requests for additional human sign-offs below are superseded. G0 member acknowledgments remain unrecorded.

### Implemented evidence

- A2: `api/main.py` exposes `/api/bike/development`; `bike_development_payload` performs the actual five-window selection and full development refit. `frontend/js/bike-rank.js` fills the empty HTML table with these actual values and uses full precision for exploratory ranking. Clean server startup requires no manually pasted results.
- A3: `load_bike_hourly` no longer converts reserved `cnt` values; it places NaN at those indices. Dates and predictor metadata remain for the existing split contract. Normal development selection, refit, and site entry point receive no reserved numeric targets. `test_normal_entry_point_masks_reserved_targets` exercises loading, selection, and refit and checks the cutoff and penalties.
- Baseline reproduced: OLS/Ridge mean window RMSE 117.460, MAE 83.835, R^2 0.471, RMSE SD 36.719. Lasso alpha 0.001: 117.474, 83.838, 0.471, 36.728. Ridge lambda is 0. Refit still records 13,915 rows ending 2012-08-07 with the shared 48-column basis.
- `src/ridge_lab/studies.py`: ten seeds, three noise levels, degree/penalty selection; solver benchmarking; full two-parameter loss surface. `results/` contains measured development artifacts. Lasso iteration failures are explicit. GD hits its cap in both condition-1e6 cases. Input memory is measured; peak workspace is not.
- `frontend/js/studies.js`: sequential Run All with visible progress and per-request failures; distinct solver output and exploratory runtime/accuracy weights; penalty-adjustable loss surface. Browser inspection confirmed three completed studies, no request failures, three bike rows, Plotly surface, correct axis labels, and unchanged bike result rows after adjusting weights. The final added degree plot and solver weight UI were checked statically, not included in that browser run.
- `src/ridge_lab/official.py`: explicit post-review scorer, absent from site routes. Fixture mode uses fabricated two-row data only. No real holdout predictions or scores were produced.

### Exact verification commands and outcomes

```powershell
.\.venv\Scripts\python.exe -m ridge_lab.reproduce
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8011
.\.venv\Scripts\python.exe -m ridge_lab.official --approved-by Nipitpon --fixture --output results/scoring-fixture.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
```

Pytest: **29 passed**, one existing Starlette/httpx deprecation warning, 7.27 seconds. Ruff: **all checks passed**. Whitespace: **passed**. Fixture scoring: RMSE=1, MAE=1, R^2=0. First artifact timings: bike 2.440 s, solver study 0.572 s, synthetic study 124.178 s. Later reruns may differ; per-run elapsed times are stored in JSON. Browser: Playwright opened a clean local server at port 8011 and observed Run All finish all three requests; loss axes were Intercept, Slope, Penalized SSE. Bike exploratory weights preserve selected settings and raw metrics; this baseline can retain the same order because OLS/Ridge dominate all supporting metrics.

### Handoff and limits

Report: `report/draft.md`. Six equal speaking slots and live-demo fallback: `docs/storyboard.md`. Commands: `README.md`. The synthetic test remains reserved under the explicitly provisional configuration. The report is a draft, not a submission-ready PDF/video. Recorded voices, member acknowledgments, final export, and submission are not claimed. Final review should resolve convergence before freezing a synthetic conclusion. Peak memory, broader benchmark hardware coverage, and independent external numerical cross-checks remain limitations.

After final approval only:

```powershell
.\.venv\Scripts\python.exe -m ridge_lab.official --approved-by Nipitpon --output results/bike-official.json
```

Current evidence locations: `src/ridge_lab/bike.py:74`, `src/ridge_lab/bike_experiment.py:181`, `tests/test_bike.py:177`, `src/ridge_lab/studies.py:13`, `src/ridge_lab/studies.py:84`, `src/ridge_lab/studies.py:152`, `src/ridge_lab/official.py:26`.

Final reproduction rerun: bike 1.880267 s, solvers 0.450644 s, synthetic 120.219538 s; synthetic artifact contains 7,735 converged candidates and 60 fixed-degree-15 comparisons. Final pytest rerun: 29 passed, one warning, 6.56 s. Ruff and git diff --check passed again. `results/review-manifest.json` identifies the uncommitted implementation/test revision by SHA-256 per file. The supplied base commit does not include these working-tree changes. Keep this manifest with the review package.

### Synthetic Lasso convergence resolution ? 2026-11-19

The synthetic protocol uses one train/validation split for each of 10 seeds at each of three noise levels; there are **no CV fold IDs**. The denominator is **1,800 individual Lasso fits**, equal to 30 distinct seed/noise splits times 15 degrees times four alpha values. Every fit is a unique `(seed, noise, degree, alpha)` configuration, so the baseline's 365 failed fits are also 365 unique failed configurations. Full per-split, seed, degree, alpha, iteration, and KKT detail is in `results/lasso-convergence-audit.json`, `results/synthetic-before.json`, `results/synthetic-intermediate.json`, and `results/synthetic.json`.

Baseline failures by seed 42?51: `37, 38, 52, 40, 31, 25, 35, 36, 38, 33`; by noise 0.1/0.2/0.4: `124, 121, 120`; by alpha 0.001/0.01/0.1/1: `291, 74, 0, 0`; by degree 1?15: `0, 0, 10, 2, 14, 23, 24, 30, 28, 35, 39, 40, 40, 40, 40`. Baseline raised RuntimeError at its 10,000-sweep limit and supplied no iteration count beyond that cap or KKT diagnostic. No convergence warning was emitted. This is recorded as an exception, not silently dropped.

A pure increase to 50,000 sweeps failed on representative high-degree fits. An incremental gradient implementation plus active-set KKT check and sign-boundary step keeps the same Lasso objective and 1e-8 stopping tolerance (`src/ridge_lab/solvers.py`). With the 50,000 cap, the intermediate full sweep had **21/1,800** failures; after the sign-boundary correction, the final full sweep had **0/1,800** failures. Final Lasso iterations ranged from 1 to 30,500; maximum KKT residual was 9.61e-9. The stricter 1e-10 reference test compares validation predictions and MSE for degrees 4, 8, and 15 (`tests/test_studies.py`). All passed. Final full sweep runtime: **38.10034 s** versus baseline **120.21954 s** and intermediate **149.52469 s**, on this machine.

The 365 missing baseline candidates affected model selection: **21 of 30 Lasso-selected degrees changed** once all candidates converged. All selected Lasso penalties remain alpha=0.001. OLS and Ridge selections did not change. Mean selected validation MSE by noise 0.1/0.2/0.4 is now OLS `0.011672/0.042793/0.165028`, Ridge `0.010059/0.039585/0.157390`, and Lasso `0.014724/0.044113/0.165766`. The aggregate validation ordering remains Ridge, OLS, Lasso at each noise level; the per-split degree changes and scores are in the audit JSON. The prespecified unrounded validation-MSE selection and exact tie keys were preserved. The synthetic test and Bike Sharing holdout remain unopened.

Final checks after the convergence correction: `.\.venv\Scripts\python.exe -m pytest -q` returned **30 passed**, one Starlette/httpx deprecation warning; `.\.venv\Scripts\python.exe -m ruff check .` passed; `git diff --check` passed. `scripts/export_report.py` generated `report/print.html`; the local Edge headless export generated `report/draft.pdf` (PDF 1.4, 136,238 bytes, four page objects). The report draft and six-speaker storyboard carry the corrected validation results. The print-ready HTML remains editable. The real official holdout scorer was not invoked.

Final local browser check: start `.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8011`, open `http://127.0.0.1:8011/?audit=final4#studies`, and click Run All. Playwright observed three completed study sections and the site text `30 seeded noise/split runs; 0 Lasso convergence failures`. No request error appeared. The synthetic candidates remain available in the JSON artifact; the page shows selected summaries and a validation-degree plot, keeping the browser responsive. The official holdout result is still a placeholder.

## Nipitpon-approved official Battle A evaluation ? 2026-11-20 Asia/Bangkok

Nipitpon approved the final review package and authorized one official Bike Sharing holdout evaluation. Before scoring, all **61** files listed in `results/review-manifest.json` matched SHA-256 exactly; no mismatch was found. The immutable approval record `results/approved-snapshot.json` stores the manifest hash `89e262472c1da4c64bee15d8bc4710e16fee33d37ab3895974b673a418f06562`, base commit `bda781cc1dae8bd0493de3f3979e351a05cd6bd7`, and the verification count. The command below was run **once** and exited 0 in 1.99 seconds:

```powershell
.\.venv\Scripts\python.exe -m ridge_lab.official --approved-by Nipitpon --output results/bike-official.json
```

`results/bike-official.json` contains the one-time 2012-08-08 through 2012-12-31 holdout scores. OLS and Ridge tie: RMSE **144.516371**, MAE **101.181910** rentals/hour, R^2 **0.569489**. Lasso: RMSE **144.531361**, MAE **101.187899** rentals/hour, R^2 **0.569399**. The source hash matches the approved UCI file. Development-selected penalties remain Ridge lambda 0 and Lasso alpha 0.001. The earlier development mean-window RMSE values (117.460/117.460/117.474) remain labeled as development results. No choice was changed after official scoring.

The English report and storyboard now carry both scopes, and `report/draft.pdf` was regenerated. Remaining human work: assign the six speaker slots, record all six voices, review the final PDF and video, complete any course submission forms, and submit by the course deadline. G0 member acknowledgments have not been invented.

## Presentation UI verification — 2026-11-20

The original orphan `#studies` section appeared after the footer and expanded the page to 1,905 px on a 390 px viewport. The visible page now keeps the 3D loss surface in the Least Squares Lab with its penalty control, Battle A's separate Bike Sharing and Synthetic views, and Battle B's four measured solvers inside the existing page shell. Saved synthetic/solver artifacts load on a fresh visit; Run All recomputes development studies with progress beside its button. The approved official Bike Sharing table is served only from `results/bike-official.json` via `/api/bike/official` and remains labeled separately from five-window development metrics.

Method trace: OLS, Ridge, and Lasso use `synthetic_study` for the ten-seed validation comparison and `bike_development_payload` for chronological bike development. Normal Equations, QR, SVD, and Batch Gradient Descent use `solver_study` on the same matrix/target per case; each row shows condition/shape context, median runtime, coefficient error, residual norm, input bytes, iteration count or Direct, and status. Iteration-limited GD is labeled and excluded from the exploratory solver order. No numerical method was missing from the implemented experiment flow.

Browser checks at 1365x768 and 390x844: the 3D Plotly surface contains OLS/Ridge markers and displays the matching objective; six main sections stay inside the page shell; neither viewport has page-wide horizontal overflow; the narrow solver table scrolls within its panel. From a fresh start Run All completed all three development/benchmark requests and displayed three synthetic rows, four solver rows, and zero Lasso convergence failures. The 500x16 condition-1e6 case marked GD `iteration_limit` at 10,000 iterations. Changing its runtime weight changed the exploratory solver order while the table stayed identical. Bike tier weights leave the OLS=Ridge > Lasso order unchanged because those two methods dominate every displayed development tier; the page now explains that data property. The official table displayed OLS/Ridge RMSE 144.516 and Lasso 144.531 without invoking official scoring.

The English storyboard was updated for the new demo order. The report's scientific results were unchanged, so its existing PDF remains the approved results draft. Human work remains assigning six speakers, recording the six voices, reviewing the final export and submitting the package.

Final checks: `.\.venv\Scripts\python.exe -m pytest -q` passed **32 tests** with one existing Starlette/httpx deprecation warning; `.\.venv\Scripts\python.exe -m ruff check .` returned `All checks passed!`; `git diff --check` exited 0 with no output. These checks did not invoke the official scorer or change `results/bike-official.json`.
