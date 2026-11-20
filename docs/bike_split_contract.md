> Current owner decision (2026-11-19): Nipitpon is the sole human reviewer. This supersedes project-managed requests below for separate Sitthawee, Sirapobe, or independent reviewer sign-offs. G0 acknowledgments remain unrecorded unless actually supplied. Protocol freeze and official holdout evaluation remain pending Nipitpon's consolidated final review.

# Battle A: Capital Bikeshare schema and split contract

**Status:** dataset, target, feature basis, and chronological split confirmed by the project owner on 2026-11-18. Development-only comparison completed. Official holdout scores remain sealed until the research protocol is frozen and independently reviewed. **Handoff:** Sitthawee (P4) and Sirapobe (P5) should use this same contract for all Battle A results.

## Source and prediction question

“Can we predict how many bikes will be rented in a given hour?” The source is UCI's [Capital Bike Sharing record and data dictionary](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), specifically its [`hour.csv` archive](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip). Attribution: Fanaee-T, H. (2013), *Bike Sharing*, UCI Machine Learning Repository, DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894). The publisher labels it [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain attribution and indicate modifications if redistributed. The source file is bundled at [`data/bike/hour.csv`](../data/bike/hour.csv) for offline use. Its SHA-256 is `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.

Target: **`cnt`**, total bike rentals in one hour (integer count; reported error units: rentals/hour). From the raw audit, `casual + registered = cnt` for all 17,379 records, so those components cannot be predictors. The only predictor inputs are **`hr`** (categorical hours 0–23) and **`workingday`** (binary 0/1). `dteday` is used only to order and split rows. `instant`, weather, season, calendar variables other than the approved two, and all remaining columns are excluded. A row ID is `YYYY-MM-DD:HH`; 17,379 are unique. The loader checks the source hash, original column order, value ranges, and duplicate date-hour keys. No learned preprocessing is needed for the fixed design.

## Fixed feature basis

`make_bike_design(hours, working_days)` yields **48 columns** in the same order for OLS, Ridge, and Lasso:

1. One all-ones intercept.
2. Indicators for hours **1–23**, with hour 0 the reference.
3. One working-day main effect.
4. The **23** products of each hour indicator with working-day status.

In plain English, the model learns a separate baseline for each hour and lets the working-day change differ by hour. The working-day effect at midnight is the main-effect coefficient; at another hour it is that coefficient plus that hour's interaction coefficient. There is no second intercept or full set of 24 hour dummies beside the intercept. The first training fold's 48 columns have full numerical rank. This is a fixed interaction model, **not** a polynomial degree search. The synthetic experiment owns degree selection and ten seeded sensitivity runs.

## Chronological membership

Sort the 731 distinct `dteday` values. Reserve the last **146 dates** (approximately 20%) as one fixed holdout: **2012-08-08 through 2012-12-31, 3,464 rows**. Development is **2011-01-01 through 2012-08-07, 585 dates and 13,915 rows**. Split the development dates into six consecutive blocks of **98, 98, 98, 97, 97, 97 dates**. Fold 1 trains on block 1 and validates on block 2; fold 2 trains on blocks 1–2 and validates on block 3; continue through fold 5. All hours from one date stay together, and every training date precedes its validation dates.

| Fold | Training dates | Training rows | Validation dates | Validation rows |
| --- | --- | ---: | --- | ---: |
| 1 | 2011-01-01–2011-04-08 | 2,259 | 2011-04-09–2011-07-15 | 2,351 |
| 2 | 2011-01-01–2011-07-15 | 4,610 | 2011-07-16–2011-10-21 | 2,335 |
| 3 | 2011-01-01–2011-10-21 | 6,945 | 2011-10-22–2012-01-26 | 2,321 |
| 4 | 2011-01-01–2012-01-26 | 9,266 | 2012-01-27–2012-05-02 | 2,321 |
| 5 | 2011-01-01–2012-05-02 | 11,587 | 2012-05-03–2012-08-07 | 2,328 |

Validation windows are sequential and subject to season and demand shift. They are not five exchangeable random samples. “Unseen” means **later dates from the same bike system**, not a new city. The official holdout is evaluated only once after model choices, ranking rules, and reporting are frozen.

## Shared evaluation and development-only smoke result

All three methods use the same 48 columns, fold memberships, and target. The primary Battle A rank is the **unweighted mean of five validation-window RMSE values** in rentals/hour. Report MAE, R², and RMSE standard deviation across windows beside it. R² is computed against each window's own target mean; it can be negative. Ridge uses `SSE + λ∑β²` and Lasso uses `SSE/(2n) + α∑|β|`, each excluding the intercept. Their penalty numbers are **not directly comparable**. The website also lets users adjust three exploratory tier weights: RMSE, supporting MAE/R², and window RMSE variation. It min-max normalizes each displayed development metric across the three methods, combines MAE and reversed R² equally, then takes a weighted average; lower exploratory score orders first. This uses the rounded development snapshot, never the sealed holdout, and cannot replace the primary RMSE ranking. Battle B's solver ranking remains separate.

The following is a **development-only smoke comparison**, computed on the five validation windows on 2026-11-18. It is not a test or final generalization claim. The examined Ridge `λ` grid was `0, 1, 10, 100`; the Lasso `α` grid was `0.001, 0.01, 0.1, 1`. Lowest mean validation RMSE, then lower penalty on a tie, chooses settings *within each method*. OLS has no penalty. These selected settings are the proposed final refit settings for the protocol-freeze review; no holdout score was used.

| Method | Development-selected setting | Mean RMSE (rentals/hour) | Mean MAE (rentals/hour) | Mean R² | RMSE SD across windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLS | — | 117.460 | 83.835 | 0.471 | 36.719 |
| Ridge | λ = 0 | 117.460 | 83.835 | 0.471 | 36.719 |
| Lasso | α = 0.001 | 117.474 | 83.838 | 0.471 | 36.728 |

| Window | OLS RMSE | Ridge RMSE | Lasso RMSE |
| --- | ---: | ---: | ---: |
| 1 | 145.914 | 145.914 | 145.936 |
| 2 | 88.495 | 88.495 | 88.512 |
| 3 | 64.822 | 64.822 | 64.815 |
| 4 | 122.708 | 122.708 | 122.724 |
| 5 | 165.359 | 165.359 | 165.382 |

Ridge selects `λ=0` on this small grid and therefore equals OLS under the stated solver convention. That is a valid observation, not evidence that regularization never helps. The large window variation limits any claim of a stable winner. P4/P5 should confirm the grid, penalty conventions, tie rule, and final refit on all 13,915 development rows **before** official holdout scoring. The scoring and any post-test interpretation belong to the existing protocol-freeze stage; do not revise the choices after seeing holdout results.

## Protocol-freeze review — pending

**Sitthawee (P4) and Sirapobe (P5):** check each item against `bike.py`, `features.py`, `solvers.py`, and `bike_experiment.py`. Record a dated response for each reviewer below; a technical sign-off does not replace the independent PR review required by `AGENTS.md` (Sirapobe cannot self-approve Ridge code). **Do not mark this protocol frozen or score the holdout until both reviews and the required independent code review are recorded.**

- [ ] **Features and leakage:** Confirm `cnt` is the only target; the 48 columns are 1 intercept + 23 indicators for hours 1–23 (hour 0 reference) + 1 `workingday` main effect + 23 hour × `workingday` terms. Confirm no duplicate intercept, `dteday` is split-only, and `instant`, `casual`, `registered`, weather, and every other raw field are excluded. The raw audit found `casual + registered = cnt` for every row.
- [ ] **Objectives:** Confirm OLS minimizes `SSE`; Ridge minimizes `SSE + λ∑_{j>0}β_j²` using `λ ∈ {0, 1, 10, 100}`; Lasso minimizes `SSE/(2n) + α∑_{j>0}|β_j|` using `α ∈ {0.001, 0.01, 0.1, 1}`. Column 0 is unpenalized for both. Ridge's λ is on the *sum-SSE* scale; its equivalent coefficient beside `SSE/(2n)` is `λ/(2n)`, so Ridge λ and Lasso α are not numerically interchangeable. The fixed basis needs no fitted preprocessing; any later learned transform must be fit on each training window only.
- [ ] **Validation and selection:** Confirm the five exact expanding date/row windows in the table above, same-date grouping, and no holdout date in development. For each candidate, compute RMSE within each validation window, then take the **unweighted arithmetic mean of the five RMSEs** as the primary selection/ranking value. Report mean MAE, mean R², each window's RMSE, and population SD of the five RMSEs beside it. Within each method, choose the lowest *unrounded* mean RMSE, breaking an exact tie by the smaller penalty. Report equal cross-method RMSE as a tie; exploratory tier weights cannot alter the primary rank. Confirm the development-selected settings OLS, Ridge `λ=0`, and Lasso `α=0.001` are acceptable for freeze.
- [ ] **Final refit:** After sign-off, build the same 48-column basis for **all 13,915 development rows** and fit each method once using its selected setting. Preserve the source checksum, split boundaries, feature order, grids, metric aggregation, and selected settings as the frozen record. Only then perform the one-time official holdout evaluation; do not revise settings after seeing it.

**Scope check:** `task.md` §5, `docs/decisions.md`, and this contract consistently assign polynomial degree selection and the ten seeded sensitivity runs to the **synthetic** experiment. Battle A has no seeded repeats; its five validation windows are chronological. The older airfoil proposal in `datasets_review.md` is historical and is superseded by the owner's bike decision. No protocol change or Nipitpon clarification is needed on this point.

**Open before freeze:** Sitthawee's and Sirapobe's review responses are not recorded; the independent PR review is not recorded; the proposed final-refit settings have not been signed off; G0's proposed P1–P6 role allocations and interfaces still lack the team's acknowledgments. None of these statuses is implied by the development-only smoke comparison. The sealed holdout remains reserved.

| Reviewer | Dated decision and notes |
| --- | --- |
| Sitthawee (P4) | Pending |
| Sirapobe (P5) | Pending |
