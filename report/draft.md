# The Least Squares Showdown - report draft

Can we predict how many bikes will be rented in a given hour?

## Mathematical foundation

For design A, coefficients beta, and observations y, OLS minimizes ||y-A beta||^2. At a minimizer the residual is orthogonal to the columns of A, giving A^T A beta=A^T y. Forming A^T A squares the condition number; QR or SVD avoids explicitly inverting it. Ridge adds lambda times the squared non-intercept coefficients. Lasso uses SSE/(2n) plus alpha times their absolute values. The intercept remains unpenalized; the two penalty scales are not interchangeable.

## Battle A: real-world development results

UCI Bike Sharing, Fanaee-T (2013), DOI 10.24432/C5W894, CC BY 4.0. The fixed 48-column design uses hour, working-day status, and interactions. Rental components reconstruct the target and are excluded. The 13,915 development rows end on 2012-08-07. Five expanding windows estimate future-date behavior within this system. Mean window RMSE is unweighted; these windows are temporally dependent.

| Model | Penalty | RMSE rentals/hour | MAE | R^2 | Window RMSE SD |
|---|---:|---:|---:|---:|---:|
| OLS | 0.0 | 117.460 | 83.835 | 0.471 | 36.719 |
| Ridge | 0.0 | 117.460 | 83.835 | 0.471 | 36.719 |
| Lasso | 0.001 | 117.474 | 83.838 | 0.471 | 36.728 |

OLS and Ridge tie because selected lambda is zero. This is not evidence that regularization never helps. Window variation is large relative to the differences. The API regenerates the table from actual development output. Refit uses every development row once; reserved targets are masked in the normal loader.

## Official Battle A Bike Sharing holdout

Nipitpon approved the frozen pre-scoring snapshot (61 manifest files, no mismatches) before the one official evaluation. The holdout spans 2012-08-08 through 2012-12-31: 3,464 later-date rows. The selected penalties and 48-column design were fixed from development; all three models were refit on the 13,915 development rows. This table is the one-time official future-date evaluation, distinct from the five-window development table above.

| Model | Development-selected penalty | Official RMSE (rentals/hour) | Official MAE (rentals/hour) | Official R^2 |
|---|---:|---:|---:|---:|
| OLS | none | 144.516 | 101.182 | 0.569 |
| Ridge | lambda = 0 | 144.516 | 101.182 | 0.569 |
| Lasso | alpha = 0.001 | 144.531 | 101.188 | 0.569 |

OLS and Ridge tie exactly because Ridge lambda is zero and uses the same coefficients. Lasso differs by about 0.015 rentals/hour in RMSE; this single future block does not establish a general winner across cities, years, or conditions. The official results were recorded in `results/bike-official.json`; no model choice was revised from the holdout.

## Synthetic sensitivity study

Seeds 42-51 use independent train/validation/test samples with 24/24/120 observations. Degrees 1-15 and the grids in docs/decisions.md are selected by validation MSE on train-standardized features. The following are means of selected validation scores, not unbiased test errors.

| Noise SD | OLS | Ridge | Lasso |
|---|---:|---:|---:|
| 0.1 | 0.011672 | 0.010059 | 0.014724 |
| 0.2 | 0.042793 | 0.039585 | 0.044113 |
| 0.4 | 0.165028 | 0.157390 | 0.165766 |

The original 10,000-iteration run failed for 365 of 1,800 Lasso configurations (20.28%). A 50,000-iteration bounded run with a numerically equivalent gradient update and an active-set KKT correction reduced failures to zero. The original exclusions changed 21 of 30 Lasso-selected degrees; all selected alpha values remain 0.001. The mean validation ranking remains Ridge, OLS, Lasso at each noise level. The final converged fits required 1 to 30,500 iterations, with maximum KKT residual 9.61e-9. These are selected validation scores, so they are optimistic estimates of future performance. The baseline implementation raised RuntimeError without iteration or KKT details; the final run records both for every Lasso candidate. No solver warning was emitted. No synthetic test score was opened. The reproduction artifact also retains fixed-degree-15 OLS/Ridge selections and all converged candidate scores; the site plots the seed-42 validation curve.

## Battle B measured benchmark

Three repeats per method/case; median perf_counter wall time, no warmup isolation. Seed 42 creates matrices with prescribed singular values and noiseless y=A beta. Input bytes exclude solver workspace and are not peak memory. GD minimizes half-SSE, starts at zero, step 1/||A||^2, tolerance 1e-8, cap 10,000. Timing includes the GD step-size computation. Machine-specific timing is descriptive.

| Shape | Condition | Solver | Seconds | Relative coefficient error | Status |
|---|---:|---|---:|---:|---|
| 100x8 | 1e+01 | Normal Equations | 0.000063 | 4.807e-15 | completed |
| 100x8 | 1e+01 | QR | 0.000049 | 3.545e-16 | completed |
| 100x8 | 1e+01 | SVD | 0.000104 | 7.535e-16 | completed |
| 100x8 | 1e+01 | Batch Gradient Descent | 0.008906 | 4.629e-07 | completed |
| 100x8 | 1e+06 | Normal Equations | 0.000010 | 1.450e-05 | completed |
| 100x8 | 1e+06 | QR | 0.000044 | 3.024e-12 | completed |
| 100x8 | 1e+06 | SVD | 0.000029 | 1.164e-12 | completed |
| 100x8 | 1e+06 | Batch Gradient Descent | 0.075927 | 8.703e-01 | iteration_limit |
| 500x16 | 1e+01 | Normal Equations | 0.000022 | 5.730e-15 | completed |
| 500x16 | 1e+01 | QR | 0.000116 | 6.488e-16 | completed |
| 500x16 | 1e+01 | SVD | 0.000088 | 1.820e-15 | completed |
| 500x16 | 1e+01 | Batch Gradient Descent | 0.009999 | 2.622e-07 | completed |
| 500x16 | 1e+06 | Normal Equations | 0.000021 | 1.939e-05 | completed |
| 500x16 | 1e+06 | QR | 0.000111 | 6.948e-12 | completed |
| 500x16 | 1e+06 | SVD | 0.000094 | 8.605e-12 | completed |
| 500x16 | 1e+06 | Batch Gradient Descent | 0.087267 | 6.652e-01 | iteration_limit |

Two ill-conditioned GD cases reach the iteration limit. A small residual does not guarantee accurate coefficients in an ill-conditioned system. The benchmark is bounded and does not establish universal speed rankings. The separate full 3D intercept/slope demonstration uses SSE + lambda*slope^2 with a free intercept.

## Reproducibility and limits

See README reproduction commands and results/*.json for actual outputs. The approved Bike Sharing settings were frozen for the one official evaluation. Synthetic experiment scope remains separate. Bike exploratory tier weights do not alter selected settings. Six speaking slots are in docs/storyboard.md; member acknowledgments and recorded voices have not been fabricated.
