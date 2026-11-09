# One-Page Project Storyboard

**Draft for P1 Nassasi and P6 Nipitpon review; no co-authorship or approval is implied.** English-only project copy. No final experiment conclusions are shown.

## Audience and question

Linear Algebra students who know projections and least squares. The demo asks: **Does the curve with the smallest training error predict new observations best?**

## Page flow

| Screen | What the learner sees | Takeaway |
|---|---|---|
| Overview | Question, the noisy sine-data setup, and links to each lesson | A close fit to observed points can still generalize poorly. |
| Least Squares | Projection picture, a small normal-equation example, and QR/SVD note | Least squares projects the observation vector onto the model's column space. |
| Overfitting | Train/validation scatter, degree control, error-by-degree and conditioning context | Generalization error and numerical conditioning are related concerns but are not the same thing. |
| Ridge Regression | Same-feature OLS/Ridge overlay, lambda control, coefficient comparison, and validation curve | Regularization trades fit against coefficient size; it can underfit when too strong and does not always improve held-out error. |
| Results | Frozen validation choices, final sealed-test table, configuration and limitations | Test data evaluates a frozen choice; it does not select one. |

## Shared visual and interaction rules

- Show split labels and define every metric; mark illustrative views as such and never invent measurements.
- Keep degree and lambda controls consistent across pages; default random seed is 42 and reruns do not resample.
- Make the train/validation/test distinction visible wherever results appear.
- Keep explanations next to plots and state when a 3D surface is a two-parameter slice.

## Visual direction

Measured, academic, tactile. The visual world is a working mathematics notebook: quiet paper-like neutrals, ink, and plotted evidence. The main fit plot carries the color; a fine residual marker identifies the difference between an observed point and its fitted value.

## Current implementation status

The responsive single-page HTML interface, FastAPI adapter, seeded fit plot, and coefficient comparison now exist. The QR solver comparison, degree sweep, lambda sweep, and official held-out results remain future work. This storyboard is still a review target for the named content owners, not a claim that they approved the page copy.
