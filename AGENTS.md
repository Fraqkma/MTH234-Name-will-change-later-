> Current owner decision (2026-11-19): Nipitpon is the sole human reviewer. This supersedes project-managed requests below for separate Sitthawee, Sirapobe, or independent reviewer sign-offs. G0 acknowledgments remain unrecorded unless actually supplied. Protocol freeze and official holdout evaluation remain pending Nipitpon's consolidated final review.

# AGENTS.md — Instructions for Codex and Collaborating Coding Agents

## Project mission

Build **The Least Squares Showdown**, a reproducible, six-member **Linear Algebra** educational project demonstrating **Least Squares -> polynomial overfitting -> Ridge Regression -> validation and generalization**. Main deliverables: modular Python package, interactive vanilla HTML/CSS/JavaScript website served by FastAPI, mathematically faithful report, and 10-minute video with six approximately equal speakers. Deadline: **2026-12-01 (Asia/Bangkok)**.

**Read `task.md` before changing code.** Treat it as the roadmap and acceptance criteria. The uploaded group README and Thai/English PDF plans define the original educational scope; `task.md` explicitly identifies corrections needed for scientific validity. Do not silently discard either. **Confirmed: all report, website, slides, video narration, README and project artifacts must be in English. Internal team conversation may be in Thai.**

**Repository setup:** Continue in this existing The Least Squares Showdown repository. Do not create another local repository or a public/private GitHub remote, select an owner/visibility, invite collaborators, or push without authorization. Do not accidentally edit a different existing project.

## Operating rules

1. **Scope first:** work only on the assigned phase, issue or user instruction. For an initial request, implement **Phase 0**, not the entire project. Report what was done and what remains.
2. **Preserve existing work:** inspect git status and file contents first. Never overwrite team work, remove tests, or force-reset branches. Do not commit artifacts merely because they are generated.
3. **Source of truth:** `src/ridge_lab/` owns data, features, math, diagnostics and evaluation. `web/` is a thin UI layer. `scripts/` orchestrate reproducible runs; notebooks are exploratory only.
4. **One public API:** prefer stable named functions and well-defined types. Coordinate interface changes across P2–P5; update imports, tests and documentation atomically.
5. **No fabricated results:** never invent scientific measurements, claim guaranteed overfitting/Ridge improvement, or present sample graphs as actual results. Clearly label synthetic, illustrative and held-out results.
6. **No leakage:** never use test targets for normalization, model/degree/`lambda` selection or UI tuning of the official evaluation. Learn preprocessing only on train. Validation selects hyperparameters. Sealed test is used only after freezing selections; any exploratory UI access to test must be marked non-official.
7. **Numerical integrity:** textbook normal equations are mandatory to **explain**, not mandatory to compute via explicit inversion. Use stable OLS `numpy.linalg.lstsq` or QR; use stable Ridge solving where possible. Detect singularity/ill-conditioning instead of hiding warnings.
8. **Ridge convention:** objective `||y-Aw||² + λ∑_{j>0}w_j²`, first column intercept unpenalized; use `P=diag(0,1,...,1)`. Verify `lambda=0` equals OLS on the same transformed features. `lambda >= 0` only. If showing the textbook-style `λI`, explicitly explain the intercept-penalizing difference.
9. **Reproducibility:** default seed 42, documented config, split IDs, deterministic transforms, fixed snapshots of result choices. Browser requests use the same seeded data unless the user explicitly starts a fresh simulation.
10. **Keep dependency footprint small:** Python/NumPy/FastAPI plus testing tools. Plotly.js is bundled locally for charts. scikit-learn is optional for independent numerical cross-checks; do not use it to replace the educational implementation. Free tools only.

## Named team responsibilities and governance

**Confirmed leadership (both must implement, test, write and speak):**
- **Nipitpon Choojit — Project Manager, P6 working role.** Own schedule, issue/PR triage, milestones, release readiness and final submission checklist. Hands-on code owner of the FastAPI/static-site shell, shared UI components, smoke/E2E tests and integration; writes ML discussion/conclusion and narrates a ~100-second segment. Do not assign all app page content or QA exclusively to Nipitpon.
- **Sirapobe Parinyarat (Sirapob) — Technical Advisor / Research Guide, P5 working role.** Guide scientific design, numerical correctness, Ridge conventions and train/validation/test protocol, unblock team design decisions. Hands-on owner of Ridge solver, lambda-selection experiments/tests, Ridge analysis and ~100-second narration. Advisor is **not** a purely supervisory/reviewer position and cannot self-approve changes.

**Proposed additional work allocation (not yet individually acknowledged by the four members):**
- **P1 Nassasi Treesuptavee:** theory, projection, normal equation proof, report section and Least Squares page copy; reviewer P2.
- **P2 Thanatorn Charoensuk:** hand examples, OLS/QR, numerical tests, report section; reviewer P5.
- **P3 Parama Polsen:** deterministic datasets, polynomial features, scaling, baseline; first stable API handoff; reviewer P4.
- **P4 Sitthawee Whangsawangwong:** overfitting degree sweep, MSE, conditioning diagnostics, result charts; reviewer P3.
- **P5 Sirapobe:** technical advisor **plus Ridge implementation**; reviewer P2.
- **P6 Nipitpon:** PM **plus app/integration implementation**; reviewer P1 (written/UI) and P4 (results).

Members may negotiate a swap of the *technical work allocation* at kickoff, but update `task.md` and related issues and record reviewer changes first. Do not quietly treat a suggestion as team-confirmed. The two leadership titles are user-confirmed.

**Decision ownership:** P6 coordinates scope/timing, P5 guides research/math, each feature owner implements their module, an uninvolved reviewer checks every PR, and scientific quality gates cannot be waived by either leader. Breaking API changes require affected module owners' agreement. Every member authors a section, provides graph/theory content and speaks approximately equally in the ten-minute video.

## Architecture boundaries

```text
config/datasets/features -> solvers -> experiments/evaluation -> web/scripts -> report
```

- `src/ridge_lab`: pure functions or typed immutable models; **must not import FastAPI, Plotly.js, frontend modules or display code**.
- `api/main.py`: FastAPI routes and static-file mounting only. It validates inputs and calls reusable core functions.
- `frontend/`: semantic HTML, design tokens, CSS and vanilla ES modules. Browser code renders returned results; it must not implement fitting or data splitting.
- `tests`: exercises invariants, numerical examples, reproducibility and absence of test leakage.
- `report/sections`: each contributor owns an independent section, submitted via PR. Charts/tables derive from versioned results and are consistent with the website.
- Avoid unnecessary services, backends, databases or JavaScript build pipelines. Do not introduce React without an explicit scope change.

## Cross-team boundaries and reviews

- **P1 Nassasi:** least squares theory/projection, educational explanations and relevant written/visual content.
- **P2 Thanatorn:** hand examples, OLS/QR and numerical-reliability tests.
- **P3 Parama:** deterministic dataset, `make_A`, scaling, frozen metadata, baseline; **publish contract early**.
- **P4 Sitthawee:** degree sweep, overfitting metrics/conditioning diagnostics and plots.
- **P5 Sirapobe:** Ridge solver, validation-selected lambda, sensitivity study, technical guidance and code tests.
- **P6 Nipitpon:** ML interpretation, web shell/integration code, report/slides/video coordination, CI and smoke tests.
- Coordinate shared modules: P2/P5 on `solvers.py`; P4/P5 on `experiments.py`. Avoid duplicate implementations or unmerged concurrent edits.
- P6 handles design system, navigation and integration; **P1–P5 own substantive content, component inputs and verification for their pages**, so P6 does not become the only UI or verification contributor.
- Use feature branches/PR review; no direct main pushes, no self-approval. Use assigned reviewers in `task.md` or document an independent substitute.
- Each PR must state behavior, mathematical/data implications, tests actually executed, visuals/docs touched, risks, and cross-owner API impact. Do not invent team approval.

## Scientific guardrails

- Dataset for the initial controlled demo: `sin(2πx)` plus seeded Gaussian noise. Record noise std and sample counts; dataset/test configuration may only be revised **before** official held-out evaluation, with a documented decision.
- Degrees 1–15 illustrate the complexity trade-off; a U-shaped validation curve is a **hypothesis**, not a guaranteed output.
- Distinguish **overfitting** (generalization gap) from **ill-conditioning** (sensitivity of numerical solutions). `cond(A^T A)` may overflow/appear infinite due to floating point precision; label it explicitly.
- Polynomial feature scaling must not use validation or test statistics. Keep intercept semantics correct after scaling.
- Do not compare different losses while using identical lambda labels unless their normalization conventions are stated.
- For 3D visuals, a two-parameter model has a genuine full 3D loss landscape. Higher-degree models have more parameters, so show a clearly labeled 2D parameter slice if plotting them in 3D.
- Show Ridge **can** improve generalization, not that it always will; include over-regularization / underfitting cases and limitations.
- Do not call every optimization path "model learning" unless the actual iterative algorithm (e.g., gradient descent) is implemented; an algebraic Ridge solution need not use gradient descent.

## Development and tool usage

- Before work, detect available tools and installed **skills/MCPs**. Never assert a connection exists without checking.
- **Context7** if installed: check official package/API documentation when unsure about FastAPI validation/static serving or Plotly.js charts.
- **GitHub MCP** if installed: work with authorized issues/PRs, CI status and reviews. Ask before creating/merging remote repositories or making broad irreversible changes.
- **Playwright MCP** if installed: UI smoke/E2E tests for page navigation, sliders, responsive charts and error states. Save only useful failure evidence.
- Other previously installed skills (architecture/code-review helpers, REA, Node REPL, etc.) are optional and purpose-limited. Never install a paid service or a plugin because it sounds impressive.
- Do not embed credentials or private data in commits. Keep large videos/binary exports out of Git unless explicitly approved.

## Quality gates (run what environment supports)

```powershell
# Windows PowerShell example after the local API and frontend are installed
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m uvicorn api.main:app --reload
```

- When commands are not executable in the agent environment, report that candidly and provide an exact command for the team to run. Never mark a test as passed without running it.
- Format/typing style should be pragmatic, with clear docstrings, useful error messages and small functions; avoid elaborate abstractions in a three-week student project.
- All experiments should work offline after installing dependencies.
- At project handoff, include a concise implementation log, current gate (G0–G4), blocked items, file ownership, and next steps.

## Written deliverables and video

- **English is required** for all academic report prose, website content/UI labels, slide text, narration/video scripts, project README, figure annotations and code documentation. Thai is permissible only for informal team communication outside final artifacts. Preserve mathematical notation consistently across media.
- Final report must include problem formulation, textbook connection, dataset protocol, train/validation/test distinction, numerical methods, plots and real values, independent evaluation, limitations, citations, contributions.
- Target six equal **~1:40** video segments covering the six respective sections; each student actually speaks their own section. Do not silently replace the team recording with generated audio or claim a video exists without it.
- Final video and final report are due **2026-12-01**. Arrange internal freeze by Nov 30.

## When uncertain

Make reversible, scientifically safe defaults and document them under `task.md` pending decisions. **Leadership, English language and new repository choice are decided**; do not ask again. The named P1–P6 assignments are a proposed working allocation awaiting team acknowledgment, while the GitHub remote owner/visibility/handles are unknown. Do not block local scaffolding on remote details. Ask when an external side effect, ownership, or submission decision materially requires authorization.
