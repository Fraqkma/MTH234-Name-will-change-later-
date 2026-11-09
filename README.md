# When Perfect Fit Fails

An interactive linear algebra lesson on least squares, polynomial overfitting, and Ridge Regression. The browser interface is vanilla HTML, CSS, and JavaScript; a thin FastAPI service calls the framework-independent `ridge_lab` package.

The current interface is an **illustrative seeded demo**, not a report of final experiments. It fits coefficients on training data and displays train and validation metrics. Test observations and test scores are not returned to the browser.

## Run locally

Python 3.13 is recommended; Python 3.11 or newer is supported.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000`. Plotly.js is served from the local `frontend/vendor/` directory, so the app does not need a CDN connection.

## Verify

```bash
python -m pytest
python -m ruff check .
```

## Project structure

- `frontend/` contains the responsive single-page lesson, design tokens, ES modules, and local Plotly.js build.
- `api/` serves the static site and validates the demo API parameters.
- `src/ridge_lab/` contains deterministic data, polynomial features, train-only scaling, stable OLS/Ridge solvers, and metrics. It does not import FastAPI or frontend code.
- `docs/decisions.md` records the architectural migration and pending method-owner sign-off; `task.md` remains the project roadmap.

The original group README is preserved at [`docs/original-group-readme.md`](docs/original-group-readme.md). The project uses a deterministic synthetic signal, `sin(2πx) + Gaussian noise`; validation is for comparison, and final test evaluation stays reserved until model choices are frozen.

Plotly.js is distributed under the MIT license; its license is included in `frontend/vendor/plotly-LICENSE.txt`. Design guidance was adapted from Unslop by Damir Sokolovsky under CC BY 4.0.
