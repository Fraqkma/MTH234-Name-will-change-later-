"""FastAPI adapter for the framework-independent ridge_lab package."""

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles

from ridge_lab.demo import build_demo

app = FastAPI(title="When Perfect Fit Fails", docs_url=None, redoc_url=None)
frontend = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo")
def demo(
    degree: Annotated[int, Query(ge=1, le=15)] = 9,
    lam: Annotated[float, Query(alias="lambda", ge=0, le=1000)] = 0.1,
) -> dict[str, object]:
    return build_demo(degree, lam)


app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
