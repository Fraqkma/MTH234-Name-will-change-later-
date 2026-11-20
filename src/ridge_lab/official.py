"""Explicit post-review scoring entry point; never imported by the site."""
import argparse
import csv
import json
from datetime import date

import numpy as np

from ridge_lab.bike import DEFAULT_HOUR_CSV, load_bike_hourly, make_bike_split
from ridge_lab.bike_experiment import (
    evaluate_bike_development,
    refit_bike_development,
    select_bike_settings,
)
from ridge_lab.features import make_bike_design
from ridge_lab.metrics import mae, r2, rmse


def score_models(models, hours, working, targets):
    design = make_bike_design(hours, working)
    return [{"method": m.method, "rmse": rmse(targets, design @ m.coefficients),
             "mae": mae(targets, design @ m.coefficients),
             "r2": r2(targets, design @ m.coefficients)} for m in models]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-by", required=True, choices=["Nipitpon"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.fixture:
        from ridge_lab.bike_experiment import BikeFittedModel

        coefficients = np.zeros(48)
        coefficients[0] = 2
        result = score_models((BikeFittedModel("OLS", 0, coefficients),),
                              np.array([0, 1]), np.array([0, 1]), np.array([1., 3.]))
        with open(args.output, "x", encoding="utf-8") as output:
            json.dump({"scope": "synthetic scoring fixture", "results": result}, output)
        return
    data = load_bike_hourly()
    split = make_bike_split(data)
    selected = select_bike_settings(evaluate_bike_development(data, split))
    fitted = refit_bike_development(data, split, selected)
    with DEFAULT_HOUR_CSV.open(encoding="utf-8", newline="") as source:
        rows = [r for r in csv.DictReader(source)
                if date(2012, 8, 8) <= date.fromisoformat(r["dteday"]) <= date(2012, 12, 31)]
    rows.sort(key=lambda r: (r["dteday"], int(r["hr"])))
    assert len(rows) == 3464
    result = score_models(fitted.models, np.array([int(r["hr"]) for r in rows]),
                          np.array([int(r["workingday"]) for r in rows]),
                          np.array([float(r["cnt"]) for r in rows]))
    with open(args.output, "x", encoding="utf-8") as output:
        json.dump({"scope": "official holdout", "approved_by": args.approved_by,
                   "source_sha256": data.sha256, "results": result}, output, indent=2)


if __name__ == "__main__":
    main()
