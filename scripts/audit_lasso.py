"""Summarize the three saved synthetic sweeps without reading test targets."""

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    saved = {
        name: json.loads(Path(f"results/synthetic{suffix}.json").read_text(encoding="utf-8"))
        for name, suffix in (("baseline", "-before"), ("bounded", "-intermediate"),
                             ("final", ""))
    }
    summary = {}
    for name, study in saved.items():
        failures = study["failures"]
        summary[name] = {
            "attempted_fits": 30 * 15 * 4,
            "unique_configurations": 30 * 15 * 4,
            "validation_splits": 30,
            "cv_folds_per_split": 0,
            "failed_fits": len(failures),
            "failed_unique_configurations": len({
                (f["seed"], f["noise"], f["degree"], f["penalty"]) for f in failures
            }),
            "by_seed": dict(sorted(Counter(f["seed"] for f in failures).items())),
            "by_noise": dict(sorted(Counter(f["noise"] for f in failures).items())),
            "by_degree": dict(sorted(Counter(f["degree"] for f in failures).items())),
            "by_alpha": dict(sorted(Counter(f["penalty"] for f in failures).items())),
            "by_split": dict(sorted(Counter(
                f"seed={f['seed']},noise={f['noise']}" for f in failures
            ).items())),
            "runtime_seconds": study["elapsed_seconds"],
        }
    first = {(r["seed"], r["noise"], r["method"]): r for r in saved["baseline"]["rows"]}
    changes = []
    for result in saved["final"]["rows"]:
        key = result["seed"], result["noise"], result["method"]
        previous = first[key]
        if (previous["degree"], previous["penalty"]) != (result["degree"], result["penalty"]):
            changes.append({"seed": key[0], "noise": key[1], "method": key[2],
                            "before_degree": previous["degree"],
                            "after_degree": result["degree"],
                            "before_penalty": previous["penalty"],
                            "after_penalty": result["penalty"],
                            "before_validation_mse": previous["validation_mse"],
                            "after_validation_mse": result["validation_mse"]})
    summary["selection_changes"] = changes
    Path("results/lasso-convergence-audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print({name: item["failed_fits"] for name, item in summary.items() if name != "selection_changes"})
    print("selection changes", len(changes))


if __name__ == "__main__":
    main()
