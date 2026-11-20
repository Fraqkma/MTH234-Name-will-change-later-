"""Generate local development artifacts without official scoring."""
import json
import time
from pathlib import Path

from ridge_lab.bike_experiment import bike_development_payload
from ridge_lab.studies import solver_study, synthetic_study


def main():
    Path("results").mkdir(exist_ok=True)
    for name, function in (("bike", bike_development_payload),
                           ("solvers", solver_study), ("synthetic", synthetic_study)):
        start = time.perf_counter()
        result = function()
        result["elapsed_seconds"] = time.perf_counter() - start
        Path(f"results/{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(name, result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
