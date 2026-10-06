import argparse
import hashlib
from pathlib import Path

from .data import load_data
from .experiment import run_experiment
from .models import ALL_MODELS, PHASE1, PHASE2


def main():
    p = argparse.ArgumentParser(description="Reproducible nested-CV PVT evaluation")
    p.add_argument("--data", required=True)
    p.add_argument("--target", choices=["Pb", "Bob"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models", nargs="+", choices=ALL_MODELS)
    p.add_argument("--phase", choices=["1", "2", "all"], default="all")
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    path = Path(a.data)
    models = a.models or {"1": PHASE1, "2": PHASE2, "all": PHASE1 + PHASE2}[a.phase]
    source = {
        "filename": path.name,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    print(
        run_experiment(
            load_data(path),
            a.target,
            a.output,
            models,
            a.outer_folds,
            a.inner_folds,
            a.seed,
            source=source,
        )
    )


if __name__ == "__main__":
    main()
