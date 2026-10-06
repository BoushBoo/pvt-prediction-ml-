"""Input/schema validation; never impute, shuffle, deduplicate, or drop rows."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Tf", "Rs", "gg", "api"]
CANONICAL = {x.lower(): x for x in FEATURES + ["Pb", "Bob"]}


def load_data(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path)
    raise ValueError("Expected .csv, .xls or .xlsx input")


def validate_data(df, target):
    target = CANONICAL.get(str(target).strip().lower())
    if target not in ("Pb", "Bob"):
        raise ValueError("target must be Pb or Bob")
    names = [CANONICAL.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    if len(set(names)) != len(names):
        raise ValueError("Column normalization produced duplicate names")
    normalized = df.set_axis(names, axis=1)
    missing = set(FEATURES + [target]) - set(names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = normalized[FEATURES + [target]].astype(float)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(
            "Inputs/target contain missing or nonfinite values; no automatic cleaning is performed"
        )
    return data, target
