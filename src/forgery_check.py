from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_license_manifest(base_dir: str | Path) -> pd.DataFrame:
    base_path = Path(base_dir)
    return pd.read_csv(base_path / "data" / "license_manifest.csv")


def simple_forgery_score(issue_type: str) -> float:
    weights = {
        "clean_template": 0.05,
        "text_tamper": 0.80,
        "photo_swap": 0.90,
        "noise_overlay": 0.65,
    }
    return weights.get(issue_type, 0.50)
