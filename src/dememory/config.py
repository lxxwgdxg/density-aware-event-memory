from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    root = Path(cfg["project_root"]).resolve()
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = root
    for key in [
        "raw_external",
        "interim",
        "processed",
        "outputs",
        "caravan_zip",
        "grdc_caravan_zip",
        "koppen_zip",
    ]:
        cfg[key] = str((root / cfg[key]).resolve())
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ["interim", "processed", "outputs"]:
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)

