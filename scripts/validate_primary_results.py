from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVENTS = ("Q95", "Q98", "Q99")
HORIZONS = (1, 5, 7)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_report(data_dir: Path) -> dict[str, object]:
    primary = pd.read_csv(data_dir / "candidate_unit_metrics_with_anchor_gains.csv")
    primary = primary[
        primary["dataset_version"].eq("fully_deduplicated")
        & primary["candidate"].eq("op_lead_adaptive_direct")
    ].copy()
    require(len(primary) == 450, f"Expected 450 primary units, found {len(primary)}")
    require(primary["source"].nunique() == 10, "Expected ten reporting groups")
    require(set(primary["event"]) == set(EVENTS), "Unexpected event-threshold grid")
    require(set(primary["horizon"].astype(int)) == set(HORIZONS), "Unexpected forecast-lead grid")

    nonnegative = int(primary["daily_ap_gain_vs_anchor"].ge(-1e-12).sum())
    require(nonnegative == 449, f"Expected 449 nonnegative primary units, found {nonnegative}")
    cell = (
        primary.groupby(["event", "horizon"], sort=True)["daily_ap_gain_vs_anchor"]
        .median()
        .round(10)
    )
    require(len(cell) == 9 and bool((cell > 0).all()), "All nine cell medians must be positive")

    nested_folds = pd.read_csv(data_dir / "nested_loso_by_fold.csv")
    nested_cells = pd.read_csv(data_dir / "nested_loso_cell_summary.csv")
    require(len(nested_folds) == 90, f"Expected 90 nested folds, found {len(nested_folds)}")
    require(int(nested_folds["heldout_median_daily_ap_gain"].ge(-1e-12).sum()) == 90, "Nested fold medians are not all nonnegative")
    require(int(nested_folds["heldout_median_daily_ap_gain"].gt(1e-12).sum()) == 81, "Expected 81 strictly positive nested fold medians")
    nested_nonnegative_units = int(nested_cells["nonnegative_heldout_units"].sum())
    require(nested_nonnegative_units == 447, f"Expected 447 nonnegative nested units, found {nested_nonnegative_units}")

    bootstrap = pd.read_csv(data_dir / "source_block_bootstrap_anchor.csv")
    require(len(bootstrap) == 9, f"Expected nine source-block cells, found {len(bootstrap)}")
    require(bool(bootstrap["ci025"].gt(0).all()), "All source-block lower confidence bounds must be positive")

    postlock = pd.read_csv(data_dir / "postlock_lamahice_anchor_unit_metrics.csv")
    postlock = postlock[postlock["dataset_version"].eq("fully_deduplicated")].copy()
    require(len(postlock) == 45, f"Expected 45 post-lock units, found {len(postlock)}")
    require(bool(postlock["ap_gain_vs_anchor"].ge(-1e-12).all()), "Post-lock AP gains must be nonnegative")
    require(set(postlock["gauges"].astype(int)) == {33}, "Expected 33 post-lock gauges")

    inventory = pd.read_csv(data_dir / "supplementary_table2_gauge_identity_audit.csv")
    final_inventory = inventory[
        inventory["scope"].eq("primary ten-source inventory")
        & inventory["stage"].eq("fully_deduplicated")
    ]
    require(len(final_inventory) == 1 and int(final_inventory.iloc[0]["count"]) == 981, "Expected 981 final gauges")

    episode = pd.read_csv(data_dir / "anchor_and_episode_cell_summary.csv")
    episode = episode[episode["dataset_version"].eq("fully_deduplicated")].copy()
    require(len(episode) == 9, f"Expected nine episode cells, found {len(episode)}")
    require(bool(episode["median_top2_episode_gain_vs_anchor"].gt(0).all()), "All nine median episode gains must be positive")

    medians = {
        f"{event}-H{int(horizon)}": float(value)
        for (event, horizon), value in cell.items()
    }
    return {
        "status": "passed",
        "final_gauges": 981,
        "reporting_groups": 10,
        "primary_units": 450,
        "primary_nonnegative_units": nonnegative,
        "positive_cell_medians": int((cell > 0).sum()),
        "median_ap_gain_by_cell": medians,
        "nested_nonnegative_fold_medians": 90,
        "nested_strictly_positive_fold_medians": 81,
        "nested_nonnegative_seed_units": nested_nonnegative_units,
        "source_block_positive_lower_intervals": 9,
        "postlock_nonnegative_units": 45,
        "postlock_gauges": 33,
        "positive_median_episode_gain_cells": 9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the archived primary DALA-Hydro results.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the derived supporting_tables directory.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    report = build_report(args.data_dir.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
