from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def _add_paths() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    return root


ROOT = _add_paths()

from dememory.config import load_config  # noqa: E402
from dememory.timeseries import make_event_features, timeseries_member  # noqa: E402
from train_dahem_event_v1 import compute_analog_features, fit_standardizer  # noqa: E402


QUARTILE_ORDER = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]

MET_FEATURES = [
    "p_1d",
    "p_7d",
    "p_30d",
    "p_future_h",
    "pet_7d",
    "pet_30d",
    "pet_future_h",
    "t_7d",
    "t_future_h",
    "swe_current",
    "swe_7d",
    "soil1_current",
    "soil2_current",
    "month_sin",
    "month_cos",
    "gauge_lat",
    "gauge_lon",
    "area_log10",
    "p_mean",
    "aridity",
    "frac_snow",
    "seasonality",
    "high_prec_freq",
    "low_prec_freq",
]


def gauge_key(row: pd.Series) -> str:
    return f"{row['source_archive']}::{row['gauge_id']}"


def read_member(
    zips: dict[str, zipfile.ZipFile],
    members: dict[str, set[str]],
    row: pd.Series,
) -> pd.DataFrame | None:
    archive_key, member = timeseries_member(row)
    if member not in members[archive_key]:
        return None
    with zips[archive_key].open(member) as f:
        return pd.read_csv(f)


def sample_training_events(
    events: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    high_flow_fraction: float,
    heavy_precip_fraction: float,
) -> pd.DataFrame:
    if len(events) == 0:
        return events
    take = min(n, len(events))
    parts = []
    used: set[int] = set()
    high_n = int(round(take * high_flow_fraction))
    if high_n > 0:
        high_pool = events[events["event_label"] == 1]
        high_take = min(high_n, len(high_pool))
        if high_take > 0:
            sample = high_pool.sample(n=high_take, random_state=int(rng.integers(0, 2**31 - 1)))
            parts.append(sample)
            used.update(sample.index.tolist())
    precip_n = int(round(take * heavy_precip_fraction))
    if precip_n > 0:
        pool = events.drop(index=list(used), errors="ignore")
        if len(pool) > 0:
            threshold = pool["p_7d"].quantile(0.90)
            pool = pool[pool["p_7d"] >= threshold]
            precip_take = min(precip_n, len(pool))
            if precip_take > 0:
                sample = pool.sample(
                    n=precip_take,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                parts.append(sample)
                used.update(sample.index.tolist())
    remaining = take - sum(len(p) for p in parts)
    if remaining > 0:
        pool = events.drop(index=list(used), errors="ignore")
        if len(pool) > 0:
            parts.append(
                pool.sample(
                    n=min(remaining, len(pool)),
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
            )
    out = pd.concat(parts, ignore_index=False) if parts else events.sample(n=take)
    return out.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))


def build_horizon_events(
    train_gauges: pd.DataFrame,
    test_gauges: pd.DataFrame,
    cfg: dict,
    horizon: int,
    out_dir: Path,
    force: bool,
    event_quantile: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = out_dir / "train_sample_events.csv.gz"
    test_path = out_dir / "full_test_events.csv.gz"
    threshold_path = out_dir / "reference_thresholds.csv"
    if train_path.exists() and test_path.exists() and threshold_path.exists() and not force:
        return (
            pd.read_csv(train_path, parse_dates=["date"]),
            pd.read_csv(test_path, parse_dates=["date"]),
            pd.read_csv(threshold_path),
        )

    transfer_cfg = cfg["transfer_v2"]
    rng = np.random.default_rng(int(cfg["random_seed"]) + horizon * 101)
    zips = {
        "caravan": zipfile.ZipFile(cfg["caravan_zip"]),
        "grdc": zipfile.ZipFile(cfg["grdc_caravan_zip"]),
    }
    members = {key: set(z.namelist()) for key, z in zips.items()}
    thresholds = []
    train_parts = []
    test_parts = []
    all_gauges = pd.concat(
        [
            train_gauges.assign(split_role="train"),
            test_gauges.assign(split_role="test"),
        ],
        ignore_index=True,
        sort=False,
    )
    try:
        for _, row in tqdm(all_gauges.iterrows(), total=len(all_gauges), desc=f"h{horizon} events"):
            key = row["gauge_key"] if "gauge_key" in row else gauge_key(row)
            try:
                ts = read_member(zips, members, row)
                if ts is None:
                    thresholds.append({"gauge_key": key, "status": "missing_timeseries"})
                    continue
                events = make_event_features(ts, row, horizon=horizon)
                if events.empty:
                    thresholds.append({"gauge_key": key, "status": "empty_events"})
                    continue
                events = events.replace([np.inf, -np.inf], np.nan)
                events = events.dropna(subset=MET_FEATURES + ["target_streamflow", "target_log1p"])
                ref = events[
                    (events["date"] >= pd.Timestamp(transfer_cfg["train_start"]))
                    & (events["date"] <= pd.Timestamp(transfer_cfg["train_end"]))
                ].copy()
                if len(ref) < 500:
                    thresholds.append(
                        {"gauge_key": key, "status": "too_few_reference_events", "n_reference": len(ref)}
                    )
                    continue
                q98 = float(ref["target_streamflow"].quantile(event_quantile))
                thresholds.append(
                    {
                        "gauge_key": key,
                        "gauge_id": row["gauge_id"],
                        "source_archive": row["source_archive"],
                        "density_quartile": row["density_quartile"],
                        "split_role": row["split_role"],
                        "reference_q98": q98,
                        "reference_quantile": float(event_quantile),
                        "n_reference": int(len(ref)),
                        "status": "ok",
                    }
                )
                events = events.copy()
                events["gauge_key"] = key
                events["event_threshold"] = q98
                events["event_label"] = (events["target_streamflow"] >= q98).astype(float)
                events["event_threshold_log"] = np.log1p(events["event_threshold"])
                if row["split_role"] == "train":
                    train_window = events[
                        (events["date"] >= pd.Timestamp(transfer_cfg["train_start"]))
                        & (events["date"] <= pd.Timestamp(transfer_cfg["train_end"]))
                    ].copy()
                    sampled = sample_training_events(
                        train_window,
                        n=int(transfer_cfg["train_samples_per_gauge"]),
                        rng=rng,
                        high_flow_fraction=float(transfer_cfg.get("train_high_flow_fraction", 0.35)),
                        heavy_precip_fraction=float(transfer_cfg.get("train_heavy_precip_fraction", 0.15)),
                    )
                    if len(sampled) > 0:
                        train_parts.append(sampled)
                else:
                    test = events[
                        (events["date"] >= pd.Timestamp(transfer_cfg["test_start"]))
                        & (events["date"] <= pd.Timestamp(transfer_cfg["test_end"]))
                    ].copy()
                    if len(test) > 0:
                        test["event_q98"] = test["event_label"].astype(bool)
                        test_parts.append(test)
            except Exception as exc:
                thresholds.append({"gauge_key": key, "status": f"error:{repr(exc)}"})
    finally:
        for z in zips.values():
            z.close()

    if not train_parts or not test_parts:
        raise RuntimeError("No horizon events were built.")
    train_events = pd.concat(train_parts, ignore_index=True, sort=False)
    test_events = pd.concat(test_parts, ignore_index=True, sort=False)
    threshold_df = pd.DataFrame(thresholds)
    train_events.to_csv(train_path, index=False)
    test_events.to_csv(test_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    return train_events, test_events, threshold_df


def split_three_way(df: pd.DataFrame, seed: int) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    memory: set[str] = set()
    cal_train: set[str] = set()
    cal_val: set[str] = set()
    gauges = df[["gauge_key", "density_quartile"]].drop_duplicates()
    for _, group in gauges.groupby("density_quartile"):
        keys = group["gauge_key"].to_numpy()
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, int(round(n * 0.20)))
        n_cal = max(1, int(round(n * 0.20)))
        cal_val.update(keys[:n_val].tolist())
        cal_train.update(keys[n_val : n_val + n_cal].tolist())
        memory.update(keys[n_val + n_cal :].tolist())
    return memory, cal_train, cal_val


def select_memory_bank(df: pd.DataFrame, max_events: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_events:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    positive = df[df["event_label"] == 1.0]
    hard_neg = df[
        (df["event_label"] == 0.0)
        & (df["p_7d"] >= df["p_7d"].quantile(0.88))
    ]
    selected = pd.concat([positive, hard_neg], ignore_index=False).drop_duplicates()
    remaining = max_events - len(selected)
    if remaining > 0:
        pool = df.drop(index=selected.index, errors="ignore")
        density = pd.to_numeric(pool["density_percentile"], errors="coerce").fillna(0.5)
        weights = 1.0 + 1.8 * (1.0 - density.clip(0.0, 1.0).to_numpy(dtype=float))
        weights = weights / weights.sum()
        chosen = rng.choice(pool.index.to_numpy(), size=min(remaining, len(pool)), replace=False, p=weights)
        selected = pd.concat([selected, pool.loc[chosen]], ignore_index=False)
    if len(selected) > max_events:
        density = pd.to_numeric(selected["density_percentile"], errors="coerce").fillna(0.5)
        weights = 1.0 + 8.0 * selected["event_label"].to_numpy(dtype=float)
        weights += 1.5 * (1.0 - density.clip(0.0, 1.0).to_numpy(dtype=float))
        weights = weights / weights.sum()
        chosen = rng.choice(selected.index.to_numpy(), size=max_events, replace=False, p=weights)
        selected = selected.loc[chosen]
    return selected.reset_index(drop=True)


def add_analog(
    cal_train: pd.DataFrame,
    cal_val: pd.DataFrame,
    test: pd.DataFrame,
    memory: pd.DataFrame,
    out_dir: Path,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = out_dir / "cal_train_analog_features.csv.gz"
    val_path = out_dir / "cal_val_analog_features.csv.gz"
    test_path = out_dir / "full_test_analog_features.csv.gz"
    retrieval_cols = MET_FEATURES + ["event_threshold_log", "density_percentile"]
    scaler = fit_standardizer(memory, retrieval_cols)
    if train_path.exists() and val_path.exists() and test_path.exists() and not force:
        train_analog = pd.read_csv(train_path)
        val_analog = pd.read_csv(val_path)
        test_analog = pd.read_csv(test_path)
    else:
        train_analog = compute_analog_features(
            cal_train,
            memory,
            retrieval_cols,
            scaler,
            k=64,
            exclude_same_gauge=False,
            label="horizon-cal-train",
        )
        val_analog = compute_analog_features(
            cal_val,
            memory,
            retrieval_cols,
            scaler,
            k=64,
            exclude_same_gauge=False,
            label="horizon-cal-val",
        )
        test_analog = compute_analog_features(
            test,
            memory,
            retrieval_cols,
            scaler,
            k=64,
            exclude_same_gauge=False,
            label="horizon-full-test",
        )
        train_analog.to_csv(train_path, index=False)
        val_analog.to_csv(val_path, index=False)
        test_analog.to_csv(test_path, index=False)
    return (
        pd.concat([cal_train.reset_index(drop=True), train_analog.reset_index(drop=True)], axis=1),
        pd.concat([cal_val.reset_index(drop=True), val_analog.reset_index(drop=True)], axis=1),
        pd.concat([test.reset_index(drop=True), test_analog.reset_index(drop=True)], axis=1),
    )


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = MET_FEATURES + [
        "density_percentile",
        "event_threshold",
        "event_threshold_log",
        "analog_event_rate",
        "analog_density_event_rate",
        "analog_log_mean",
        "analog_log_q85",
        "analog_positive_log_mean",
        "analog_mean_distance",
        "analog_min_distance",
        "analog_positive_count",
    ]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["label"] = out["event_label"].astype(bool).astype(int)
    out["low_support"] = 1.0 - out["density_percentile"].clip(0.0, 1.0)
    out["analog_log_ratio"] = out["analog_log_mean"] - np.log1p(out["event_threshold"])
    out["analog_q85_ratio"] = out["analog_log_q85"] - np.log1p(out["event_threshold"])
    out["analog_pos_ratio"] = out["analog_positive_log_mean"] - np.log1p(out["event_threshold"])
    out["analog_support"] = out["analog_positive_count"] / 64.0
    out["distance_support"] = out["analog_mean_distance"] / (
        0.1 + out["density_percentile"].clip(0.0, 1.0)
    )
    koppen_major = out.get("koppen_major", pd.Series("", index=out.index)).astype(str)
    for major in ["A", "B", "C", "D", "E"]:
        out[f"koppen_{major}"] = (koppen_major == major).astype(int)
    return out


def binary_stats(obs: np.ndarray, pred: np.ndarray) -> dict:
    obs = np.asarray(obs, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((obs & pred).sum())
    fp = int((~obs & pred).sum())
    fn = int((obs & ~pred).sum())
    tn = int((~obs & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "csi": csi,
        "pred_rate": float(pred.mean()) if len(pred) else 0.0,
        "event_rate": float(obs.mean()) if len(obs) else 0.0,
    }


def tune_threshold(df: pd.DataFrame, score_col: str) -> float:
    y = df["label"].to_numpy(dtype=bool)
    scores = df[score_col].to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        return 0.5
    thresholds = np.unique(np.quantile(scores[finite], np.linspace(0.001, 0.999, 500)))
    best_t = float(np.nanmedian(scores[finite]))
    best_score = -np.inf
    for threshold in thresholds:
        stats = binary_stats(y, scores >= threshold)
        if stats["f1"] > best_score:
            best_score = stats["f1"]
            best_t = float(threshold)
    return best_t


def apply_thresholds(
    cal_val: pd.DataFrame,
    test: pd.DataFrame,
    score_col: str,
    strategy: str,
) -> tuple[np.ndarray, dict]:
    if strategy == "global":
        threshold = tune_threshold(cal_val, score_col)
        return test[score_col].to_numpy(dtype=float) >= threshold, {"global": threshold}
    pred = np.zeros(len(test), dtype=bool)
    thresholds: dict[str, float] = {}
    for quartile, subset in cal_val.groupby("density_quartile"):
        threshold = tune_threshold(subset, score_col)
        thresholds[str(quartile)] = threshold
        mask = test["density_quartile"].astype(str).to_numpy() == str(quartile)
        pred[mask] = test.loc[mask, score_col].to_numpy(dtype=float) >= threshold
    return pred, thresholds


def evaluate_models(
    cal_train: pd.DataFrame,
    cal_val: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def unique_existing(columns: list[str]) -> list[str]:
        out = []
        seen = set()
        for col in columns:
            if col in seen or col not in cal_train.columns:
                continue
            seen.add(col)
            out.append(col)
        return out

    static = [
        "event_threshold_log",
        "area_log10",
        "p_mean",
        "aridity",
        "frac_snow",
        "seasonality",
        "high_prec_freq",
        "low_prec_freq",
        "koppen_A",
        "koppen_B",
        "koppen_C",
        "koppen_D",
        "koppen_E",
    ]
    density = ["density_percentile", "low_support"]
    analog = [
        "analog_event_rate",
        "analog_density_event_rate",
        "analog_log_mean",
        "analog_log_q85",
        "analog_positive_log_mean",
        "analog_mean_distance",
        "analog_min_distance",
        "analog_positive_count",
        "analog_log_ratio",
        "analog_q85_ratio",
        "analog_pos_ratio",
        "analog_support",
        "distance_support",
    ]
    feature_sets = {
        "met_static": MET_FEATURES + static,
        "met_static_density": MET_FEATURES + static + density,
        "analog_only": static + density + analog,
        "met_analog": MET_FEATURES + static + density + analog,
    }
    rows = []
    y = cal_train["label"].to_numpy(dtype=int)
    pos_weight = min(50.0, (len(y) - y.sum()) / (y.sum() + 1e-9)) * 0.65
    weights = 1.0 + y * pos_weight + 2.0 * cal_train["low_support"].to_numpy(dtype=float)
    for model_name, columns in feature_sets.items():
        columns = unique_existing(columns)
        for frame in [cal_train, cal_val, test]:
            frame.loc[:, columns] = frame[columns].replace([np.inf, -np.inf], np.nan)
        med = cal_train[columns].median(numeric_only=True)
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            min_samples_leaf=80,
            random_state=seed,
        )
        model.fit(cal_train[columns].fillna(med), y, sample_weight=weights)
        score_col = f"score_{model_name}"
        cal_val[score_col] = model.predict_proba(cal_val[columns].fillna(med))[:, 1]
        test[score_col] = model.predict_proba(test[columns].fillna(med))[:, 1]
        ap_all = float(average_precision_score(test["label"], test[score_col]))
        q1 = test[test["density_quartile"] == "Q1_lowest"]
        ap_q1 = float(average_precision_score(q1["label"], q1[score_col]))
        for strategy in ["global", "density"]:
            pred, thresholds = apply_thresholds(cal_val, test, score_col, strategy)
            for quartile, idx in test.groupby("density_quartile").groups.items():
                stats = binary_stats(test.loc[idx, "label"], pred[idx])
                stats.update(
                    {
                        "model": model_name,
                        "strategy": strategy,
                        "density_quartile": quartile,
                        "n_features": len(columns),
                        "ap_all": ap_all,
                        "ap_q1": ap_q1,
                        "thresholds": json.dumps(thresholds),
                    }
                )
                rows.append(stats)
    event_level = pd.DataFrame(rows)
    rank_rows = []
    for (model, strategy), group in event_level.groupby(["model", "strategy"]):
        by_q = {row["density_quartile"]: row for _, row in group.iterrows()}
        rank_rows.append(
            {
                "model": model,
                "strategy": strategy,
                "n_features": int(group["n_features"].iloc[0]),
                "ap_all": float(group["ap_all"].iloc[0]),
                "ap_q1": float(group["ap_q1"].iloc[0]),
                "q1_precision": by_q["Q1_lowest"]["precision"],
                "q1_recall": by_q["Q1_lowest"]["recall"],
                "q1_f1": by_q["Q1_lowest"]["f1"],
                "q1_csi": by_q["Q1_lowest"]["csi"],
                "q2_f1": by_q["Q2"]["f1"],
                "q3_f1": by_q["Q3"]["f1"],
                "q4_f1": by_q["Q4_highest"]["f1"],
                "mean_f1": float(np.mean([by_q[q]["f1"] for q in QUARTILE_ORDER if q in by_q])),
            }
        )
    rank = pd.DataFrame(rank_rows).sort_values(["mean_f1", "q1_f1"], ascending=False)
    return event_level, rank


def write_report(
    rank: pd.DataFrame,
    event_level: pd.DataFrame,
    threshold_df: pd.DataFrame,
    out_path: Path,
    horizon: int,
    sizes: dict,
) -> None:
    best = rank.iloc[0]
    baseline = rank[
        (rank["model"] == "met_static_density")
        & (rank["strategy"] == best["strategy"])
    ]
    baseline_row = baseline.iloc[0] if len(baseline) else rank[rank["model"] == "met_static_density"].iloc[0]
    lines = [
        f"# Horizon-{horizon} Q98 transfer gate",
        "",
        "## Setup",
        "",
        "- Target: rare high-flow event, streamflow exceeding a basin-specific Q98 threshold.",
        "- Q98 threshold: computed from 1981-2008 reference period.",
        "- Test period: complete daily 2013-2020 on held-out basins.",
        "- Current observed streamflow is excluded from model features.",
        "",
        "## Data",
        "",
        f"- Train sample events: {sizes['train_events']:,}.",
        f"- Full test events: {sizes['test_events']:,}.",
        f"- Memory events: {sizes['memory_events']:,}.",
        f"- Test basins: {sizes['test_basins']:,}.",
        f"- Threshold status: {threshold_df['status'].value_counts().to_dict()}.",
        "",
        "## Best Result",
        "",
        f"- Best model: `{best['model']}` / `{best['strategy']}`.",
        f"- Q1 F1: {best['q1_f1']:.3f}; Q1 precision: {best['q1_precision']:.3f}; mean F1: {best['mean_f1']:.3f}.",
        f"- Matched met_static_density baseline Q1 F1: {baseline_row['q1_f1']:.3f}; mean F1: {baseline_row['mean_f1']:.3f}.",
        "",
        "## Interpretation",
        "",
        "This gate tests whether event memory remains useful when the task is no longer dominated by one-day flow persistence. A positive result here is more relevant to the intended public-data transfer story than the operational 1-day nowcast protocol.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pilot_config.json"))
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-events", type=int, default=100000)
    parser.add_argument("--event-quantile", type=float, default=0.98)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / "outputs" / "pilot" / f"horizon_q98_h{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_gauges = pd.read_csv(ROOT / "data" / "processed" / "transfer_v2_pilot" / "selected_train_gauges.csv")
    test_gauges = pd.read_csv(ROOT / "data" / "processed" / "transfer_v2_pilot" / "selected_test_gauges.csv")
    train_events, test_events, threshold_df = build_horizon_events(
        train_gauges,
        test_gauges,
        cfg,
        horizon=args.horizon,
        out_dir=out_dir,
        force=args.force,
        event_quantile=args.event_quantile,
    )

    memory_keys, cal_train_keys, cal_val_keys = split_three_way(
        train_events,
        seed=int(cfg["random_seed"]) + args.horizon * 17,
    )
    memory_source = train_events[train_events["gauge_key"].isin(memory_keys)].reset_index(drop=True)
    cal_train = train_events[train_events["gauge_key"].isin(cal_train_keys)].reset_index(drop=True)
    cal_val = train_events[train_events["gauge_key"].isin(cal_val_keys)].reset_index(drop=True)
    memory = select_memory_bank(
        memory_source,
        max_events=int(args.memory_events),
        seed=int(cfg["random_seed"]) + args.horizon * 19,
    )
    cal_train, cal_val, test_events = add_analog(
        cal_train,
        cal_val,
        test_events,
        memory,
        out_dir,
        force=args.force,
    )
    cal_train = add_model_features(cal_train)
    cal_val = add_model_features(cal_val)
    test_events = add_model_features(test_events)
    event_level, rank = evaluate_models(
        cal_train,
        cal_val,
        test_events,
        seed=int(cfg["random_seed"]) + args.horizon,
    )
    score_cols = [c for c in test_events.columns if c.startswith("score_")]
    score_keep = [
        "date",
        "gauge_key",
        "density_quartile",
        "density_percentile",
        "event_threshold",
        "event_label",
    ] + score_cols
    test_events[score_keep].to_csv(out_dir / "full_test_model_scores.csv.gz", index=False)
    cal_val[score_keep].to_csv(out_dir / "cal_val_model_scores.csv.gz", index=False)
    event_level.to_csv(out_dir / "horizon_q98_event_level.csv", index=False)
    rank.to_csv(out_dir / "horizon_q98_rank.csv", index=False)
    sizes = {
        "train_events": int(len(train_events)),
        "test_events": int(len(test_events)),
        "memory_events": int(len(memory)),
        "test_basins": int(test_events["gauge_key"].nunique()),
    }
    (out_dir / "horizon_q98_summary.json").write_text(
        json.dumps(
            {
                "horizon": args.horizon,
                "sizes": sizes,
                "memory_positive_rate": float(memory["event_label"].mean()),
                "rank": rank.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        rank,
        event_level,
        threshold_df,
        out_dir / "horizon_q98_transfer_report.md",
        horizon=args.horizon,
        sizes=sizes,
    )
    print(rank.to_string(index=False))


if __name__ == "__main__":
    main()
