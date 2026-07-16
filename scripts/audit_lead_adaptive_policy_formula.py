from __future__ import annotations

import argparse
import ast
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
BASE = "score_met_static_density"
STATIC = "score_met_static"
BLEND = "score_external_frontier_blend25"
PRIMARY = "op_lead_adaptive_direct"
TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-quantile", type=float, default=0.98)
    parser.add_argument("--horizons", default="1,5,7")
    parser.add_argument("--policy-dir", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=ROOT / "outputs" / "pilot")
    parser.add_argument("--external-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def qtag(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def parse_horizons(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def function_source(path: Path, function_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return ""


def static_checks() -> list[dict[str, Any]]:
    path = ROOT / "scripts" / "evaluate_lead_adaptive_operational_policy.py"
    score_body = "\n".join(
        [
            function_source(path, "policy_scores"),
            function_source(path, "frozen_weight"),
            function_source(path, "positive_overlay"),
        ]
    )
    forbidden_score_inputs = [
        "event_label",
        "target_streamflow",
        "target_log1p",
        "p_future_h",
        "pet_future_h",
        "t_future_h",
    ]
    found = [name for name in forbidden_score_inputs if name in score_body]
    rows = [
        {
            "check": "lead_adaptive_score_formula_forbidden_inputs",
            "scope": "policy_scores/frozen_weight/positive_overlay",
            "status": "PASS" if not found else "FAIL",
            "detail": "none" if not found else ",".join(found),
        }
    ]
    ts_body = function_source(ROOT / "src" / "dememory" / "timeseries.py", "make_event_features")
    target_ok = 'target_streamflow"] = df[q_col].shift(-horizon)' in ts_body
    current_ok = 'q_current"] = df[q_col]' in ts_body
    rows.append(
        {
            "check": "event_feature_time_definition",
            "scope": "make_event_features",
            "status": "PASS" if target_ok and current_ok else "FAIL",
            "detail": "q_current=q(t); target_streamflow=q(t+horizon)" if target_ok and current_ok else "definition not found",
        }
    )
    return rows


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    if mask.sum() == 0 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(average_precision_score(y[mask], score[mask]))


def top_hits(y: np.ndarray, score: np.ndarray, frac: float = 0.02) -> tuple[int, float]:
    mask = np.isfinite(score)
    if mask.sum() == 0:
        return 0, float("nan")
    yy = y[mask]
    ss = score[mask]
    k = max(1, int(np.ceil(len(yy) * frac)))
    order = np.argsort(-ss, kind="mergesort")
    hits = int(yy[order[:k]].sum())
    return hits, float(hits / k)


def cal20_mask(scores: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    dates = pd.to_datetime(scores["date"], errors="coerce").to_numpy()
    for _, idx in scores.groupby(["source_archive", "gauge_key"], sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        order = np.argsort(dates[idx_arr], kind="mergesort")
        sorted_idx = idx_arr[order]
        n = len(sorted_idx)
        if n <= 1:
            continue
        cut = min(n - 1, max(30, int(np.floor(n * 0.20))))
        mask[sorted_idx[cut:]] = True
    return mask


def load_attrs(batch: str, external_root: Path, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = batch
    if key not in cache:
        cache[key] = pd.read_csv(
            external_root / batch / "external_holdout_events.csv.gz",
            usecols=["q_current", "event_threshold", "p_mean"],
            low_memory=False,
        )
    return cache[key]


def lead_adaptive_score(horizon: int, scores: pd.DataFrame, attrs: pd.DataFrame) -> np.ndarray:
    base = scores[BASE].to_numpy(dtype=float)
    static = scores[STATIC].to_numpy(dtype=float)
    blend = scores[BLEND].to_numpy(dtype=float)
    posmean = base + 0.5 * np.maximum(static - base, 0.0) + 0.5 * np.maximum(blend - base, 0.0)

    q = pd.to_numeric(attrs["q_current"], errors="coerce").to_numpy(dtype=float)
    thr = pd.to_numeric(attrs["event_threshold"], errors="coerce").replace(0.0, np.nan).to_numpy(dtype=float)
    q_ratio = np.log1p(np.clip(q, 0.0, None)) - np.log1p(np.clip(thr, 0.0, None))
    aux = sigmoid(8.0 * (q_ratio + 0.5))

    frontier = scores["external_frontier_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    p_mean = pd.to_numeric(attrs["p_mean"], errors="coerce")
    source_p = p_mean.groupby([scores["source_archive"], scores["country"]], sort=False).transform("median").to_numpy(float)
    hyperwet_supported = (~frontier) & np.isfinite(source_p) & (source_p > 6.0)
    frozen_weight = np.where(hyperwet_supported, 0.0, np.where(frontier, 0.5, 0.35))
    weight = 1.0 if horizon <= 1 else frozen_weight * (0.6 if horizon >= 7 else 1.0)
    return posmean + weight * np.maximum(aux - posmean, 0.0)


def metric_row(
    horizon: int,
    protocol: str,
    batch: str,
    seed: int,
    y: np.ndarray,
    mask: np.ndarray,
    base: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    yy = y[mask]
    bb = base[mask]
    pp = pred[mask]
    base_hits, base_precision = top_hits(yy, bb)
    hits, precision = top_hits(yy, pp)
    base_ap = safe_ap(yy, bb)
    ap = safe_ap(yy, pp)
    return {
        "horizon": horizon,
        "eval_protocol": protocol,
        "batch": batch,
        "seed": seed,
        "score": PRIMARY,
        "n": int(np.isfinite(pp).sum()),
        "events": int(yy[np.isfinite(pp)].sum()),
        "ap": ap,
        "base_ap": base_ap,
        "ap_gain_vs_base": ap - base_ap,
        "p020_hits": hits,
        "base_p020_hits": base_hits,
        "p020_hits_gain_vs_base": hits - base_hits,
        "p020_precision": precision,
        "base_p020_precision": base_precision,
        "p020_precision_gain_vs_base": precision - base_precision,
    }


def recompute_metrics(args: argparse.Namespace, policy_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_tag = qtag(args.event_quantile)
    saved = pd.read_csv(policy_dir / "lead_adaptive_run_metrics.csv")
    saved = saved[saved["score"].eq(PRIMARY)].copy()
    rows: list[dict[str, Any]] = []
    attr_cache: dict[str, pd.DataFrame] = {}
    for horizon in parse_horizons(args.horizons):
        model_root = args.model_root / f"model_horizon_stress_{event_tag}h{horizon}"
        paths = sorted(model_root.glob(f"external_holdout_events_{event_tag}h{horizon}_*/*/external_holdout_scores.csv.gz"))
        for path in paths:
            batch = path.parent.parent.name
            seed = int(path.parent.name.replace("seed_", ""))
            scores = pd.read_csv(
                path,
                usecols=[
                    "date",
                    "gauge_key",
                    "source_archive",
                    "country",
                    "event_label",
                    "external_frontier_flag",
                    BASE,
                    STATIC,
                    BLEND,
                ],
                low_memory=False,
            )
            attrs = load_attrs(batch, args.external_root, attr_cache)
            if len(scores) != len(attrs):
                raise RuntimeError(f"Row mismatch for {path}")
            pred = lead_adaptive_score(horizon, scores, attrs)
            y = scores["event_label"].to_numpy(dtype=np.int8)
            base = scores[BASE].to_numpy(dtype=float)
            masks = {
                "all": np.ones(len(scores), dtype=bool),
                "cal20": cal20_mask(scores),
            }
            for protocol, mask in masks.items():
                rows.append(metric_row(horizon, protocol, batch, seed, y, mask, base, pred))

    recomputed = pd.DataFrame(rows)
    keys = ["horizon", "eval_protocol", "batch", "seed", "score"]
    merged = saved.merge(recomputed, on=keys, suffixes=("_saved", "_recomputed"), validate="one_to_one")
    return recomputed, merged


def formula_checks(merged: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cols = [
        "n",
        "events",
        "ap",
        "base_ap",
        "ap_gain_vs_base",
        "p020_hits",
        "base_p020_hits",
        "p020_hits_gain_vs_base",
        "p020_precision",
        "base_p020_precision",
        "p020_precision_gain_vs_base",
    ]
    for col in cols:
        saved_col = f"{col}_saved"
        rec_col = f"{col}_recomputed"
        diffs = (pd.to_numeric(merged[saved_col], errors="coerce") - pd.to_numeric(merged[rec_col], errors="coerce")).abs()
        max_diff = float(diffs.max()) if len(diffs) else float("nan")
        rows.append(
            {
                "check": f"formula_recompute_{col}",
                "scope": "all_runs",
                "status": "PASS" if max_diff <= TOL else "FAIL",
                "max_abs_diff": f"{max_diff:.15g}",
                "tolerance": f"{TOL:.1e}",
                "detail": f"rows={len(merged)}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    event_tag = qtag(args.event_quantile)
    policy_dir = args.policy_dir
    if policy_dir is None:
        policy_dir = ROOT / "outputs" / "pilot" / f"lead_adaptive_operational_policy_{event_tag}"
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ROOT / "outputs" / "pilot" / f"lead_adaptive_operational_policy_{event_tag}_audit"

    out_dir.mkdir(parents=True, exist_ok=True)
    static = static_checks()
    recomputed, merged = recompute_metrics(args, policy_dir)
    checks = formula_checks(merged)
    recomputed.to_csv(out_dir / "recomputed_run_metrics.csv", index=False)
    merged.to_csv(out_dir / "formula_reproduction_merged.csv", index=False)
    write_csv(out_dir / "static_checks.csv", static)
    write_csv(out_dir / "formula_checks.csv", checks)
    all_checks = static + checks
    failures = [row for row in all_checks if row["status"] != "PASS"]
    pass_count = len(all_checks) - len(failures)
    status = "PASS" if not failures else "FAIL"
    lines = [
        f"# Lead-Adaptive Policy Formula Audit {event_tag.upper()}",
        "",
        f"Status: {status}",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Checks passed: {pass_count}/{len(all_checks)}.",
        "",
        "The audit recomputes `op_lead_adaptive_direct` from stored model anchors, `q_current`, event thresholds, support-frontier flags, source precipitation regime, and the fixed lead-adaptive weight schedule.",
        "",
        "## Scope",
        "",
        f"- Event threshold: `{event_tag.upper()}`",
        f"- Horizons: `{','.join(str(x) for x in parse_horizons(args.horizons))}`",
        f"- Saved policy directory: `{policy_dir}`",
        f"- Model root: `{args.model_root}`",
        f"- External sidecar root: `{args.external_root}`",
        "",
        "## Fixed Formula",
        "",
        "- Anchor: `positive_blend_static_mean = base + 0.5 * max(met_static - base, 0) + 0.5 * max(blend25 - base, 0)`",
        "- High-state signal: `sigmoid(8 * ((log1p(q_current) - log1p(event_threshold)) + 0.5))`",
        "- H1 weight: `1.0`",
        "- H5 weight: frozen support-aware weights `0.5 frontier / 0.35 supported / 0 hyper-wet supported`",
        "- H7 weight: `0.6 x` frozen support-aware weights",
        "",
        "## Output",
        "",
        f"- Static checks: `{out_dir / 'static_checks.csv'}`",
        f"- Formula checks: `{out_dir / 'formula_checks.csv'}`",
        f"- Recomputed metrics: `{out_dir / 'recomputed_run_metrics.csv'}`",
        "",
    ]
    if failures:
        lines.extend(["## Failed Checks", ""])
        for row in failures:
            lines.append(f"- {row['check']} / {row.get('scope', '')}: {row.get('detail', '')}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "summary.md")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
