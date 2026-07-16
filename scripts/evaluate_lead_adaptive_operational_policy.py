from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
BASE = "score_met_static_density"
STATIC = "score_met_static"
BLEND = "score_external_frontier_blend25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", default="1,5,7")
    parser.add_argument("--event-quantile", type=float, default=0.98)
    parser.add_argument("--eval-protocols", default="all,cal20")
    parser.add_argument("--slope", type=float, default=8.0)
    parser.add_argument("--center", type=float, default=-0.5)
    parser.add_argument("--long-lead-scale", type=float, default=0.6)
    parser.add_argument("--model-root", type=Path, default=ROOT / "outputs" / "pilot")
    parser.add_argument("--external-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def qtag(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


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
    if batch not in cache:
        cache[batch] = pd.read_csv(
            external_root / batch / "external_holdout_events.csv.gz",
            usecols=["q_current", "event_threshold", "p_mean"],
            low_memory=False,
        )
    return cache[batch]


def frozen_weight(scores: pd.DataFrame, attrs: pd.DataFrame) -> np.ndarray:
    frontier = scores["external_frontier_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    p_mean = pd.to_numeric(attrs["p_mean"], errors="coerce")
    source_p = p_mean.groupby([scores["source_archive"], scores["country"]], sort=False).transform("median").to_numpy(float)
    hyperwet_supported = (~frontier) & np.isfinite(source_p) & (source_p > 6.0)
    return np.where(hyperwet_supported, 0.0, np.where(frontier, 0.5, 0.35))


def positive_overlay(anchor: np.ndarray, aux: np.ndarray, weight: np.ndarray | float) -> np.ndarray:
    return anchor + weight * np.maximum(aux - anchor, 0.0)


def policy_scores(
    horizon: int,
    scores: pd.DataFrame,
    attrs: pd.DataFrame,
    *,
    slope: float,
    center: float,
    long_lead_scale: float,
) -> dict[str, np.ndarray]:
    base = scores[BASE].to_numpy(dtype=float)
    static = scores[STATIC].to_numpy(dtype=float)
    blend = scores[BLEND].to_numpy(dtype=float)
    posmean = base + 0.5 * np.maximum(static - base, 0.0) + 0.5 * np.maximum(blend - base, 0.0)

    q = pd.to_numeric(attrs["q_current"], errors="coerce").to_numpy(dtype=float)
    thr = pd.to_numeric(attrs["event_threshold"], errors="coerce").replace(0.0, np.nan).to_numpy(dtype=float)
    q_ratio = np.log1p(np.clip(q, 0.0, None)) - np.log1p(np.clip(thr, 0.0, None))
    aux = sigmoid(slope * (q_ratio - center))
    fw = frozen_weight(scores, attrs)

    long_scale = long_lead_scale if horizon >= 7 else 1.0
    direct_weight = 1.0 if horizon <= 1 else fw * long_scale
    return {
        "base": base,
        "positive_blend_static_mean": posmean,
        "op_posmean_frozen": positive_overlay(posmean, aux, fw),
        "op_posmean_horizon_shrink": positive_overlay(posmean, aux, fw * long_scale),
        "op_lead_adaptive_direct": positive_overlay(posmean, aux, direct_weight),
    }


def metric_rows_for_scores(
    *,
    horizon: int,
    protocol: str,
    batch: str,
    seed: int,
    y: np.ndarray,
    mask: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    base = candidates["base"][mask]
    yy = y[mask]
    base_ap = safe_ap(yy, base)
    base_hits, base_precision = top_hits(yy, base)
    rows = []
    for name, score in candidates.items():
        ss = score[mask]
        hits, precision = top_hits(yy, ss)
        current_ap = safe_ap(yy, ss)
        rows.append(
            {
                "horizon": horizon,
                "eval_protocol": protocol,
                "batch": batch,
                "seed": seed,
                "score": name,
                "n": int(np.isfinite(ss).sum()),
                "events": int(yy[np.isfinite(ss)].sum()),
                "ap": current_ap,
                "base_ap": base_ap,
                "ap_gain_vs_base": current_ap - base_ap,
                "p020_hits": hits,
                "base_p020_hits": base_hits,
                "p020_hits_gain_vs_base": hits - base_hits,
                "p020_precision": precision,
                "base_p020_precision": base_precision,
                "p020_precision_gain_vs_base": precision - base_precision,
            }
        )
    return rows


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_summary = (
        rows.groupby(["horizon", "eval_protocol", "score"], sort=False)
        .agg(
            runs=("ap_gain_vs_base", "size"),
            batches=("batch", "nunique"),
            median_ap=("ap", "median"),
            median_ap_gain_vs_base=("ap_gain_vs_base", "median"),
            min_ap_gain_vs_base=("ap_gain_vs_base", "min"),
            fraction_nonnegative_ap_gain=("ap_gain_vs_base", lambda x: float((x >= -1e-12).mean())),
            median_p020_precision_gain_vs_base=("p020_precision_gain_vs_base", "median"),
            median_p020_hits_gain_vs_base=("p020_hits_gain_vs_base", "median"),
        )
        .reset_index()
        .sort_values(["horizon", "eval_protocol", "median_ap_gain_vs_base"], ascending=[True, True, False])
    )
    batch_summary = (
        rows.groupby(["horizon", "eval_protocol", "score", "batch"], sort=False)
        .agg(
            runs=("ap_gain_vs_base", "size"),
            median_ap_gain_vs_base=("ap_gain_vs_base", "median"),
            min_ap_gain_vs_base=("ap_gain_vs_base", "min"),
            fraction_nonnegative_ap_gain=("ap_gain_vs_base", lambda x: float((x >= -1e-12).mean())),
            median_p020_precision_gain_vs_base=("p020_precision_gain_vs_base", "median"),
            median_p020_hits_gain_vs_base=("p020_hits_gain_vs_base", "median"),
        )
        .reset_index()
        .sort_values(
            ["horizon", "eval_protocol", "score", "median_ap_gain_vs_base"],
            ascending=[True, True, True, False],
        )
    )
    return score_summary, batch_summary


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 120) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, columns].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    lines = [header, sep]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    metrics: pd.DataFrame,
    score_summary: pd.DataFrame,
    batch_summary: pd.DataFrame,
    event_tag: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "lead_adaptive_run_metrics.csv", index=False)
    score_summary.to_csv(out_dir / "lead_adaptive_score_summary.csv", index=False)
    batch_summary.to_csv(out_dir / "lead_adaptive_batch_summary.csv", index=False)
    lines = [
        f"# Lead-Adaptive Operational Policy {event_tag.upper()}",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Score Summary",
        "",
        markdown_table(
            score_summary,
            [
                "horizon",
                "eval_protocol",
                "score",
                "runs",
                "batches",
                "median_ap_gain_vs_base",
                "min_ap_gain_vs_base",
                "fraction_nonnegative_ap_gain",
                "median_p020_precision_gain_vs_base",
                "median_p020_hits_gain_vs_base",
            ],
        ),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    event_tag = qtag(args.event_quantile)
    if args.out_dir is None:
        args.out_dir = ROOT / "outputs" / "pilot" / f"lead_adaptive_operational_policy_{event_tag}"
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    protocols = [x.strip() for x in args.eval_protocols.split(",") if x.strip()]
    attr_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        root = args.model_root / f"model_horizon_stress_{event_tag}h{horizon}"
        paths = sorted(root.glob(f"external_holdout_events_{event_tag}h{horizon}_*/*/external_holdout_scores.csv.gz"))
        for idx, path in enumerate(paths, start=1):
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
            candidates = policy_scores(
                horizon,
                scores,
                attrs,
                slope=args.slope,
                center=args.center,
                long_lead_scale=args.long_lead_scale,
            )
            y = scores["event_label"].to_numpy(dtype=np.int8)
            masks = {"all": np.ones(len(scores), dtype=bool)}
            if "cal20" in protocols:
                masks["cal20"] = cal20_mask(scores)
            for protocol in protocols:
                rows.extend(
                    metric_rows_for_scores(
                        horizon=horizon,
                        protocol=protocol,
                        batch=batch,
                        seed=seed,
                        y=y,
                        mask=masks[protocol],
                        candidates=candidates,
                    )
                )
            if idx % 10 == 0:
                print(f"H{horizon}: processed {idx}/{len(paths)}")
    metrics = pd.DataFrame(rows)
    score_summary, batch_summary = summarize(metrics)
    write_outputs(args.out_dir, metrics, score_summary, batch_summary, event_tag)
    print((args.out_dir / "summary.md").resolve())


if __name__ == "__main__":
    main()
