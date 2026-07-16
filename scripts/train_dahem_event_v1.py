from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def _add_paths() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    return root


ROOT = _add_paths()


QUARTILE_ORDER = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]


class EventRiskNet(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.14),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Standardizer:
    mean: pd.Series
    std: pd.Series

    def transform(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        x = (df[cols] - self.mean) / self.std
        return x.to_numpy(dtype=np.float32)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot access CUDA.")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(1)


def load_feature_columns(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_event_labels(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    out = df.copy()
    key = "gauge_key" if "gauge_key" in out.columns else "gauge_id"
    threshold = out.groupby(key)["target_streamflow"].transform(lambda s: s.quantile(quantile))
    out["event_threshold"] = threshold
    out["event_label"] = (out["target_streamflow"] >= threshold).astype(np.float32)
    return out


def split_fit_val_gauges(df: pd.DataFrame, seed: int, val_fraction: float = 0.20) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    fit_keys: set[str] = set()
    val_keys: set[str] = set()
    gauge_table = df[["gauge_key", "density_quartile"]].drop_duplicates()
    for _, group in gauge_table.groupby("density_quartile"):
        keys = group["gauge_key"].to_numpy()
        rng.shuffle(keys)
        n_val = max(1, int(round(len(keys) * val_fraction)))
        val_keys.update(keys[:n_val].tolist())
        fit_keys.update(keys[n_val:].tolist())
    return fit_keys, val_keys


def select_memory_bank(df: pd.DataFrame, max_events: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_events:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    positive = df[df["event_label"] == 1.0]
    hard_neg = df[
        (df["event_label"] == 0.0)
        & (
            (df["p_7d"] >= df["p_7d"].quantile(0.88))
            | (df["pred_log1p_mlp"] >= df["pred_log1p_mlp"].quantile(0.88))
        )
    ]
    selected = pd.concat([positive, hard_neg], ignore_index=False).drop_duplicates()
    remaining = max_events - len(selected)
    if remaining > 0:
        pool = df.drop(index=selected.index, errors="ignore")
        density = pd.to_numeric(pool["density_percentile"], errors="coerce").fillna(0.5)
        weights = 1.0 + 1.8 * (1.0 - density.clip(0.0, 1.0).to_numpy(dtype=float))
        weights = weights / weights.sum()
        take = min(remaining, len(pool))
        sampled_idx = rng.choice(pool.index.to_numpy(), size=take, replace=False, p=weights)
        selected = pd.concat([selected, pool.loc[sampled_idx]], ignore_index=False)
    if len(selected) > max_events:
        density = pd.to_numeric(selected["density_percentile"], errors="coerce").fillna(0.5)
        weights = 1.0 + 8.0 * selected["event_label"].to_numpy(dtype=float)
        weights += 1.5 * (1.0 - density.clip(0.0, 1.0).to_numpy(dtype=float))
        weights = weights / weights.sum()
        sampled_idx = rng.choice(selected.index.to_numpy(), size=max_events, replace=False, p=weights)
        selected = selected.loc[sampled_idx]
    return selected.reset_index(drop=True)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return float(np.nanmean(values))
    return float(values[np.searchsorted(cdf, q * cdf[-1], side="left")])


def fit_standardizer(df: pd.DataFrame, cols: list[str]) -> Standardizer:
    mean = df[cols].mean()
    std = df[cols].std(ddof=0).replace(0, 1.0)
    return Standardizer(mean=mean, std=std)


def compute_analog_features(
    query: pd.DataFrame,
    memory: pd.DataFrame,
    retrieval_cols: list[str],
    standardizer: Standardizer,
    k: int,
    exclude_same_gauge: bool,
    label: str,
) -> pd.DataFrame:
    print(f"computing analog features: {label}, query={len(query)}, memory={len(memory)}", flush=True)
    mem_x = standardizer.transform(memory, retrieval_cols)
    query_x = standardizer.transform(query, retrieval_cols)
    n_neighbors = min(max(k * 3, k), len(memory))
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_index.fit(mem_x)

    mem_label = memory["event_label"].to_numpy(dtype=float)
    mem_y = memory["target_log1p"].to_numpy(dtype=float)
    mem_density = pd.to_numeric(memory["density_percentile"], errors="coerce").fillna(0.5)
    mem_density = mem_density.clip(0.0, 1.0).to_numpy(dtype=float)
    mem_gauge = memory["gauge_key"].astype(str).to_numpy()
    query_gauge = query["gauge_key"].astype(str).to_numpy()

    chunks = []
    chunk_size = 12000
    for start in range(0, len(query), chunk_size):
        end = min(start + chunk_size, len(query))
        distances, indices = nn_index.kneighbors(query_x[start:end], return_distance=True)
        rows = []
        for i in range(end - start):
            idx = indices[i]
            dist = distances[i]
            if exclude_same_gauge:
                keep = mem_gauge[idx] != query_gauge[start + i]
                idx = idx[keep]
                dist = dist[keep]
            if len(idx) == 0:
                idx = indices[i]
                dist = distances[i]
            idx = idx[:k]
            dist = dist[:k]
            scale = np.median(dist) + 1e-6
            w = np.exp(-dist / scale)
            density_w = w * (1.0 / (0.25 + mem_density[idx]))
            labels = mem_label[idx]
            y = mem_y[idx]
            if labels.sum() > 0:
                pos_w = w * labels
                pos_log = float(np.sum(pos_w * y) / np.sum(pos_w))
            else:
                pos_log = float(np.sum(w * y) / np.sum(w))
            rows.append(
                {
                    "analog_event_rate": float(np.sum(w * labels) / np.sum(w)),
                    "analog_density_event_rate": float(np.sum(density_w * labels) / np.sum(density_w)),
                    "analog_log_mean": float(np.sum(w * y) / np.sum(w)),
                    "analog_log_q85": weighted_quantile(y, w, 0.85),
                    "analog_positive_log_mean": pos_log,
                    "analog_mean_distance": float(np.mean(dist)),
                    "analog_min_distance": float(np.min(dist)),
                    "analog_positive_count": float(labels.sum()),
                }
            )
        chunks.append(pd.DataFrame(rows, index=query.index[start:end]))
    return pd.concat(chunks).sort_index()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.72, gamma: float = 2.0) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * bce


def train_classifier(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    seed: int,
) -> tuple[EventRiskNet, Standardizer, dict]:
    scaler = fit_standardizer(fit_df, feature_cols)
    x_fit = scaler.transform(fit_df, feature_cols)
    x_val = scaler.transform(val_df, feature_cols)
    y_fit = fit_df["event_label"].to_numpy(dtype=np.float32)
    y_val = val_df["event_label"].to_numpy(dtype=np.float32)

    density = pd.to_numeric(fit_df["density_percentile"], errors="coerce").fillna(0.5)
    low_support = 1.0 - density.clip(0.0, 1.0).to_numpy(dtype=float)
    analog_rate = fit_df["analog_density_event_rate"].to_numpy(dtype=float)
    hard_negative = ((y_fit == 0) & (analog_rate >= np.quantile(analog_rate, 0.85))).astype(float)
    weights = 1.0 + 10.0 * y_fit + 2.0 * low_support + 3.0 * hard_negative
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights.astype(np.float64)),
        num_samples=len(weights),
        replacement=True,
    )
    ds = TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit))
    loader = DataLoader(
        ds,
        batch_size=4096,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )

    model = EventRiskNet(n_features=len(feature_cols), hidden_dim=192).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    history = []
    best_state = None
    best_score = -np.inf
    bad = 0

    for epoch in range(1, 61):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = focal_loss(logits, yb).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_x)).detach().cpu().numpy()
        val_score = average_precision(y_val, val_prob)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_average_precision": float(val_score),
            }
        )
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch {epoch:03d} train_loss={history[-1]['train_loss']:.4f} "
                f"val_ap={val_score:.4f} best_ap={best_score:.4f}",
                flush=True,
            )
        if bad >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, {"history": history, "best_val_average_precision": float(best_score)}


def predict_prob(model: EventRiskNet, scaler: Standardizer, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    device = next(model.parameters()).device
    x = scaler.transform(df, cols)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 16384):
            xb = torch.from_numpy(x[start : start + 16384]).to(device)
            out.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(out)


def average_precision(y_true: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(-prob)
    y = y_true[order]
    positives = y.sum()
    if positives <= 0:
        return 0.0
    precision = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float(np.sum(precision * y) / positives)


def binary_stats(obs: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> dict:
    obs = np.asarray(obs, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((pred & obs).sum())
    fp = int((pred & ~obs).sum())
    fn = int((~pred & obs).sum())
    tn = int((~pred & ~obs).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and (precision + recall) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    far = fp / (tp + fp) if (tp + fp) else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "csi": csi,
        "false_alarm_ratio": far,
        "event_rate": float(obs.mean()),
        "pred_rate": float(pred.mean()),
    }


def tune_thresholds(df: pd.DataFrame, prob_col: str, min_recall: float = 0.15) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(0.02, 0.98, 97)
    best_global = 0.5
    best_score = -np.inf
    for t in thresholds:
        stats = binary_stats(df["event_label"], df[prob_col] >= t)
        score = stats["f1"]
        if stats["recall"] < min_recall:
            score *= 0.75
        if score > best_score:
            best_score = score
            best_global = float(t)

    group_thresholds = {}
    for q, group in df.groupby("density_quartile"):
        best_t = best_global
        best_score = -np.inf
        for t in thresholds:
            stats = binary_stats(group["event_label"], group[prob_col] >= t)
            score = stats["f1"]
            if stats["recall"] < min_recall:
                score *= 0.75
            if score > best_score:
                best_score = score
                best_t = float(t)
        group_thresholds[str(q)] = best_t
    return best_global, group_thresholds


def summarize_models(df: pd.DataFrame, specs: list[tuple[str, pd.Series | np.ndarray]]) -> pd.DataFrame:
    rows = []
    for model_name, pred in specs:
        pred = np.asarray(pred, dtype=bool)
        for q, idx in df.groupby("density_quartile").groups.items():
            group = df.loc[idx]
            stats = binary_stats(group["event_label"], pred[group.index.to_numpy()])
            stats.update({"model": model_name, "density_quartile": q})
            rows.append(stats)
    out = pd.DataFrame(rows)
    out["density_quartile"] = pd.Categorical(out["density_quartile"], QUARTILE_ORDER, ordered=True)
    return out.sort_values(["density_quartile", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", default=str(ROOT / "data" / "processed" / "transfer_v2_pilot"))
    parser.add_argument("--baseline-dir", default=str(ROOT / "outputs" / "pilot" / "transfer_v2_baseline"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "pilot" / "dahem_event_v1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_dir = Path(args.events_dir)
    baseline_dir = Path(args.baseline_dir)
    base_features = load_feature_columns(events_dir / "feature_columns.json")

    train = pd.read_csv(baseline_dir / "train_predictions.csv.gz", parse_dates=["date"])
    test = pd.read_csv(baseline_dir / "test_predictions.csv.gz", parse_dates=["date"])
    for df in [train, test]:
        for col in base_features + ["pred_log1p_mlp", "target_log1p", "target_streamflow"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["gauge_key"] = df.get("gauge_key", df["source_archive"].astype(str) + "::" + df["gauge_id"].astype(str))
    train = add_event_labels(train.dropna(subset=base_features + ["pred_log1p_mlp"]), args.event_quantile)
    test = add_event_labels(test.dropna(subset=base_features + ["pred_log1p_mlp"]), args.event_quantile)

    fit_keys, val_keys = split_fit_val_gauges(train, seed=args.seed)
    fit = train[train["gauge_key"].isin(fit_keys)].copy().reset_index(drop=True)
    val = train[train["gauge_key"].isin(val_keys)].copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)
    print(f"fit events={len(fit)}, val events={len(val)}, test events={len(test)}", flush=True)

    retrieval_cols = base_features + ["pred_log1p_mlp", "density_percentile"]
    retrieval_scaler = fit_standardizer(fit, retrieval_cols)
    memory = select_memory_bank(fit, max_events=args.memory_size, seed=args.seed + 1)
    print(
        f"memory events={len(memory)}, positive_rate={memory['event_label'].mean():.4f}",
        flush=True,
    )

    fit_analog = compute_analog_features(
        fit,
        memory,
        retrieval_cols,
        retrieval_scaler,
        k=args.neighbors,
        exclude_same_gauge=True,
        label="fit",
    )
    val_analog = compute_analog_features(
        val,
        memory,
        retrieval_cols,
        retrieval_scaler,
        k=args.neighbors,
        exclude_same_gauge=False,
        label="val",
    )
    test_analog = compute_analog_features(
        test,
        memory,
        retrieval_cols,
        retrieval_scaler,
        k=args.neighbors,
        exclude_same_gauge=False,
        label="test",
    )
    analog_cols = list(fit_analog.columns)
    fit = pd.concat([fit, fit_analog.reset_index(drop=True)], axis=1)
    val = pd.concat([val, val_analog.reset_index(drop=True)], axis=1)
    test = pd.concat([test, test_analog.reset_index(drop=True)], axis=1)

    event_features = base_features + [
        "pred_log1p_mlp",
        "density_percentile",
    ] + analog_cols
    model, classifier_scaler, training = train_classifier(
        fit,
        val,
        event_features,
        device=device,
        seed=args.seed + 2,
    )
    val["event_probability"] = predict_prob(model, classifier_scaler, val, event_features)
    test["event_probability"] = predict_prob(model, classifier_scaler, test, event_features)
    global_t, group_t = tune_thresholds(val, "event_probability")
    print(f"global_threshold={global_t:.3f}, group_thresholds={group_t}", flush=True)

    mlp_pred = test["pred_streamflow_mlp"] >= test["event_threshold"]
    old_memory_pred = (
        test["pred_streamflow_density_aware_event_memory"] >= test["event_threshold"]
        if "pred_streamflow_density_aware_event_memory" in test.columns
        else mlp_pred
    )
    analog_pred = test["analog_density_event_rate"] >= tune_thresholds(
        val.assign(analog_density_event_rate=val["analog_density_event_rate"]),
        "analog_density_event_rate",
    )[0]
    global_pred = test["event_probability"] >= global_t
    group_pred = np.zeros(len(test), dtype=bool)
    for q, threshold in group_t.items():
        group_pred[test["density_quartile"].astype(str).to_numpy() == q] = (
            test.loc[test["density_quartile"].astype(str) == q, "event_probability"] >= threshold
        )

    summary = summarize_models(
        test,
        [
            ("mlp_point_threshold", mlp_pred),
            ("old_density_memory_point", old_memory_pred),
            ("analog_memory_only", analog_pred),
            ("dahem_event_global", global_pred),
            ("dahem_event_density_calibrated", group_pred),
        ],
    )

    val.to_csv(out_dir / "validation_predictions.csv.gz", index=False)
    test.to_csv(out_dir / "test_predictions.csv.gz", index=False)
    summary.to_csv(out_dir / "event_detection_summary.csv", index=False)
    report = {
        "device": str(device),
        "event_quantile": args.event_quantile,
        "fit_events": int(len(fit)),
        "validation_events": int(len(val)),
        "test_events": int(len(test)),
        "memory_events": int(len(memory)),
        "memory_positive_rate": float(memory["event_label"].mean()),
        "global_threshold": global_t,
        "group_thresholds": group_t,
        "training": training,
        "event_features": event_features,
        "summary": summary.to_dict(orient="records"),
    }
    (out_dir / "event_training_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": event_features,
            "feature_mean": classifier_scaler.mean.to_dict(),
            "feature_std": classifier_scaler.std.to_dict(),
            "training": training,
            "thresholds": {"global": global_t, "density": group_t},
        },
        out_dir / "dahem_event_v1.pt",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
