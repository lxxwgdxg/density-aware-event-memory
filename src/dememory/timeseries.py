from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def timeseries_member(row: pd.Series) -> tuple[str, str]:
    archive = row["source_archive"]
    collection = row["source_collection"]
    gauge_id = row["gauge_id"]
    if archive == "Caravan_v1_2":
        return "caravan", f"Caravan/timeseries/csv/{collection}/{gauge_id}.csv"
    if archive == "GRDC_Caravan":
        return "grdc", f"GRDC_Caravan_extension_csv/timeseries/csv/grdc/{gauge_id}.csv"
    raise ValueError(f"Unsupported archive: {archive}")


def read_timeseries_from_zip(zip_path: str | Path, member: str) -> pd.DataFrame | None:
    with zipfile.ZipFile(zip_path) as z:
        if member not in z.namelist():
            return None
        with z.open(member) as f:
            df = pd.read_csv(f)
    return df


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def make_event_features(
    ts: pd.DataFrame,
    static: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    df = ts.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    q_col = "streamflow"
    p_col = first_existing_col(df, ["total_precipitation_sum"])
    pet_col = first_existing_col(
        df,
        [
            "potential_evaporation_sum",
            "potential_evaporation_sum_ERA5_LAND",
            "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
        ],
    )
    t_col = first_existing_col(df, ["temperature_2m_mean"])
    swe_col = first_existing_col(df, ["snow_depth_water_equivalent_mean"])
    soil1_col = first_existing_col(df, ["volumetric_soil_water_layer_1_mean"])
    soil2_col = first_existing_col(df, ["volumetric_soil_water_layer_2_mean"])

    required = [q_col, p_col, pet_col, t_col]
    if any(c is None for c in required):
        return pd.DataFrame()

    for col in [q_col, p_col, pet_col, t_col, swe_col, soil1_col, soil2_col]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feat = pd.DataFrame(index=df.index)
    feat["date"] = df["date"]
    feat["target_streamflow"] = df[q_col].shift(-horizon)
    feat["q_current"] = df[q_col]
    feat["p_1d"] = df[p_col]
    feat["p_7d"] = df[p_col].rolling(7, min_periods=5).sum()
    feat["p_30d"] = df[p_col].rolling(30, min_periods=20).sum()
    feat["pet_7d"] = df[pet_col].rolling(7, min_periods=5).sum()
    feat["pet_30d"] = df[pet_col].rolling(30, min_periods=20).sum()
    feat["t_7d"] = df[t_col].rolling(7, min_periods=5).mean()
    # Retrospective perfect-forcing proxy: these target-horizon meteorological
    # values come from the observed/reanalysis record, not an operational NWP
    # archive. Target discharge remains excluded from score construction.
    future_min_periods = max(1, min(int(horizon), 5))
    feat["p_future_h"] = (
        df[p_col]
        .shift(-1)
        .iloc[::-1]
        .rolling(int(horizon), min_periods=future_min_periods)
        .sum()
        .iloc[::-1]
    )
    feat["pet_future_h"] = (
        df[pet_col]
        .shift(-1)
        .iloc[::-1]
        .rolling(int(horizon), min_periods=future_min_periods)
        .sum()
        .iloc[::-1]
    )
    feat["t_future_h"] = (
        df[t_col]
        .shift(-1)
        .iloc[::-1]
        .rolling(int(horizon), min_periods=future_min_periods)
        .mean()
        .iloc[::-1]
    )
    if swe_col is not None:
        feat["swe_current"] = df[swe_col]
        feat["swe_7d"] = df[swe_col].rolling(7, min_periods=5).mean()
    else:
        feat["swe_current"] = np.nan
        feat["swe_7d"] = np.nan
    feat["soil1_current"] = df[soil1_col] if soil1_col is not None else np.nan
    feat["soil2_current"] = df[soil2_col] if soil2_col is not None else np.nan
    month = feat["date"].dt.month
    feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12)

    static_cols = [
        "gauge_id",
        "source_archive",
        "source_collection",
        "country",
        "density_quartile",
        "density_percentile",
        "koppen_class",
        "koppen_major",
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
    for col in static_cols:
        if col in static.index:
            feat[col] = static[col]

    # For the ungauged-style pilot model, q_current is retained for diagnostics but
    # excluded from the predictive feature list. It lets us later compare persistence.
    feat["target_log1p"] = np.log1p(feat["target_streamflow"].clip(lower=0))
    return feat.dropna(subset=["date", "target_streamflow", "p_7d", "p_30d", "pet_7d", "t_7d"])


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 5:
        return np.nan
    obs = obs[mask]
    sim = sim[mask]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((sim - obs) ** 2) / denom


def high_flow_bias(obs: np.ndarray, sim: np.ndarray, q: float = 0.95) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 20:
        return np.nan
    obs = obs[mask]
    sim = sim[mask]
    threshold = np.quantile(obs, q)
    event = obs >= threshold
    if event.sum() == 0:
        return np.nan
    return float(np.mean(sim[event] - obs[event]))
