from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CARAVAN_ATTRIBUTE_ROOT = "Caravan/attributes"
GRDC_ATTRIBUTE_ROOT = "GRDC_Caravan_extension_csv/attributes/grdc"


def _read_zip_csv(zip_file: zipfile.ZipFile, member: str, **kwargs) -> pd.DataFrame:
    with zip_file.open(member) as f:
        return pd.read_csv(f, **kwargs)


def _caravan_sources(zip_file: zipfile.ZipFile) -> list[str]:
    names = zip_file.namelist()
    sources = set()
    prefix = f"{CARAVAN_ATTRIBUTE_ROOT}/"
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".csv"):
            continue
        parts = name.split("/")
        if len(parts) >= 4:
            sources.add(parts[2])
    return sorted(sources)


def load_caravan_gauge_catalog(zip_path: str | Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for source in _caravan_sources(z):
            base = f"{CARAVAN_ATTRIBUTE_ROOT}/{source}"
            other_name = f"{base}/attributes_other_{source}.csv"
            caravan_name = f"{base}/attributes_caravan_{source}.csv"
            hydro_name = f"{base}/attributes_hydroatlas_{source}.csv"
            if other_name not in z.namelist() or caravan_name not in z.namelist():
                continue
            other = _read_zip_csv(z, other_name)
            caravan = _read_zip_csv(z, caravan_name)
            df = other.merge(caravan, on="gauge_id", how="left", suffixes=("", "_caravan"))
            if hydro_name in z.namelist():
                hydro_cols = [
                    "gauge_id",
                    "gdp_ud_sav",
                    "pop_ct_usu",
                    "hdi_ix_sav",
                    "lka_pc_sse",
                    "dor_pc_pva",
                    "ari_ix_sav",
                    "run_mm_syr",
                    "pre_mm_syr",
                    "pet_mm_syr",
                    "clz_cl_smj",
                ]
                hydro = _read_zip_csv(z, hydro_name, usecols=lambda c: c in hydro_cols)
                df = df.merge(hydro, on="gauge_id", how="left")
            df["source_collection"] = source
            df["source_archive"] = "Caravan_v1_2"
            rows.append(df)
    if not rows:
        raise RuntimeError(f"No Caravan attributes found in {zip_path}")
    return pd.concat(rows, ignore_index=True, sort=False)


def load_grdc_gauge_catalog(zip_path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        other = _read_zip_csv(z, f"{GRDC_ATTRIBUTE_ROOT}/attributes_other_grdc.csv")
        caravan = _read_zip_csv(z, f"{GRDC_ATTRIBUTE_ROOT}/attributes_caravan_grdc.csv")
        additional = _read_zip_csv(z, f"{GRDC_ATTRIBUTE_ROOT}/attributes_additional_grdc.csv")
        hydro = _read_zip_csv(
            z,
            f"{GRDC_ATTRIBUTE_ROOT}/attributes_hydroatlas_grdc.csv",
            usecols=lambda c: c
            in {
                "gauge_id",
                "gdp_ud_sav",
                "pop_ct_usu",
                "hdi_ix_sav",
                "lka_pc_sse",
                "dor_pc_pva",
                "ari_ix_sav",
                "run_mm_syr",
                "pre_mm_syr",
                "pet_mm_syr",
                "clz_cl_smj",
            },
        )
    df = other.merge(caravan, on="gauge_id", how="left", suffixes=("", "_caravan"))
    keep_additional = [
        "gauge_id",
        "country",
        "d_start",
        "d_end",
        "d_yrs",
        "d_miss",
        "quality",
        "source",
        "wmo_reg",
        "altitude",
        "lta_discharge",
    ]
    additional = additional[[c for c in keep_additional if c in additional.columns]].rename(
        columns={"country": "country_code"}
    )
    df = df.merge(additional, on="gauge_id", how="left")
    df = df.merge(hydro, on="gauge_id", how="left")
    coalesce_map = {
        "aridity": ["aridity_ERA5_LAND", "aridity_FAO_PM"],
        "pet_mean": ["pet_mean_ERA5_LAND", "pet_mean_FAO_PM"],
        "moisture_index": ["moisture_index_ERA5_LAND", "moisture_index_FAO_PM"],
        "seasonality": ["seasonality_ERA5_LAND", "seasonality_FAO_PM"],
    }
    for target, candidates in coalesce_map.items():
        if target not in df.columns:
            df[target] = np.nan
        for candidate in candidates:
            if candidate in df.columns:
                df[target] = df[target].combine_first(df[candidate])
    df["source_collection"] = "grdc"
    df["source_archive"] = "GRDC_Caravan"
    return df


def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "gauge_lat",
        "gauge_lon",
        "area",
        "p_mean",
        "pet_mean",
        "aridity",
        "frac_snow",
        "moisture_index",
        "seasonality",
        "high_prec_freq",
        "high_prec_dur",
        "low_prec_freq",
        "low_prec_dur",
        "gdp_ud_sav",
        "pop_ct_usu",
        "hdi_ix_sav",
        "lka_pc_sse",
        "dor_pc_pva",
        "ari_ix_sav",
        "run_mm_syr",
        "pre_mm_syr",
        "pet_mm_syr",
        "d_yrs",
        "d_miss",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["gauge_id", "gauge_lat", "gauge_lon"]).copy()
    out["area_log10"] = np.log10(out["area"].clip(lower=1e-3))
    out["country"] = out["country"].replace(
        {
            "United States of America": "United States",
            "Scotland": "United Kingdom",
            "England": "United Kingdom",
            "Wales": "United Kingdom",
        }
    )
    return out


def add_hydroclimatic_density(
    df: pd.DataFrame,
    features: list[str],
    k: int = 50,
) -> pd.DataFrame:
    out = df.copy()
    present_features = [f for f in features if f in out.columns]
    matrix = out[present_features].copy()
    for col in present_features:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
        matrix[col] = matrix[col].fillna(matrix[col].median())
    scaler = StandardScaler()
    x = scaler.fit_transform(matrix)
    k_eff = min(k + 1, len(out))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nbrs.fit(x)
    distances, _ = nbrs.kneighbors(x)
    kth = distances[:, -1]
    eps = 1e-9
    # Smaller kth distance means denser local support.
    out["density_k"] = k
    out["density_kdist"] = kth
    out["density_support"] = 1.0 / (kth + eps)
    out["density_log_support"] = np.log(out["density_support"])
    out["density_percentile"] = pd.Series(out["density_support"]).rank(pct=True).to_numpy()
    out["density_quartile"] = pd.qcut(
        out["density_percentile"],
        q=4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
        duplicates="drop",
    ).astype(str)
    return out


def summarize_catalog(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_source = (
        df.groupby(["source_archive", "source_collection"], dropna=False)
        .agg(
            gauges=("gauge_id", "count"),
            countries=("country", "nunique"),
            median_density_percentile=("density_percentile", "median"),
            median_area_km2=("area", "median"),
        )
        .reset_index()
        .sort_values("gauges", ascending=False)
    )
    by_country = (
        df.groupby("country", dropna=False)
        .agg(
            gauges=("gauge_id", "count"),
            source_archives=("source_archive", "nunique"),
            source_collections=("source_collection", "nunique"),
            median_density_percentile=("density_percentile", "median"),
            median_aridity=("aridity", "median"),
            median_area_km2=("area", "median"),
        )
        .reset_index()
        .sort_values("gauges", ascending=False)
    )
    by_koppen = (
        df.groupby(["koppen_major", "koppen_class"], dropna=False)
        .agg(
            gauges=("gauge_id", "count"),
            countries=("country", "nunique"),
            median_density_percentile=("density_percentile", "median"),
            median_aridity=("aridity", "median"),
        )
        .reset_index()
        .sort_values("gauges", ascending=False)
    )
    by_density = (
        df.groupby("density_quartile", dropna=False)
        .agg(
            gauges=("gauge_id", "count"),
            countries=("country", "nunique"),
            median_aridity=("aridity", "median"),
            median_frac_snow=("frac_snow", "median"),
            median_area_km2=("area", "median"),
        )
        .reset_index()
    )
    return {
        "summary_by_source": by_source,
        "summary_by_country": by_country,
        "summary_by_koppen": by_koppen,
        "summary_by_density_quartile": by_density,
    }
