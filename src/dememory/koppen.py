from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


KOPPEN_CLASSES = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}


def extract_koppen_tif(zip_path: str | Path, member: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / member.replace("/", "_")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as src, out_path.open("wb") as dst:
            dst.write(src.read())
    return out_path


def sample_koppen(
    df: pd.DataFrame,
    tif_path: str | Path,
    lat_col: str = "gauge_lat",
    lon_col: str = "gauge_lon",
) -> pd.DataFrame:
    out = df.copy()
    coords = [(x, y) for x, y in zip(out[lon_col].to_numpy(), out[lat_col].to_numpy())]
    values: list[float] = []
    with rasterio.open(tif_path) as src:
        for val in src.sample(coords):
            v = val[0]
            if src.nodata is not None and v == src.nodata:
                values.append(np.nan)
            else:
                values.append(float(v))
    out["koppen_code"] = pd.Series(values, index=out.index).astype("Int64")
    out["koppen_class"] = out["koppen_code"].map(KOPPEN_CLASSES)
    out["koppen_major"] = out["koppen_class"].str[0]
    return out

