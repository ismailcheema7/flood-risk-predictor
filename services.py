# services.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests

# Import your existing model helpers
from inference import predict_one, load_pipeline


# -----------------------------
# Unpickle compatibility hook
# -----------------------------
# Your saved pipeline references a function named `aspect_to_sin_cos`
# in the __main__ module. This registers it so joblib.load works
# without you redefining it inside app.py.
def aspect_to_sin_cos(X):
    arr = np.asarray(X).astype(float)
    radians = np.deg2rad(arr)
    return np.c_[np.sin(radians), np.cos(radians)]

def wire_aspect_unpickle() -> None:
    """Make aspect_to_sin_cos discoverable under __main__ for joblib unpickle."""
    sys.modules["__main__"].__dict__["aspect_to_sin_cos"] = aspect_to_sin_cos


# -----------------------------
# Static configuration
# -----------------------------
# Approx coords for big, flood-prone Pakistani cities (edit if you want)
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Karachi":         (24.8607, 67.0011),
    "Lahore":          (31.5204, 74.3587),
    "Faisalabad":      (31.4180, 73.0790),
    "Multan":          (30.1984, 71.4687),
    "Hyderabad":       (25.3960, 68.3578),
    "Peshawar":        (34.0151, 71.5249),
    "Quetta":          (30.1798, 66.9750),
    "Sukkur":          (27.7052, 68.8574),
    "Rawalpindi":      (33.5651, 73.0169),
    "Dera Ghazi Khan": (30.0561, 70.6348),
}


# -----------------------------
# Data models
# -----------------------------
@dataclass
class StaticFeatures:
    elevation: float
    slope: float
    aspect: float
    landcover: int  # model-internal code 0..12


@dataclass
class CityPrediction:
    city: str
    lat: float
    lon: float
    precip_1d_mm: float
    precip_3d_mm: float
    static: StaticFeatures
    probability: float  # 0..1


# -----------------------------
# Static features file helpers
# -----------------------------
@lru_cache(maxsize=1)
def load_static_features(path: str = "static_features.csv") -> pd.DataFrame:
    """
    Load static features CSV with columns:
    city,elevation,slope,aspect,landcover
    landcover is the model-internal code (0..12).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing '{path}'. Create it with columns: city,elevation,slope,aspect,landcover"
        )
    df = pd.read_csv(p)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"city", "elevation", "slope", "aspect", "landcover"}
    if not needed.issubset(set(df.columns)):
        missing = needed - set(df.columns)
        raise ValueError(f"'{path}' missing required column(s): {sorted(missing)}")
    # Add normalized key
    df["city_norm"] = df["city"].astype(str).str.strip().str.lower()
    return df


def get_city_record(city: str, path: str = "static_features.csv") -> Optional[StaticFeatures]:
    df = load_static_features(path)
    key = str(city).strip().lower()
    hit = df.loc[df["city_norm"] == key]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return StaticFeatures(
        elevation=float(r["elevation"]),
        slope=float(r["slope"]),
        aspect=float(r["aspect"]),
        landcover=int(r["landcover"]),
    )


def static_template() -> pd.DataFrame:
    """A ready-to-fill template for your static_features.csv."""
    rows = []
    for name, (lat, lon) in CITY_COORDS.items():
        rows.append({
            "city": name,
            "elevation": 50,  # <-- put your real values here
            "slope": 1.0,
            "aspect": 180,
            "landcover": 12,  # model code 0..12
        })
    return pd.DataFrame(rows)


# -----------------------------
# Precipitation fetching
# -----------------------------
def fetch_precip_sums(lat: float, lon: float, timeout: int = 10) -> Tuple[float, float]:
    """
    Fetch last 24h and 72h precipitation (mm) using Open-Meteo hourly precipitation.
    Returns (precip_1d, precip_3d).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "past_days": 3,        # last 72h history
        "forecast_days": 1,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    times = pd.to_datetime(data["hourly"]["time"], utc=True)
    vals = pd.Series(data["hourly"]["precipitation"], index=times)

    now = pd.Timestamp(datetime.now(timezone.utc))
    last24 = (vals.index > now - pd.Timedelta(hours=24)) & (vals.index <= now)
    last72 = (vals.index > now - pd.Timedelta(hours=72)) & (vals.index <= now)
    p1 = float(vals[last24].sum())
    p3 = float(vals[last72].sum())
    return p1, p3


# -----------------------------
# End-to-end city prediction
# -----------------------------
def predict_for_city(
    city: str,
    artifacts_dir: str = "artifacts",
    static_path: str = "static_features.csv",
) -> CityPrediction:
    """
    Resolve city -> coords, fetch precip sums, pull static features,
    and return a CityPrediction with probability.
    """
    # Ensure unpickle works
    wire_aspect_unpickle()
    # Warm pipeline (also ensures artifacts are present)
    load_pipeline(artifacts_dir)

    if city not in CITY_COORDS:
        raise KeyError(f"Unknown city '{city}'. Choose one of: {list(CITY_COORDS)}")
    lat, lon = CITY_COORDS[city]

    static = get_city_record(city, static_path)
    if static is None:
        raise KeyError(
            f"City '{city}' not found in {static_path}. "
            "Add a row with columns: city,elevation,slope,aspect,landcover."
        )

    p1, p3 = fetch_precip_sums(lat, lon)

    prob = predict_one(
        precip_1d=p1,
        precip_3d=p3,
        elevation=static.elevation,
        slope=static.slope,
        aspect=static.aspect,
        landcover=static.landcover,
        artifacts_dir=artifacts_dir,
    )

    return CityPrediction(
        city=city,
        lat=lat,
        lon=lon,
        precip_1d_mm=p1,
        precip_3d_mm=p3,
        static=static,
        probability=prob,
    )


# -----------------------------
# Convenience for UI
# -----------------------------
def list_cities() -> Tuple[str, ...]:
    return tuple(CITY_COORDS.keys())
