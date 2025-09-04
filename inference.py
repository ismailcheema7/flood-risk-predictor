# inference.py
import joblib, pandas as pd
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def load_pipeline(artifacts_dir="artifacts"):
    return joblib.load(Path(artifacts_dir) / "flood_lgbm_pipeline.joblib")

def predict_one(precip_1d, precip_3d, elevation, slope, aspect, landcover, artifacts_dir="artifacts"):
    pipe = load_pipeline(artifacts_dir)
    X = pd.DataFrame([{
        "precip_1d": precip_1d, "precip_3d": precip_3d,
        "elevation": elevation, "slope": slope,
        "aspect": aspect, "landcover": landcover
    }])

    try:
        enc = pipe.named_steps["pre"].named_transformers_["landcover"]
        cat_dtype = enc.categories_[0].dtype
        X["landcover"] = X["landcover"].astype(cat_dtype)
    except Exception:
        pass

    proba = pipe.predict_proba(X)[:,1][0]
    return float(proba)
