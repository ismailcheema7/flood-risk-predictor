import streamlit as st
import pandas as pd
from inference import predict_one, load_pipeline

st.set_page_config(page_title="Flood Risk", page_icon="🌧️", layout="centered")
st.title("Flood Risk Predictor 🌧️")

# app.py (TOP of file, before you call load_pipeline)
import numpy as np

# --- hotfix for scikit-learn pickle mismatch ---
import sklearn.compose._column_transformer as _ct
# define the missing internal symbol so unpickling works
class _RemainderColsList(list):
    pass
if not hasattr(_ct, "_RemainderColsList"):
    _ct._RemainderColsList = _RemainderColsList
# ----------------------------------------------


def aspect_to_sin_cos(X):
    # X will be a 2D array-like with a single 'aspect' column (degrees)
    arr = np.asarray(X).astype(float)
    radians = np.deg2rad(arr)
    sin_col = np.sin(radians)
    cos_col = np.cos(radians)
    # ensure 2D shape for sklearn
    return np.c_[sin_col, cos_col]

# make unpickling robust on Streamlit Cloud
import sys
sys.modules["__main__"].__dict__["aspect_to_sin_cos"] = aspect_to_sin_cos


# warm up the pipeline
load_pipeline("artifacts")

pipe = load_pipeline("artifacts")
cats = pipe.named_steps["pre"].named_transformers_["landcover"].categories_[0]
cats_list = [int(x) for x in cats]  # pretty print as normal ints


tab1, tab2 = st.tabs(["Manual input", "Pick a location"])

with tab1:
    st.subheader("Manual input")
    c1, c2 = st.columns(2)
    with c1:
        p1  = st.number_input("precip_1d (mm)", 0.0, 600.0, 10.0, 0.1)
        elev= st.number_input("elevation (m)",  -50.0, 10000.0, 50.0, 1.0)
        elev = elev / 10
        asp = st.number_input("aspect (deg 0–360)", 0.0, 360.0, 180.0, 1.0)
    with c2:
        p3  = st.number_input("precip_3d (mm)", 0.0, 3000.0, 30.0, 0.1)
        slope=st.number_input("slope (deg)",  0.0,  30.0,  1.0, 0.1)
        lc = st.selectbox("landcover code", options=cats_list)

        LC_NAMES = {
          0:  "Grassland / pasture",
          1:  "Seasonal cropland",
          2:  "Perennial orchards & plantations",
          3:  "Shrubland / bush",
          4:  "Exposed / bare soil",
          5:  "Woodland / forest",
          6:  "Paved ground / asphalt",
          7:  "Urban — concrete/fibercement roofs",
          8:  "Urban — metal roofs",
          9:  "Urban — clay/terracotta roofs",
          10: "Cropland – settlement mosaic",
          11: "Shallow / ponded water",
          12: "RiverSide",
        }

        st.caption(f"Selected landcover: {LC_NAMES.get(lc, 'Unknown')}")


    if st.button("Predict risk", type="primary"):
        proba = predict_one(p1, p3, elev, slope, asp, lc, artifacts_dir="artifacts")
        def band(p):
            if p < 0.10:        return "Low"
            elif p < 0.40:      return "Medium"
            else:               return "High"
        # ...
        st.metric("Flood risk", f"{proba*100:.2f}%  ({band(proba)})")
            # was .1f
        st.caption(f"Raw probability = {proba:.6f}")     # add this line temporarily
        thr = st.slider("Alert threshold", 0.0, 1.0, 0.20, 0.01)
        st.write("Alert:", "🚨 RISK" if proba >= thr else "✅ OK")

from services import (
    list_cities, static_template, predict_for_city,
)

with tab2:
    st.subheader("Pick a location")
    st.caption("Select a city → we fetch last 24h/72h precipitation and combine with your static features.")

    # Downloadable template the first time
    city = st.selectbox("City", options=list_cities())

    if st.button("Predict for city", type="primary"):
        try:
            result = predict_for_city(city, artifacts_dir="artifacts", static_path="static_features.csv")
            st.markdown("### Results")
            st.metric(f"Flood risk — {city}", f"{result.probability*100:.2f}%")
            st.caption(f"Raw probability = {result.probability:.6f}")

            st.markdown("**Inputs used**")
            st.write(pd.DataFrame([{
                "city": result.city,
                "lat": result.lat, "lon": result.lon,
                "precip_1d_mm": round(result.precip_1d_mm, 2),
                "precip_3d_mm": round(result.precip_3d_mm, 2),
                "elevation_m": result.static.elevation,
                "slope_deg": result.static.slope,
                "aspect_deg": result.static.aspect,
                "landcover_code": result.static.landcover,
            }]))
        except Exception as e:
            st.error(str(e))
