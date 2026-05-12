"""Quick runtime checks for local environment."""

import importlib
import streamlit as st

st.set_page_config(page_title="System Check", page_icon="🧪", layout="wide")

st.title("🧪 System Check")

packages = [
    "cv2",
    "numpy",
    "pandas",
    "ultralytics",
    "supervision",
    "streamlit",
]

rows = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        rows.append({"package": pkg, "status": "ok"})
    except Exception as e:
        rows.append({"package": pkg, "status": f"missing: {e.__class__.__name__}"})

st.dataframe(rows, use_container_width=True)
