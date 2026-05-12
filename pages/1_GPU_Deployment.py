"""GPU deployment guidance page for Streamlit multi-page app."""

import streamlit as st

st.set_page_config(page_title="GPU Deployment", page_icon="🚀", layout="wide")

st.title("🚀 GPU Deployment Guide")
st.markdown("Use this page to deploy the project on RunPod for long-match analysis.")

st.subheader("Recommended Settings")
st.code(
    """
analysis_fps=1.0
resize_width=960
max_frames=5400
GPU=RTX 4090 or better
execution_timeout=7200
""".strip(),
    language="bash",
)

st.subheader("Reference")
st.markdown("Open: `runpod/README_RUNPOD.md`")
