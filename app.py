"""
Streamlit demo: Explainable DR screening
Run:  streamlit run app.py
Needs models/best_model.pth and models/seg_model.pth next to this file.
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import stage7_pipeline as s7

st.set_page_config(page_title="Explainable DR Screening", layout="wide")


@st.cache_resource
def load_models():
    return s7.load_models("models/best_model.pth", "models/seg_model.pth")


st.title("Explainable Diabetic Retinopathy Screening")
st.caption("Portfolio project inspired by SIH 2026 problem statement 26038. "
           "Screening aid only - not a diagnosis. A clinician must confirm.")

with st.sidebar:
    st.header("About this demo")
    st.write("Pipeline: image quality check -> severity grading + lesion segmentation "
             "-> evidence report.")
    st.subheader("Known limitations")
    st.write("- Microaneurysms and soft exudates are **not assessed** "
             "(the segmentation model was not reliable for them).")
    st.write("- Segmentation was trained on 43 full-resolution IDRiD images. "
             "Works best on full-resolution fundus photos, not small thumbnails.")
    st.write("- Locations are image-relative (upper/lower, left/right); "
             "left vs right eye is not detected.")

file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = np.array(Image.open(file).convert("RGB"))
    cls_model, seg_model = load_models()

    with st.spinner("Analysing image..."):
        res = s7.run_pipeline(img, cls_model, seg_model)

    with st.expander("Image quality checks", expanded=(res["status"] == "recapture")):
        st.json(res["quality"])

    if res["status"] == "recapture":
        st.error(res["report"])
    else:
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original", use_container_width=True)
        c2.image(res["overlay"], use_container_width=True,
                 caption="Evidence overlay: yellow = hard exudates, blue = hemorrhages, "
                         "green = optic disc")

        m1, m2 = st.columns(2)
        m1.metric("Predicted severity", res["severity"])
        m2.metric("Confidence", f"{res['confidence'] * 100:.0f}%")

        callout = st.warning if res["recommendation"].startswith(("Refer", "Urgent")) else st.info
        callout(f"Suggested next step: {res['recommendation']}")

        st.subheader("Evidence report")
        st.text(res["report"])

        st.subheader("Severity probabilities")
        st.bar_chart(pd.DataFrame({"probability": res["probs"]}, index=s7.SEVERITY))
else:
    st.info("Upload a fundus photo to begin. For a good demo, use a full-resolution "
            "image such as one from the IDRiD test set.")
