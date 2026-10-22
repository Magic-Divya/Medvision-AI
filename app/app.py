
"""Streamlit interface for the MedVision-AI research project."""

import streamlit as st
from PIL import Image
from predict import load_model, predict_image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MedVision-AI",
    page_icon="🩻",
    layout="centered",
)

# --------------------------------------------------
# Application header
# --------------------------------------------------
st.title("🩻 MedVision-AI")
st.subheader("Deep Learning for Chest X-ray Image Classification")

st.write(
    "Upload a chest X-ray image to explore the predictions "
    "made by our trained DenseNet121 deep learning model."
)

st.warning(
    "Educational research demonstration only. This model is "
    "not validated for clinical use and cannot diagnose or "
    "rule out pneumonia. Do not use its predictions for "
    "medical decisions."
)


# --------------------------------------------------
# Load model once and cache it
# --------------------------------------------------
@st.cache_resource
def get_model():
    """Load and cache the trained model."""
    return load_model()


# --------------------------------------------------
# Image upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Updated Streamlit parameter
        st.image(
            image,
            caption="Uploaded chest X-ray",
            width="stretch",
        )

        if st.button("Analyze image", type="primary"):
            with st.spinner("Running model inference..."):
                model = get_model()
                result = predict_image(image, model)

            st.subheader("Model output")

            prediction = result["prediction"]
            confidence = result["confidence"]

            st.write("**Predicted class:**", prediction)

            st.metric(
                label="Model confidence",
                value=f"{confidence * 100:.2f}%",
            )

            st.write("### Class probabilities")
            st.bar_chart(result["probabilities"])

            st.caption(
                "Confidence represents the model's estimated "
                "probability, not clinical certainty. Predictions "
                "may be incorrect."
            )

    except Exception as exc:
        st.error(f"Unable to process the image: {exc}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "MedVision-AI | Final-year Deep Learning project"
)

st.caption(
    "Research and educational use only — not a medical "
    "diagnostic tool."
)
