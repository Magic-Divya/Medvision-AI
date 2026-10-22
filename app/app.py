
import sys
from pathlib import Path

# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError
from predict import load_model, predict_image


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedVision AI | X-ray Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CONSTANTS
# ============================================================

APP_TITLE = "MedVision AI"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

SUPPORTED_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8b95a5;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .info-card {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading the trained AI model...")
def get_model():
    """Load and cache the trained model for this app session."""
    return load_model()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def normalize_prediction(prediction):
    """Convert the prediction to a readable class label."""

    if isinstance(prediction, str):
        label = prediction.strip().upper()

        if label in CLASS_NAMES:
            return label

        raise ValueError(
            f"Unexpected prediction label returned by predict.py: "
            f"{prediction}"
        )

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    if isinstance(prediction, (int, float)):
        index = int(prediction)

        if index in range(len(CLASS_NAMES)):
            return CLASS_NAMES[index]

    raise ValueError(
        f"Unsupported prediction value: {prediction!r}"
    )


def normalize_confidence(confidence):
    """Convert confidence to a percentage for display."""

    if hasattr(confidence, "item"):
        confidence = confidence.item()

    confidence = float(confidence)

    if 0 <= confidence <= 1:
        confidence *= 100

    return max(0.0, min(confidence, 100.0))


def normalize_probabilities(probabilities):
    """Convert class probabilities into a displayable dictionary."""

    if hasattr(probabilities, "detach"):
        probabilities = probabilities.detach().cpu().numpy()

    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()

    if isinstance(probabilities, dict):
        result = {}

        for label, value in probabilities.items():
            result[str(label).upper()] = float(value)

        return result

    if isinstance(probabilities, (list, tuple)):
        values = probabilities

        # Handle a single-row batch such as [[0.2, 0.8]].
        if (
            len(values) == 1
            and isinstance(values[0], (list, tuple))
        ):
            values = values[0]

        if len(values) == len(CLASS_NAMES):
            return {
                label: float(value)
                for label, value in zip(CLASS_NAMES, values)
            }

    return {}


def format_probability(value):
    """Format a probability or percentage consistently."""

    value = float(value)

    if 0 <= value <= 1:
        value *= 100

    return max(0.0, min(value, 100.0))


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🩺 MedVision AI")

    st.caption("Deep Learning • Computer Vision")

    st.divider()

    st.subheader("About this project")

    st.write(
        "MedVision AI is an educational deep-learning project "
        "that demonstrates binary classification of chest "
        "X-ray images."
    )

    st.write("**Classes**")
    st.write("- NORMAL")
    st.write("- PNEUMONIA")

    st.divider()

    st.subheader("Model information")

    st.write("**Architecture:** DenseNet121")
    st.write("**Input size:** 224 × 224 pixels")
    st.write("**Framework:** PyTorch")

    st.divider()

    st.caption(
        "This application is for academic demonstration only. "
        "It is not a medical device and must not be used for "
        "diagnosis or treatment decisions."
    )


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 MedVision AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Chest X-ray Classification using Deep Learning'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Upload a chest X-ray image to demonstrate the model's "
    "NORMAL/PNEUMONIA classification."
)


# ============================================================
# 8. IMAGE UPLOAD
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large",
)

with left_column:
    st.subheader("📤 Upload X-ray")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=SUPPORTED_TYPES,
        help="Supported formats: PNG, JPG, JPEG, BMP and WEBP.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded chest X-ray",
                width="stretch",
            )

            st.caption(
                f"Image dimensions: {image.width} × {image.height}"
            )

        except (UnidentifiedImageError, OSError):
            image = None
            st.error(
                "The uploaded file could not be opened as an image. "
                "Please upload a valid image."
            )
    else:
        image = None

        st.markdown(
            """
            <div class="info-card">
                <h4>Get started</h4>
                <p>Upload a chest X-ray image using the control above.</p>
                <p>The model will then generate a classification result.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 9. MODEL INFERENCE
# ============================================================

with right_column:
    st.subheader("🔬 Analysis result")

    analyze_clicked = st.button(
        "Analyze image",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )

    if analyze_clicked and image is not None:
        try:
            with st.spinner("Loading model and analyzing image..."):
                model = get_model()

                result = predict_image(image, model)

                if not isinstance(result, (tuple, list)):
                    raise ValueError(
                        "predict_image() must return "
                        "(prediction, confidence, probabilities)."
                    )

                if len(result) != 3:
                    raise ValueError(
                        "Unexpected prediction output. Expected "
                        "three values: prediction, confidence, "
                        "probabilities."
                    )

                prediction, confidence, probabilities = result

                label = normalize_prediction(prediction)
                confidence_pct = normalize_confidence(confidence)

                probability_dict = normalize_probabilities(
                    probabilities
                )

            st.session_state["medvision_result"] = {
                "label": label,
                "confidence": confidence_pct,
                "probabilities": probability_dict,
                "filename": uploaded_file.name,
            }

        except Exception as exc:
            st.session_state.pop("medvision_result", None)

            st.error(
                "The image could not be analyzed. Check the "
                "application logs for the underlying error."
            )

            with st.expander("Technical error details"):
                st.code(str(exc))


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

result = st.session_state.get("medvision_result")

if result is not None:
    with right_column:
        st.divider()

        label = result["label"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        st.markdown("### Classification")

        if label == "NORMAL":
            st.success("Model prediction: NORMAL")
        else:
            st.warning("Model prediction: PNEUMONIA")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Predicted class",
                label,
            )

        with metric_col2:
            st.metric(
                "Model confidence",
                f"{confidence:.2f}%",
            )

        st.progress(int(round(confidence)))

        st.caption(
            "Confidence is the model's output score, not a "
            "measure of clinical certainty."
        )

        if probabilities:
            st.divider()
            st.markdown("### Class probabilities")

            chart_data = pd.DataFrame(
                {
                    "Class": list(probabilities.keys()),
                    "Probability (%)": [
                        format_probability(value)
                        for value in probabilities.values()
                    ],
                }
            )

            st.bar_chart(
                chart_data.set_index("Class"),
                y="Probability (%)",
            )

            for class_name, value in probabilities.items():
                st.write(
                    f"**{class_name}:** "
                    f"{format_probability(value):.2f}%"
                )

        st.caption(f"File analyzed: {result['filename']}")

else:
    with right_column:
        st.markdown(
            """
            <div class="info-card">
                <h4>Awaiting image analysis</h4>
                <p>Your classification result will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "MedVision AI | Academic Deep Learning Mini Project"
)

st.warning(
    "Educational use only. This model may make mistakes, "
    "including missing pneumonia or incorrectly classifying "
    "normal images. Do not use its output for medical decisions."
)
