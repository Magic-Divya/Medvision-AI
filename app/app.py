
"""MedVision AI — Advanced research dashboard."""

import sys
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from PIL import Image, ImageStat, UnidentifiedImageError

# ------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import load_model, predict_image


APP_NAME = "MedVision AI"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]

METRICS_PATH = PROJECT_ROOT / "reports" / "evaluation" / "metrics.json"
AUDIT_PATH = PROJECT_ROOT / "reports" / "dataset_quality_audit.md"

st.set_page_config(
    page_title="MedVision AI | Intelligence Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# 2. PROFESSIONAL UI STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            rgba(12, 24, 43, 0.035),
            rgba(25, 118, 140, 0.035)
        );
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(100, 160, 190, 0.30);
        border-radius: 20px;
        background: linear-gradient(
            120deg,
            #10243b 0%,
            #153e58 60%,
            #126b78 100%
        );
        color: white;
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        font-size: 2.35rem;
        margin: 0;
        color: white;
        letter-spacing: -1px;
    }

    .hero p {
        color: #d6e8f0;
        margin-bottom: 0;
        margin-top: 0.55rem;
        font-size: 1.02rem;
    }

    .section-label {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.74rem;
        font-weight: 700;
        opacity: 0.7;
    }

    .panel {
        border: 1px solid rgba(128, 128, 128, 0.27);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        margin: 0.3rem 0;
    }

    .muted {
        opacity: 0.72;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.22);
        padding: 14px;
        border-radius: 13px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 3. SESSION STATE
# ------------------------------------------------------------

if "medvision_history" not in st.session_state:
    st.session_state.medvision_history = []

if "medvision_result" not in st.session_state:
    st.session_state.medvision_result = None


# ------------------------------------------------------------
# 4. MODEL LOADING
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Loading DenseNet121...")
def get_model():
    """Load the existing trained model once per app process."""
    return load_model()


# ------------------------------------------------------------
# 5. IMAGE AND PREDICTION UTILITIES
# ------------------------------------------------------------

def get_image_statistics(image):
    """Calculate basic image statistics, not medical features."""
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)

    return {
        "width": image.width,
        "height": image.height,
        "mean_brightness": round(stats.mean[0], 2),
        "contrast_std": round(stats.stddev[0], 2),
    }


def run_prediction(image):
    """
    Use the existing predict.py function.

    Expected result:
    {
        "prediction": "NORMAL" or "PNEUMONIA",
        "confidence": probability between 0 and 1,
        "probabilities": {"NORMAL": ..., "PNEUMONIA": ...}
    }
    """
    model = get_model()
    output = predict_image(image, model)

    if not isinstance(output, dict):
        raise TypeError(
            "predict_image() must return a dictionary."
        )

    for key in ("prediction", "confidence", "probabilities"):
        if key not in output:
            raise ValueError(
                f"Prediction result is missing '{key}'."
            )

    label = str(output["prediction"]).strip().upper()

    if label not in CLASS_NAMES:
        raise ValueError(f"Unexpected predicted class: {label}")

    probabilities = output["probabilities"]

    if not isinstance(probabilities, dict):
        raise TypeError("Class probabilities must be a dictionary.")

    clean_probs = {}

    for class_name in CLASS_NAMES:
        if class_name not in probabilities:
            raise ValueError(
                f"Probability for {class_name} is missing."
            )

        value = float(probabilities[class_name])

        if not 0 <= value <= 1:
            raise ValueError(
                f"Probability for {class_name} must be between 0 and 1."
            )

        clean_probs[class_name] = value

    confidence = float(output["confidence"])

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": clean_probs,
    }


def percent(value):
    return float(value) * 100.0


def load_json_report(path):
    """Read a report if available; do not invent missing metrics."""
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


# ------------------------------------------------------------
# 6. SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🩺 MedVision AI")
    st.caption("COMPUTER VISION RESEARCH PLATFORM")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "X-ray Analysis",
            "Analysis History",
            "Model Insights",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### Model configuration")
    st.write("**Architecture**")
    st.write("DenseNet121")

    st.write("**Input resolution**")
    st.write("224 × 224 pixels")

    st.write("**Output classes**")
    st.write("NORMAL · PNEUMONIA")

    st.divider()

    st.caption(
        "Academic research demonstration only. "
        "Not intended for clinical diagnosis."
    )


# ------------------------------------------------------------
# 7. HEADER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="section-label">Deep Learning Research Workspace</div>
        <h1>MedVision AI</h1>
        <p>
            Chest X-ray classification and model-output exploration
            powered by DenseNet121.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Project overview")
    st.write(
        "Explore model predictions, review session activity, "
        "and inspect the evaluation results saved by your project."
    )

    history = st.session_state.medvision_history
    total = len(history)
    normal_count = sum(
        row["prediction"] == "NORMAL" for row in history
    )
    pneumonia_count = sum(
        row["prediction"] == "PNEUMONIA" for row in history
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Session analyses", total)
    col2.metric("NORMAL outputs", normal_count)
    col3.metric("PNEUMONIA outputs", pneumonia_count)

    if history:
        average_confidence = sum(
            row["confidence_pct"] for row in history
        ) / total
        col4.metric(
            "Average model confidence",
            f"{average_confidence:.1f}%",
        )
    else:
        col4.metric("Average model confidence", "—")

    st.divider()

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("### Analysis activity")

        if history:
            history_df = pd.DataFrame(history)

            chart_df = history_df[
                ["timestamp", "confidence_pct"]
            ].rename(
                columns={
                    "timestamp": "Analysis time",
                    "confidence_pct": "Confidence (%)",
                }
            )

            st.line_chart(
                chart_df.set_index("Analysis time"),
                y="Confidence (%)",
            )

            st.caption(
                "Confidence trend for this browser session only. "
                "It is not a model accuracy or performance trend."
            )
        else:
            st.info(
                "No images have been analyzed in this session yet. "
                "Open X-ray Analysis to begin."
            )

    with right:
        st.markdown("### Prediction distribution")

        if history:
            distribution = pd.DataFrame(
                {
                    "Class": ["NORMAL", "PNEUMONIA"],
                    "Analyses": [normal_count, pneumonia_count],
                }
            )

            st.bar_chart(
                distribution.set_index("Class"),
                y="Analyses",
            )
        else:
            st.info(
                "The class distribution will appear after "
                "you analyze images."
            )

    st.divider()
    st.markdown("### Evaluation report")

    metrics_report = load_json_report(METRICS_PATH)

    if metrics_report is not None:
        st.success("Saved evaluation metrics were found.")

        with st.expander("View saved evaluation metrics"):
            st.json(metrics_report)

        st.caption(
            "These are values from your saved evaluation file. "
            "They are separate from the current session's predictions."
        )
    else:
        st.info(
            "No readable reports/evaluation/metrics.json file was found. "
            "Run your project's evaluation script to create one."
        )

    if st.button("Open X-ray Analysis", type="primary"):
        st.session_state["requested_page"] = "X-ray Analysis"
        st.rerun()


# ============================================================
# PAGE 2 — X-RAY ANALYSIS
# ============================================================

elif page == "X-ray Analysis":

    st.subheader("X-ray analysis workspace")
    st.write(
        "Upload an image, inspect its basic properties, and "
        "run inference with the existing trained model."
    )

    upload_col, result_col = st.columns(
        [1, 1],
        gap="large",
    )

    with upload_col:
        st.markdown("### 01 · Input image")

        uploaded_file = st.file_uploader(
            "Upload chest X-ray",
            type=SUPPORTED_TYPES,
            help="Supported formats: PNG, JPG, JPEG, BMP and WEBP.",
            key="xray_uploader",
        )

        image = None
        image_hash = None
        image_stats = None

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                image_hash = hashlib.sha256(raw_bytes).hexdigest()

                image = Image.open(
                    io.BytesIO(raw_bytes)
                ).convert("RGB")

                image_stats = get_image_statistics(image)

                st.image(
                    image,
                    caption="Uploaded image preview",
                    use_container_width=True,
                )

                m1, m2 = st.columns(2)
                m1.metric(
                    "Width",
                    f"{image_stats['width']} px",
                )
                m2.metric(
                    "Height",
                    f"{image_stats['height']} px",
                )

                with st.expander("Image properties"):
                    st.write(
                        f"**Filename:** {uploaded_file.name}"
                    )
                    st.write(
                        f"**Format:** {uploaded_file.type or 'Unknown'}"
                    )
                    st.write(
                        f"**File size:** "
                        f"{len(raw_bytes) / (1024 * 1024):.2f} MB"
                    )
                    st.write(
                        f"**Mean grayscale brightness:** "
                        f"{image_stats['mean_brightness']}"
                    )
                    st.write(
                        f"**Grayscale intensity standard deviation:** "
                        f"{image_stats['contrast_std']}"
                    )

                st.caption(
                    "Brightness and contrast are basic image statistics. "
                    "They do not determine whether an X-ray is medically valid."
                )

            except (UnidentifiedImageError, OSError, ValueError) as exc:
                image = None
                st.error("This file could not be processed as an image.")

                with st.expander("Technical details"):
                    st.code(str(exc))

        else:
            st.markdown(
                """
                <div class="panel">
                    <h4>Awaiting image input</h4>
                    <p>Upload an image to see its preview and properties.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with result_col:
        st.markdown("### 02 · Model inference")

        st.write("**Inference engine:** DenseNet121")
        st.write("**Preprocessing:** Resize, tensor conversion, ImageNet normalization")
        st.write("**Output:** NORMAL / PNEUMONIA")

        analyze_button = st.button(
            "Run AI analysis",
            type="primary",
            use_container_width=True,
            disabled=image is None,
        )

        if analyze_button and image is not None:
            try:
                with st.spinner(
                    "Loading model and running inference..."
                ):
                    output = run_prediction(image)

                record = {
                    "timestamp": datetime.now().astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "filename": uploaded_file.name,
                    "image_hash": image_hash,
                    "prediction": output["prediction"],
                    "confidence_pct": round(
                        percent(output["confidence"]), 4
                    ),
                    "normal_probability_pct": round(
                        percent(output["probabilities"]["NORMAL"]), 4
                    ),
                    "pneumonia_probability_pct": round(
                        percent(output["probabilities"]["PNEUMONIA"]), 4
                    ),
                    "width": image_stats["width"],
                    "height": image_stats["height"],
                    "mean_brightness": image_stats["mean_brightness"],
                    "contrast_std": image_stats["contrast_std"],
                }

                st.session_state.medvision_result = record

                if "medvision_history" not in st.session_state:
                    st.session_state.medvision_history = []

                st.session_state.medvision_history.append(record)

            except Exception as exc:
                st.session_state.medvision_result = None

                st.error(
                    "Inference failed. Check the technical details "
                    "below and the terminal logs."
                )

                with st.expander("Technical error details"):
                    st.code(str(exc))

        current_result = st.session_state.medvision_result

        if (
            current_result is not None
            and image_hash is not None
            and current_result["image_hash"] == image_hash
        ):
            st.divider()
            st.markdown("### 03 · Prediction result")

            prediction = current_result["prediction"]
            confidence = current_result["confidence_pct"]

            if prediction == "NORMAL":
                st.success("Model output: NORMAL")
            else:
                st.warning("Model output: PNEUMONIA")

            c1, c2 = st.columns(2)

            c1.metric(
                "Predicted class",
                prediction,
            )

            c2.metric(
                "Model confidence",
                f"{confidence:.2f}%",
            )

            st.progress(
                min(100, max(0, int(round(confidence))))
            )

            st.markdown("#### Class probability distribution")

            probabilities_df = pd.DataFrame(
                {
                    "Class": ["NORMAL", "PNEUMONIA"],
                    "Probability (%)": [
                        current_result["normal_probability_pct"],
                        current_result["pneumonia_probability_pct"],
                    ],
                }
            )

            st.bar_chart(
                probabilities_df.set_index("Class"),
                y="Probability (%)",
            )

            st.dataframe(
                probabilities_df,
                hide_index=True,
                use_container_width=True,
            )

            csv_data = pd.DataFrame([current_result]).to_csv(
                index=False
            )

            st.download_button(
                "Download this result as CSV",
                data=csv_data,
                file_name="medvision_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.caption(
                "Confidence is a model score, not a measure of "
                "clinical certainty. The output must not be used "
                "to diagnose or rule out pneumonia."
            )

        elif current_result is None:
            st.info(
                "Upload an image and select Run AI analysis to see "
                "the prediction and probability distribution."
            )

        elif (
            image_hash is not None
            and current_result["image_hash"] != image_hash
        ):
            st.info(
                "You selected a different image. Run the analysis "
                "to generate a new result for it."
            )

    st.divider()

    st.markdown("### How to interpret the output")

    explain_col1, explain_col2 = st.columns(2)

    with explain_col1:
        st.markdown(
            """
            <div class="panel">
                <b>Predicted class</b>
                <p>The class with the highest model output probability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with explain_col2:
        st.markdown(
            """
            <div class="panel">
                <b>Model confidence</b>
                <p>The softmax probability assigned to the predicted class.
                It is not calibrated clinical certainty.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 3 — ANALYSIS HISTORY
# ============================================================

elif page == "Analysis History":

    st.subheader("Analysis history")
    st.write(
        "Review predictions generated during this browser session "
        "and export them for your project report."
    )

    history = st.session_state.medvision_history

    if history:
        history_df = pd.DataFrame(history)

        st.metric("Total analyses in this session", len(history))

        st.dataframe(
            history_df[
                [
                    "timestamp",
                    "filename",
                    "prediction",
                    "confidence_pct",
                    "normal_probability_pct",
                    "pneumonia_probability_pct",
                ]
            ].rename(
                columns={
                    "timestamp": "Timestamp",
                    "filename": "Filename",
                    "prediction": "Prediction",
                    "confidence_pct": "Confidence (%)",
                    "normal_probability_pct": "NORMAL (%)",
                    "pneumonia_probability_pct": "PNEUMONIA (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False)

        st.download_button(
            "Export complete history as CSV",
            data=csv_data,
            file_name="medvision_analysis_history.csv",
            mime="text/csv",
            type="primary",
        )

        st.caption(
            "History is stored in session state and is not a permanent "
            "database. It may disappear when the session resets."
        )

        if st.button("Clear session history"):
            st.session_state.medvision_history = []
            st.session_state.medvision_result = None
            st.rerun()

    else:
        st.info(
            "No predictions have been recorded yet. "
            "Analyze an image to populate this page."
        )


# ============================================================
# PAGE 4 — MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.subheader("Model insights and evaluation")
    st.write(
        "Review the model configuration, saved evaluation metrics, "
        "and dataset-quality documentation."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Architecture", "DenseNet121")
    col2.metric("Input resolution", "224 × 224")
    col3.metric("Output classes", "2")

    st.markdown("### Preprocessing pipeline")

    preprocessing_steps = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4",
            ],
            "Operation": [
                "Image loading",
                "RGB conversion",
                "Resize",
                "Normalization",
            ],
            "Configuration": [
                "PIL image",
                "3-channel RGB",
                "224 × 224 pixels",
                "ImageNet mean and standard deviation",
            ],
        }
    )

    st.dataframe(
        preprocessing_steps,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Saved evaluation results")

    report = load_json_report(METRICS_PATH)

    if report is not None:
        st.json(report)

        st.download_button(
            "Download evaluation metrics JSON",
            data=json.dumps(report, indent=2),
            file_name="medvision_evaluation_metrics.json",
            mime="application/json",
        )
    else:
        st.warning(
            "Evaluation metrics were not found. Ensure the file "
            "reports/evaluation/metrics.json exists in your project."
        )

    st.markdown("### Dataset quality audit")

    if AUDIT_PATH.is_file():
        try:
            audit_text = AUDIT_PATH.read_text(encoding="utf-8")

            with st.expander("View dataset audit report"):
                st.markdown(audit_text)

            st.download_button(
                "Download dataset audit",
                data=audit_text,
                file_name="medvision_dataset_quality_audit.md",
                mime="text/markdown",
            )
        except OSError:
            st.warning("The dataset audit file could not be read.")
    else:
        st.info(
            "No dataset audit report was found at the configured path."
        )

    st.divider()

    st.markdown("### Important evaluation considerations")

    st.write(
        "Overall accuracy alone does not fully describe a classifier. "
        "Review per-class precision, recall, F1-score, confusion matrix, "
        "ROC-AUC, class imbalance, and the independence of the test set."
    )

    st.warning(
        "Softmax probabilities can be overconfident. Evaluation results "
        "on a dataset do not establish clinical safety or generalization "
        "to every hospital, scanner, or patient population."
    )


# ============================================================
# PAGE 5 — ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.subheader("About MedVision AI")

    st.markdown(
        """
        MedVision AI is an academic Deep Learning project that
        demonstrates binary classification of chest X-ray images
        using a DenseNet121 convolutional neural network.
        """
    )

    st.markdown("### Project objectives")

    st.markdown(
        """
        - Build an image-classification pipeline for chest X-rays.
        - Apply consistent image preprocessing.
        - Load a trained DenseNet121 model for inference.
        - Display class probabilities and saved evaluation results.
        - Explore dataset quality and model limitations.
        """
    )

    st.markdown("### System workflow")

    st.code(
        """
        Chest X-ray image
                |
                v
        Image validation and RGB conversion
                |
                v
        Resize to 224 x 224
                |
                v
        ImageNet normalization
                |
                v
        Trained DenseNet121 model
                |
                v
        Softmax probabilities
                |
                v
        NORMAL / PNEUMONIA output
                |
                v
        Dashboard and CSV report
        """,
        language="text",
    )

    st.markdown("### Technology stack")

    st.dataframe(
        pd.DataFrame(
            {
                "Component": [
                    "Deep learning framework",
                    "Model architecture",
                    "Image processing",
                    "User interface",
                    "Data analysis",
                ],
                "Technology": [
                    "PyTorch",
                    "DenseNet121",
                    "Pillow / torchvision",
                    "Streamlit",
                    "Pandas",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.error(
        "Medical safety notice: this application is an educational "
        "demonstration, not a validated diagnostic system. Its outputs "
        "must not be used for medical decisions."
    )


# ------------------------------------------------------------
# 8. FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "MedVision AI · Deep Learning Mini Project · "
    "Educational and research use only"
)
