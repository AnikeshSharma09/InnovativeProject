"""
Streamlit Web Application for Real-Time Motion-Aware Sign Language Detection & Recognition.

Features:
- Live Webcam Feed with MediaPipe Landmark Mesh Visualization
- Real-Time ASL Motion Gesture Prediction (PyTorch Deep Learning LSTM)
- 129-Dimensional Motion Feature Analysis & Kinetic Energy Tracking
- Recognized Text Accumulation & Editing (Add, Space, Backspace, Clear)
- Offline Text-to-Speech Narration (pyttsx3)
- Image Upload with Temporal Motion Projection
- Model Metrics & Confusion Matrix Inspection Dashboard
"""
import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.predictor import SignPredictor
from utils.landmarks_util import (
    draw_styled_landmarks,
    robust_extract_landmarks_from_bgr_frame,
    TOTAL_MOTION_FEATURES
)
from utils.labels import format_label_for_display
from utils.speech import speak_text
import torch

# Page Configuration
st.set_page_config(
    page_title="Motion-Aware Sign Language Recognition (LSTM)",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .text-box {
        background-color: #F1F5F9;
        border: 2px solid #CBD5E1;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.4rem;
        font-weight: 600;
        min-height: 70px;
        color: #0F172A;
        word-wrap: break-word;
    }
    .badge-active {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }
    .badge-static {
        background-color: #DBEAFE;
        color: #1E40AF;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }
    .badge-fast {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }
    .badge-uncertain {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "accumulated_text" not in st.session_state:
    st.session_state.accumulated_text = ""
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "last_stable_char" not in st.session_state:
    st.session_state.last_stable_char = ""


@st.cache_resource(show_spinner="Loading Deep Learning LSTM model...")
def load_predictor():
    """Caches the SignPredictor instance to avoid reloading models on every rerun."""
    try:
        return SignPredictor(
            lstm_model_path="models/sign_lstm.pth",
            encoder_path="models/label_encoder.pkl",
            rf_model_path="models/sign_model.pkl",
            sequence_length=20
        )
    except FileNotFoundError:
        return None


def get_motion_badge_html(motion_state: str, is_uncertain: bool) -> str:
    """Returns styled HTML badge for hand kinetic motion state."""
    if is_uncertain:
        return '<span class="badge-uncertain">⚠️ UNCERTAIN / TRANSITION</span>'
    elif motion_state == "ACTIVE MOTION":
        return '<span class="badge-active">⚡ ACTIVE MOTION</span>'
    elif motion_state == "FAST MOTION":
        return '<span class="badge-fast">🚀 FAST TRAJECTORY</span>'
    else:
        return '<span class="badge-static">📍 STATIC HOLD</span>'


def main():
    st.markdown('<div class="main-header">🤟 Real-Time Motion-Aware Sign Language Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Recurrent Architecture (PyTorch LSTM) with 129 Spatiotemporal Motion Features & Kinetic Tracking</div>', unsafe_allow_html=True)

    predictor = load_predictor()

    # Sidebar: System configuration and details
    with st.sidebar:
        st.header("⚙️ System Control")
        
        if predictor is not None:
            if predictor.lstm_model is not None and predictor.rf_model is not None:
                st.success("🟢 Dual Engines: **PyTorch LSTM + Random Forest**")
            elif predictor.lstm_model is not None:
                st.success("🟢 Deep Learning Model: **PyTorch LSTM**")
            else:
                st.info("🟡 Baseline Model: **Random Forest**")
            
            st.info(f"📊 Trained Classes ({len(predictor.classes)}): {', '.join(predictor.classes[:8])}...")

            # Engine selector
            engine_choice = st.selectbox(
                "Inference Engine Mode",
                options=[
                    "Auto (Hybrid Motion/Static)",
                    "Random Forest (High-Precision 99.3%)",
                    "Deep LSTM (Motion Trajectories)"
                ],
                index=0
            )
            mode_map = {
                "Auto (Hybrid Motion/Static)": "auto",
                "Random Forest (High-Precision 99.3%)": "rf",
                "Deep LSTM (Motion Trajectories)": "lstm"
            }
            predictor.engine_mode = mode_map[engine_choice]
        else:
            st.error("🔴 Model Not Found!")
            st.warning("Please train the model first:\n```bash\npython training/train_lstm.py\n```")

        st.markdown("---")
        st.subheader("🎥 Camera Settings")
        camera_id = st.selectbox("Select Webcam Device Index", options=[0, 1, 2], index=0)

        st.markdown("---")
        st.subheader("🧠 Motion Parameters")
        st.caption("• **Spatial Features**: 63 relative coordinates")
        st.caption("• **Velocity Features**: 63 joint speed deltas")
        st.caption("• **Global Trajectory**: 3 wrist deltas (dx, dy, dz)")
        st.caption("• **Total Input**: 129 features per frame")

        if st.button("🧹 Reset Sequence Buffer"):
            if predictor:
                predictor.reset_buffer()
            st.success("Buffer reset.")

    # Main Tabs
    tab_webcam, tab_upload, tab_metrics = st.tabs([
        "📹 Live Motion Webcam Recognition",
        "🖼️ Static Image Analysis",
        "📊 LSTM Model Evaluation & Confusion Matrix"
    ])

    # =========================================================================
    # TAB 1: LIVE WEBCAM RECOGNITION
    # =========================================================================
    with tab_webcam:
        col_video, col_info = st.columns([3, 2])

        with col_info:
            st.subheader("🎯 Real-Time Status")
            metric_pred_box = st.empty()
            metric_prob_box = st.empty()
            metric_motion_box = st.empty()
            metric_buffer_box = st.empty()

            st.markdown("---")
            st.subheader("📝 Recognized Sentence")
            text_display_box = st.empty()
            text_display_box.markdown(
                f'<div class="text-box">{st.session_state.accumulated_text or "(Recognized text will appear here...)"}</div>',
                unsafe_allow_html=True
            )

            # Sentence Builder Controls
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("➕ Add Current", use_container_width=True):
                    if st.session_state.last_stable_char:
                        char = st.session_state.last_stable_char
                        if char == "space":
                            st.session_state.accumulated_text += " "
                        elif char == "del":
                            st.session_state.accumulated_text = st.session_state.accumulated_text[:-1]
                        elif char != "nothing":
                            st.session_state.accumulated_text += char
                        text_display_box.markdown(
                            f'<div class="text-box">{st.session_state.accumulated_text}</div>',
                            unsafe_allow_html=True
                        )
            with c2:
                if st.button("␣ Space", use_container_width=True):
                    st.session_state.accumulated_text += " "
                    text_display_box.markdown(
                        f'<div class="text-box">{st.session_state.accumulated_text}</div>',
                        unsafe_allow_html=True
                    )
            with c3:
                if st.button("⌫ Backspace", use_container_width=True):
                    st.session_state.accumulated_text = st.session_state.accumulated_text[:-1]
                    text_display_box.markdown(
                        f'<div class="text-box">{st.session_state.accumulated_text}</div>',
                        unsafe_allow_html=True
                    )

            c4, c5 = st.columns(2)
            with c4:
                if st.button("🗑️ Clear Text", use_container_width=True):
                    st.session_state.accumulated_text = ""
                    text_display_box.markdown(
                        f'<div class="text-box">(Cleared)</div>',
                        unsafe_allow_html=True
                    )
            with c5:
                if st.button("🔊 Speak Text", use_container_width=True, type="primary"):
                    if st.session_state.accumulated_text.strip():
                        speak_text(st.session_state.accumulated_text)
                        st.toast(f"Speaking: '{st.session_state.accumulated_text}'")
                    else:
                        st.warning("Sentence buffer is empty! Perform signs first.")

        with col_video:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                start_btn = st.button("▶️ Start Webcam", use_container_width=True, type="primary")
            with btn_col2:
                stop_btn = st.button("⏹️ Stop Webcam", use_container_width=True)

            video_placeholder = st.empty()

            if start_btn:
                st.session_state.webcam_running = True
            if stop_btn:
                st.session_state.webcam_running = False

            if st.session_state.webcam_running:
                if predictor is None:
                    st.error("Cannot start webcam without a trained LSTM model.")
                else:
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        st.error(f"Could not open webcam index {camera_id}.")
                        st.session_state.webcam_running = False
                    else:
                        auto_commit_counter = 0
                        last_auto_char = ""

                        while st.session_state.webcam_running:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("Camera disconnected or frame dropped.")
                                break

                            frame = cv2.flip(frame, 1)
                            pred = predictor.predict_frame(frame)

                            # Draw landmarks on frame
                            if pred.hand_detected and pred.hand_landmarks_proto:
                                draw_styled_landmarks(frame, pred.hand_landmarks_proto)

                            # Handle automatic stable character debouncing
                            if pred.is_stable and pred.stable_label and not pred.is_uncertain:
                                st.session_state.last_stable_char = pred.stable_label
                                if pred.stable_label == last_auto_char:
                                    auto_commit_counter = 0
                                else:
                                    auto_commit_counter += 1
                                    if auto_commit_counter >= 15:  # ~0.5 seconds stable hold
                                        ch = pred.stable_label
                                        if ch == "space":
                                            st.session_state.accumulated_text += " "
                                        elif ch == "del":
                                            st.session_state.accumulated_text = st.session_state.accumulated_text[:-1]
                                        elif ch != "nothing":
                                            st.session_state.accumulated_text += ch
                                        last_auto_char = ch
                                        auto_commit_counter = 0
                                        text_display_box.markdown(
                                            f'<div class="text-box">{st.session_state.accumulated_text}</div>',
                                            unsafe_allow_html=True
                                        )
                            else:
                                auto_commit_counter = 0
                                if not pred.hand_detected:
                                    last_auto_char = ""

                            # Update Real-time Status Card
                            disp_sign = pred.display_label if pred.hand_detected else "NO HAND DETECTED"
                            metric_pred_box.markdown(f"### Predicted Sign: **{disp_sign}**")
                            metric_prob_box.markdown(
                                f"**Confidence:** `{pred.confidence_score:.1f}%` ({pred.confidence_level}) | **Margin:** `{pred.margin:.2f}`\n\n"
                                f"<small style='color: #64748b;'>{pred.status_message}</small>",
                                unsafe_allow_html=True
                            )
                            badge_html = get_motion_badge_html(pred.motion_state, pred.is_uncertain)
                            metric_motion_box.markdown(f"**Kinetic State:** {badge_html} (Energy: `{pred.kinetic_energy:.3f}`)", unsafe_allow_html=True)
                            
                            buffer_ratio = pred.buffer_fill / max(1, predictor.sequence_length)
                            metric_buffer_box.markdown(f"**LSTM Temporal Window:** `[{pred.buffer_fill}/{predictor.sequence_length} frames buffered]` ({buffer_ratio:.0%})")

                            # Convert BGR to RGB for Streamlit rendering
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        cap.release()
            else:
                video_placeholder.info("Webcam is idle. Click **'Start Webcam'** above to begin live motion detection.")

    # =========================================================================
    # TAB 2: STATIC IMAGE ANALYSIS
    # =========================================================================
    with tab_upload:
        st.subheader("🖼️ High-Precision Gesture Image Analysis")
        st.markdown(
            "Test hand landmark detection and sign classification on any uploaded image "
            "or directly select from the **Kaggle ASL Alphabet test dataset**."
        )

        dataset_dir = Path("dataset")
        if not dataset_dir.exists():
            dataset_dir = Path("archive/asl_dataset")
        if not dataset_dir.exists():
            dataset_dir = Path("archive/asl_alphabet_test/asl_alphabet_test")

        tab2_col_mode, tab2_col_model = st.columns(2)
        with tab2_col_mode:
            image_source = st.radio(
                "Select Image Source:",
                options=["📁 Upload Image from Computer", "📦 Select Sample from Dataset (39 Classes: 0-9, A-Z)"],
                horizontal=True
            )
        with tab2_col_model:
            eval_model_choice = st.selectbox(
                "Evaluation Model:",
                options=["Random Forest (92.4% Accuracy - Recommended for Photos)", "Deep LSTM (80.3% Accuracy)"],
                index=0
            )
            selected_model_type = "rf" if "Random Forest" in eval_model_choice else "lstm"

        bgr_img = None
        source_label = ""

        if image_source == "📁 Upload Image from Computer":
            uploaded_file = st.file_uploader(
                "Choose an image (JPG, PNG, WEBP, BMP)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="tab2_file_uploader"
            )
            if uploaded_file is not None:
                try:
                    bytes_data = uploaded_file.getvalue()
                    if bytes_data and len(bytes_data) > 0:
                        file_bytes = np.frombuffer(bytes_data, dtype=np.uint8)
                        bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        source_label = f"Uploaded File: {uploaded_file.name}"
                        if bgr_img is None:
                            st.error("Could not decode uploaded image format. Please try another image file.")
                except Exception as ex:
                    st.error(f"Error reading uploaded file: {ex}")
        else:
            if dataset_dir.exists() and dataset_dir.is_dir():
                subdirs = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name.lower() != "asl_dataset"])
                if subdirs:
                    c1, c2 = st.columns(2)
                    with c1:
                        default_idx = subdirs.index("A") if "A" in subdirs else 0
                        chosen_class = st.selectbox("1. Choose Sign Class (0-9, A-Z, del, space, nothing):", options=subdirs, index=default_idx)
                    with c2:
                        class_folder = dataset_dir / chosen_class
                        class_imgs = sorted(list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")))
                        if class_imgs:
                            chosen_img_name = st.selectbox("2. Choose Sample Image:", options=[f.name for f in class_imgs[:30]], index=0)
                            selected_sample_path = class_folder / chosen_img_name
                            bgr_img = cv2.imread(str(selected_sample_path))
                            source_label = f"Dataset Class '{chosen_class}': {chosen_img_name}"
                        else:
                            st.warning(f"No image files in folder {class_folder}.")
                else:
                    st.warning("No class subdirectories found in dataset.")
            else:
                st.warning("Dataset directory not found.")

        if bgr_img is not None and predictor is not None:
            pred, annotated_bgr = predictor.predict_image(bgr_img, model_type=selected_model_type)

            up_col1, up_col2 = st.columns(2)
            with up_col1:
                st.image(
                    cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB),
                    caption=f"Source: {source_label}",
                    use_container_width=True
                )
            with up_col2:
                if annotated_bgr is not None:
                    st.image(
                        cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                        caption="MediaPipe 21 Hand Landmarks & Skeleton Overlay",
                        use_container_width=True
                    )
                else:
                    st.image(
                        cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB),
                        caption="No Landmarks Found",
                        use_container_width=True
                    )

            st.markdown("---")
            st.subheader("🔍 Prediction Results")

            if pred.hand_detected:
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric("Predicted Sign", pred.display_label)
                with r2:
                    st.metric("Probability", f"{pred.raw_probability:.1%}")
                with r3:
                    st.metric("Confidence Margin", f"{pred.margin:.3f}")
                with r4:
                    st.metric("Confidence Score", f"{pred.confidence_score:.1f}%")

                st.success(f"✅ {pred.status_message}")

                # Top-3 predictions distribution
                if hasattr(predictor, 'classes') and predictor.classes:
                    try:
                        if selected_model_type == "rf" and predictor.rf_model is not None:
                            feats_63, _, _, _ = robust_extract_landmarks_from_bgr_frame(bgr_img)
                            if feats_63:
                                all_probs = predictor.rf_model.predict_proba([feats_63])[0]
                                top3_idx = np.argsort(all_probs)[::-1][:3]
                                st.markdown("##### 📊 Top 3 Candidate Predictions")
                                t_cols = st.columns(3)
                                for c_idx, t_idx in enumerate(top3_idx):
                                    with t_cols[c_idx]:
                                        cls_name = predictor.classes[t_idx]
                                        p_val = all_probs[t_idx]
                                        st.write(f"**#{c_idx+1}: {cls_name}** ({p_val:.1%})")
                                        st.progress(float(p_val))
                        elif predictor.lstm_model is not None:
                            feats_63, _, _, _ = robust_extract_landmarks_from_bgr_frame(bgr_img)
                            if feats_63:
                                m_vec = feats_63 + [0.0]*63 + [0.0, 0.0, 0.0]
                                seq = np.array([m_vec]*predictor.sequence_length, dtype=np.float32).reshape(1, predictor.sequence_length, TOTAL_MOTION_FEATURES)
                                probs_t = predictor.lstm_model.predict_probabilities(torch.tensor(seq).to(predictor.device))[0]
                                all_probs = probs_t.cpu().numpy()
                                top3_idx = np.argsort(all_probs)[::-1][:3]
                                st.markdown("##### 📊 Top 3 Candidate Predictions (LSTM)")
                                t_cols = st.columns(3)
                                for c_idx, t_idx in enumerate(top3_idx):
                                    with t_cols[c_idx]:
                                        cls_name = predictor.classes[t_idx]
                                        p_val = all_probs[t_idx]
                                        st.write(f"**#{c_idx+1}: {cls_name}** ({p_val:.1%})")
                                        st.progress(float(p_val))
                    except Exception:
                        pass

                if st.button("➕ Append to Recognized Sentence", type="primary"):
                    char = pred.predicted_label
                    if char == "space":
                        st.session_state.accumulated_text += " "
                    elif char == "del":
                        st.session_state.accumulated_text = st.session_state.accumulated_text[:-1]
                    elif char != "nothing":
                        st.session_state.accumulated_text += char
                    st.success(f"Appended '{char}'! Current sentence: {st.session_state.accumulated_text}")
            else:
                st.warning(f"⚠️ {pred.status_message}")
                st.info("The image was inspected through all 8 robust detection stages (raw, reflection padding, replicate padding, constant padding, and CLAHE contrast enhancement). If this image contains a hand, ensure the palm is not occluded.")

    # =========================================================================
    # TAB 3: MODEL PERFORMANCE & METRICS
    # =========================================================================
    with tab_metrics:
        st.subheader("📊 Motion-Aware LSTM Evaluation Dashboard (Held-Out 20% Test Split)")
        
        eval_json_path = Path("models/evaluation_results.json")
        cm_img_path = Path("models/confusion_matrix.png")

        if eval_json_path.exists():
            with open(eval_json_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            m = metrics_data.get("metrics", {})
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Overall Accuracy", f"{m.get('accuracy', 0.0):.2%}")
            with m2:
                st.metric("Macro Precision", f"{m.get('precision_macro', 0.0):.2%}")
            with m3:
                st.metric("Macro Recall", f"{m.get('recall_macro', 0.0):.2%}")
            with m4:
                st.metric("Macro F1-Score", f"{m.get('f1_score_macro', 0.0):.2%}")

            st.markdown("---")
            m_col1, m_col2 = st.columns([3, 2])

            with m_col1:
                st.markdown("#### 🎯 LSTM Confusion Matrix Heatmap")
                if cm_img_path.exists():
                    st.image(str(cm_img_path), use_container_width=True)
                else:
                    st.info("Confusion matrix image not yet generated. Run 'python training/evaluate_lstm.py'.")

            with m_col2:
                st.markdown("#### 📋 Detailed Per-Class Performance")
                per_class = metrics_data.get("per_class_report", {})
                if per_class:
                    rows = []
                    for k, v in per_class.items():
                        if isinstance(v, dict):
                            rows.append({
                                "Class": k,
                                "Precision": f"{v.get('precision', 0):.2f}",
                                "Recall": f"{v.get('recall', 0):.2f}",
                                "F1-Score": f"{v.get('f1-score', 0):.2f}",
                                "Support": int(v.get("support", 0))
                            })
                    df_report = pd.DataFrame(rows)
                    st.dataframe(df_report, use_container_width=True, height=450)
        else:
            st.info("No evaluation results found yet. Run `python training/evaluate_lstm.py` to evaluate your trained LSTM on the 20% test dataset.")




if __name__ == "__main__":
    main()
