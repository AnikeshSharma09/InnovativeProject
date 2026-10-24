"""
Streamlit Web Application for Real-Time Motion-Aware Sign Language Detection & Recognition.

Features:
- Live Webcam with MediaPipe 21 Landmark Mesh Visualization
- Adaptive 1€ (One-Euro) Anti-Shiver & Tremor Landmark Stabilization
- Direct Real-Time Sign-Driven Predicted Sentence & Word Recognition
- Automatic Persistent Logging to File (`data/predicted_signs.txt`)
- Real-Time Closed-Captions HUD (Zoom / Teams Screen-Share Compatible)
- Multi-Dataset Explorer (ASL Alphabet 39 classes, WLASL 2000 words, MS-ASL)
- High-Precision Spatial Random Forest (99.3%) + PyTorch Deep Motion LSTM
- Offline Text-to-Speech (pyttsx3)
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
    draw_dual_styled_landmarks,
    robust_extract_landmarks_from_bgr_frame,
    TOTAL_MOTION_FEATURES
)
from utils.labels import format_label_for_display
from utils.speech import speak_text
from utils.sentence_engine import EnterpriseSentenceEngine
from utils.prediction_logger import PredictionLogger
try:
    from tutorial_view import render_tutorial_tab
except (ModuleNotFoundError, ImportError):
    from app.tutorial_view import render_tutorial_tab
import torch

# Page Configuration
st.set_page_config(
    page_title="Motion-Aware Sign Language Recognition | Real-Time AI",
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
        font-size: 1.02rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .text-box {
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        border-radius: 10px;
        padding: 1.1rem;
        font-size: 1.5rem;
        font-weight: 700;
        min-height: 75px;
        color: #0F172A;
        word-wrap: break-word;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    .file-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: #166534;
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
    .badge-shiver {
        background-color: #F3E8FF;
        color: #6B21A8;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }
    .badge-phrase {
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 12px;
        display: inline-block;
        border: 1px solid #EAB308;
    }
    .cc-hud {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 6px solid #38BDF8;
        margin-top: 10px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .cc-speaker {
        color: #38BDF8;
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .cc-content {
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 4px;
        color: #FFFFFF;
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
if "sentence_engine" not in st.session_state:
    st.session_state.sentence_engine = EnterpriseSentenceEngine()
if "pred_logger" not in st.session_state:
    st.session_state.pred_logger = PredictionLogger()
if "speaker_name" not in st.session_state:
    st.session_state.speaker_name = "Alex (ASL Presenter)"


@st.cache_resource(show_spinner="Loading Deep Learning LSTM model & Landmark Stabilizers...")
def load_predictor():
    """Caches the SignPredictor instance to avoid reloading models on every rerun."""
    try:
        return SignPredictor(
            lstm_model_path="models/sign_lstm.pth",
            encoder_path="models/label_encoder.pkl",
            rf_model_path="models/sign_model.pkl",
            sequence_length=20,
            enable_stabilization=True
        )
    except FileNotFoundError:
        return None


def get_motion_badge_html(motion_state: str, is_uncertain: bool, is_shivering: bool) -> str:
    """Returns styled HTML badge for hand kinetic motion state & anti-shiver status."""
    if is_uncertain:
        return '<span class="badge-uncertain">⚠️ UNCERTAIN / TRANSITION</span>'
    elif is_shivering:
        return '<span class="badge-shiver">🛡️ SHIVER DAMPENED (STABILIZED)</span>'
    elif motion_state == "ACTIVE MOTION":
        return '<span class="badge-active">⚡ ACTIVE MOTION</span>'
    elif motion_state == "FAST MOTION":
        return '<span class="badge-fast">🚀 FAST TRAJECTORY</span>'
    else:
        return '<span class="badge-static">📍 STATIC HOLD</span>'


def main():
    st.markdown('<div class="main-header">🤟 Real-Time Motion-Aware Sign Language AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Direct Sign-to-Word & Sentence Prediction • 1€ Anti-Tremor Filter • Persistent File Logger • Multi-Dataset</div>', unsafe_allow_html=True)

    predictor = load_predictor()
    engine: EnterpriseSentenceEngine = st.session_state.sentence_engine
    logger: PredictionLogger = st.session_state.pred_logger

    # Sidebar: System configuration and details
    with st.sidebar:
        st.header("⚙️ System Control")
        
        if predictor is not None:
            if predictor.lstm_model is not None and predictor.rf_model is not None:
                st.success("🟢 Dual Base Engines: **PyTorch LSTM + Random Forest**")
            elif predictor.lstm_model is not None:
                st.success("🟢 Deep Learning: **PyTorch LSTM**")
            else:
                st.info("🟡 Baseline Model: **Random Forest**")
            
            if hasattr(predictor, 'wlasl_model') and predictor.wlasl_model is not None:
                st.success(f"🎬 WLASL Word Engine: **{len(predictor.wlasl_classes)} Signs Active**")

            st.info(f"📊 Base Alphabet Classes ({len(predictor.classes)}): {', '.join(predictor.classes[:6])}...")
            if hasattr(predictor, 'wlasl_classes') and predictor.wlasl_classes:
                with st.expander(f"📚 View {len(predictor.wlasl_classes)} WLASL Vocabulary Signs"):
                    st.caption(", ".join(sorted([w.upper() for w in predictor.wlasl_classes])))

            # Engine selector
            engine_choice = st.selectbox(
                "Inference Engine Mode",
                options=[
                    "Auto (Hybrid Motion/Static)",
                    "Hybrid (Letters + Words + Sentences)",
                    "WLASL Word-Level Signs (New Dataset)",
                    "Random Forest (Alphabet & Digits 99.3%)",
                    "Deep LSTM (Motion Trajectories)"
                ],
                index=0
            )
            mode_map = {
                "Auto (Hybrid Motion/Static)": "auto",
                "Hybrid (Letters + Words + Sentences)": "hybrid",
                "WLASL Word-Level Signs (New Dataset)": "wlasl",
                "Random Forest (Alphabet & Digits 99.3%)": "rf",
                "Deep LSTM (Motion Trajectories)": "lstm"
            }
            predictor.engine_mode = mode_map[engine_choice]
        else:
            st.error("🔴 Model Not Found!")
            st.warning("Please train the model first:\n```bash\npython training/train_lstm.py\n```")

        st.markdown("---")
        st.subheader("🛡️ Anti-Tremor & Shiver Control")
        enable_stabilization = st.checkbox(
            "Adaptive 1€ Anti-Tremor Filter",
            value=True,
            help="Filters high-frequency hand tremors and shivering while preserving fast transitions with zero lag."
        )
        min_cutoff = st.slider(
            "Tremor Smoothing (Min Cutoff fc_min)",
            min_value=0.1,
            max_value=3.0,
            value=0.8,
            step=0.1,
            help="Lower values dampen stronger resting hand tremors."
        )
        beta_val = st.slider(
            "Dynamic Responsiveness (Beta)",
            min_value=0.001,
            max_value=0.08,
            value=0.015,
            step=0.002,
            help="Higher values eliminate latency during rapid hand repositioning."
        )
        enable_deadband = st.checkbox(
            "Tremor Kinetic Deadband",
            value=True,
            help="Clamps false kinetic velocity spikes caused by hand tremors, keeping static signs rock-solid."
        )

        if predictor is not None:
            predictor.enable_stabilization = enable_stabilization
            predictor.landmark_stabilizer.min_cutoff = min_cutoff
            predictor.landmark_stabilizer.beta = beta_val
            predictor.tremor_deadband = enable_deadband

        st.markdown("---")
        st.subheader("📁 Prediction Storage File")
        st.code(str(logger.file_path.resolve()), language="bash")
        
        st.markdown("---")
        st.subheader("🎥 Camera Settings")
        camera_id = st.selectbox("Select Webcam Device Index", options=[0, 1, 2], index=0)

        st.markdown("---")
        if st.button("🧹 Reset Sentence & Sequence Buffer"):
            if predictor:
                predictor.reset_buffer()
            engine.reset()
            st.session_state.accumulated_text = ""
            st.success("Buffers reset.")

    # Main Tabs: Cleaned to 4 focused tabs
    tab_webcam, tab_upload, tab_metrics, tab_tutorial = st.tabs([
        "📹 Live Motion Webcam & Prediction Logger",
        "🖼️ Static Image Analysis & Dataset Explorer",
        "📊 Model Performance & Metrics",
        "📖 Interactive Sign Tutorial & Practice Guide"
    ])

    # =========================================================================
    # TAB 1: LIVE WEBCAM RECOGNITION & PREDICTION LOGGER
    # =========================================================================
    with tab_webcam:
        col_video, col_info = st.columns([3, 2])

        with col_info:
            st.subheader("🎯 Real-Time Detection Status")
            metric_hands_box = st.empty()
            metric_pred_box = st.empty()
            metric_prob_box = st.empty()
            metric_motion_box = st.empty()
            metric_shiver_box = st.empty()
            metric_phrase_box = st.empty()
            metric_buffer_box = st.empty()

            st.markdown("---")
            st.subheader("📝 Predicted Sentence & Words Through Signs")
            text_display_box = st.empty()
            
            current_sentence = engine.get_full_sentence(include_current=True) or "(Predicted sign sentence will form here as you sign...)"
            text_display_box.markdown(
                f'<div class="text-box">{current_sentence}</div>',
                unsafe_allow_html=True
            )

            # Persistent File Storage Info Card
            st.markdown(
                f'<div class="file-card">'
                f'💾 <b>Persistent File Storage Active:</b><br>'
                f'<code>{logger.file_path.as_posix()}</code>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Manual Sentence Controls
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("➕ Add Current", use_container_width=True):
                    if st.session_state.last_stable_char:
                        engine.add_character(st.session_state.last_stable_char)
                        text_display_box.markdown(
                            f'<div class="text-box">{engine.get_full_sentence()}</div>',
                            unsafe_allow_html=True
                        )
            with c2:
                if st.button("␣ Space / Commit", use_container_width=True):
                    committed = engine.commit_current_word(autocorrect=False)
                    if committed:
                        logger.log_prediction(committed, tag="WORD")
                    text_display_box.markdown(
                        f'<div class="text-box">{engine.get_full_sentence()}</div>',
                        unsafe_allow_html=True
                    )
            with c3:
                if st.button("⌫ Backspace", use_container_width=True):
                    engine.backspace()
                    text_display_box.markdown(
                        f'<div class="text-box">{engine.get_full_sentence()}</div>',
                        unsafe_allow_html=True
                    )

            c4, c5 = st.columns(2)
            with c4:
                if st.button("💾 Save Sentence Now", use_container_width=True):
                    full_txt = engine.get_full_sentence(include_current=False)
                    if full_txt.strip():
                        logger.log_prediction(full_txt, tag="SENTENCE")
                        st.success(f"Saved: '{full_txt}' to file!")
                    else:
                        st.warning("Current sentence is empty.")
            with c5:
                if st.button("🔊 Speak Text", use_container_width=True, type="primary"):
                    full_txt = engine.get_full_sentence(include_current=False)
                    if full_txt.strip():
                        speak_text(full_txt)
                        st.toast(f"Speaking: '{full_txt}'")
                    else:
                        st.warning("Sentence is empty! Perform signs first.")

            # File Management Section
            st.markdown("---")
            st.markdown("#### 📄 Stored Predictions File Actions")
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                file_text = logger.get_file_content()
                st.download_button(
                    "📥 Download Log File (.txt)",
                    data=file_text,
                    file_name="predicted_signs.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with f_col2:
                if st.button("🗑️ Clear Log File", use_container_width=True):
                    logger.clear()
                    st.toast("Predictions log file cleared.")

            with st.expander("📜 View File Contents (`data/predicted_signs.txt`)", expanded=False):
                st.text_area("Recent File Log", value=logger.get_file_content(), height=180)

        with col_video:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                start_btn = st.button("▶️ Start Webcam", use_container_width=True, type="primary")
            with btn_col2:
                stop_btn = st.button("⏹️ Stop Webcam", use_container_width=True)

            # Live Closed-Captions HUD Banner
            cc_placeholder = st.empty()
            cc_placeholder.markdown(
                f'<div class="cc-hud">'
                f'<div class="cc-speaker">🎙️ LIVE ASL PREDICTED SIGNS</div>'
                f'<div class="cc-content">{engine.get_full_sentence() or "Awaiting sign language input..."}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

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
                        last_rendered_phrase = ""

                        while st.session_state.webcam_running:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("Camera disconnected or frame dropped.")
                                break

                            frame = cv2.flip(frame, 1)
                            h, w, _ = frame.shape
                            pred = predictor.predict_frame(frame)

                            # Draw landmarks on frame (Dual-Hand or Single-Hand)
                            if pred.hand_detected:
                                if pred.dual_result is not None:
                                    draw_dual_styled_landmarks(frame, pred.dual_result)
                                elif pred.hand_landmarks_proto:
                                    draw_styled_landmarks(frame, pred.hand_landmarks_proto)

                            # Check for whole-sentence / macro phrase
                            if pred.is_phrase and pred.phrase_text:
                                if pred.phrase_text != last_rendered_phrase:
                                    engine.add_phrase(pred.phrase_text)
                                    logger.log_prediction(pred.phrase_text, tag="2-HAND" if pred.is_two_handed else "PHRASE")
                                    last_rendered_phrase = pred.phrase_text
                                    st.toast(f"{'🙌 2-Hand' if pred.is_two_handed else '🎉'} Sign Detected: {pred.phrase_text}")

                            # Automatic character accumulation debouncing
                            if pred.is_stable and pred.stable_label and not pred.is_uncertain:
                                st.session_state.last_stable_char = pred.stable_label
                                if pred.stable_label == last_auto_char:
                                    auto_commit_counter = 0
                                else:
                                    auto_commit_counter += 1
                                    if auto_commit_counter >= 15:  # ~0.5 seconds stable hold
                                        ch = pred.stable_label
                                        engine.add_character(ch)
                                        last_auto_char = ch
                                        auto_commit_counter = 0
                            else:
                                auto_commit_counter = 0
                                if not pred.hand_detected:
                                    last_auto_char = ""
                                    last_rendered_phrase = ""

                            # Check pause autospace (automatically commits and stores finished word)
                            if engine.current_word and (time.time() - engine.last_char_timestamp) > engine.pause_autospace_seconds:
                                word_saved = engine.commit_current_word(autocorrect=False)
                                if word_saved:
                                    logger.log_prediction(word_saved, tag="WORD")

                            # Render native on-frame Closed Caption bar for screen recording / sharing
                            cv2.rectangle(frame, (0, h - 75), (w, h), (15, 23, 42), -1)
                            disp_cc = engine.get_full_sentence() or "(Listening for sign language...)"
                            cv2.putText(
                                frame,
                                f"[SIGN PREDICTION]: {disp_cc[:65]}",
                                (20, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

                            if pred.is_shivering:
                                cv2.putText(
                                    frame,
                                    f"SHIVER STABILIZED (Jitter={pred.jitter_score:.1f}px)",
                                    (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (240, 160, 255),
                                    2,
                                    cv2.LINE_AA
                                )
                            elif pred.is_two_handed:
                                cv2.putText(
                                    frame,
                                    f"2-HANDS: {pred.phrase_text}",
                                    (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.85,
                                    (50, 255, 120),
                                    2,
                                    cv2.LINE_AA
                                )
                            elif pred.is_phrase:
                                cv2.putText(
                                    frame,
                                    f"PHRASE: {pred.phrase_text}",
                                    (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (60, 240, 120),
                                    2,
                                    cv2.LINE_AA
                                )

                            # Update Real-time Status Card
                            if pred.num_hands == 2:
                                metric_hands_box.markdown("🙌 **Mode:** `Dual-Hand (2 Hands Active)`")
                            elif pred.num_hands == 1:
                                metric_hands_box.markdown("✋ **Mode:** `Single Hand Active`")
                            else:
                                metric_hands_box.markdown("⚪ **Mode:** `Awaiting Hand Presence`")

                            disp_sign = pred.display_label if pred.hand_detected else "NO HAND DETECTED"
                            metric_pred_box.markdown(f"### Predicted Sign: **{disp_sign}**")
                            metric_prob_box.markdown(
                                f"**Confidence:** `{pred.confidence_score:.1f}%` ({pred.confidence_level}) | **Margin:** `{pred.margin:.2f}`\n\n"
                                f"<small style='color: #64748b;'>{pred.status_message}</small>",
                                unsafe_allow_html=True
                            )
                            badge_html = get_motion_badge_html(pred.motion_state, pred.is_uncertain, pred.is_shivering)
                            metric_motion_box.markdown(f"**Kinetic State:** {badge_html} (Energy: `{pred.kinetic_energy:.3f}`)", unsafe_allow_html=True)
                            
                            if pred.is_shivering:
                                metric_shiver_box.markdown(f"🛡️ **Anti-Shiver Status:** Active (`Jitter: {pred.jitter_score:.2f}px` filtered)")
                            else:
                                metric_shiver_box.markdown(f"🛡️ **Anti-Shiver Status:** Monitoring (`Jitter: {pred.jitter_score:.2f}px`)")

                            if pred.is_phrase and pred.phrase_text:
                                icon = "🙌 2-HAND" if pred.is_two_handed else "✨ PHRASE"
                                metric_phrase_box.markdown(f'<div class="badge-phrase">{icon}: {pred.phrase_text}</div>', unsafe_allow_html=True)
                            else:
                                metric_phrase_box.empty()

                            buffer_ratio = pred.buffer_fill / max(1, predictor.sequence_length)
                            metric_buffer_box.markdown(f"**LSTM Temporal Window:** `[{pred.buffer_fill}/{predictor.sequence_length} frames buffered]` ({buffer_ratio:.0%})")

                            # Update Closed Captions HUD
                            cc_placeholder.markdown(
                                f'<div class="cc-hud">'
                                f'<div class="cc-speaker">🎙️ LIVE ASL PREDICTED SIGNS</div>'
                                f'<div class="cc-content">{disp_cc}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            # Convert BGR to RGB for Streamlit rendering
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        cap.release()
            else:
                video_placeholder.info("Webcam is idle. Click **'Start Webcam'** above to begin live motion detection and captions.")

    # =========================================================================
    # TAB 2: STATIC IMAGE ANALYSIS & DATASET EXPLORER
    # =========================================================================
    with tab_upload:
        st.subheader("🖼️ High-Precision Gesture Image Analysis & Multi-Dataset Explorer")
        st.markdown(
            "Test hand landmark detection, 1€ anti-shiver stabilization, and sign classification on any uploaded image "
            "or explore across available datasets (**ASL Alphabet**, **WLASL 2000 Words**, and **MS-ASL**)."
        )

        dataset_dir = Path("dataset")
        if not dataset_dir.exists():
            dataset_dir = Path("archive/asl_dataset")
        if not dataset_dir.exists():
            dataset_dir = Path("archive/asl_alphabet_test/asl_alphabet_test")

        wlasl_dir = Path("archive/wlasl")

        tab2_col_mode, tab2_col_model = st.columns(2)
        with tab2_col_mode:
            options_source = ["📁 Upload Image from Computer", "📦 Select Sample from ASL Alphabet (39 Classes: 0-9, A-Z)"]
            if wlasl_dir.exists():
                options_source.append("📚 Explore WLASL Word Dataset (2,000 Sign Words)")

            image_source = st.radio(
                "Select Image / Dataset Source:",
                options=options_source,
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
                key="tab3_file_uploader"
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
        elif "WLASL Word Dataset" in image_source:
            wlasl_json_path = wlasl_dir / "WLASL_v0.3.json"
            v_dir = wlasl_dir / "videos"
            trained_words = getattr(predictor, "wlasl_classes", []) if predictor else []
            if not trained_words:
                trained_words = ["accident", "bed", "before", "call", "candy", "change", "cold", "computer", "cool", "cousin", "drink", "go", "help", "later", "man", "no", "play", "shirt", "short", "tall", "thin", "walk", "what", "who", "work", "yes"]

            st.markdown("### 🎬 WLASL Video Sign Recognition & Explorer")
            st.info(f"📚 **WLASL Word Dataset:** {len(trained_words)} trained vocabulary words (from 11,980 video clips in `archive/wlasl/videos/`).")

            wlasl_mode = st.radio(
                "Select Video Source:",
                options=["🎞️ Browse & Test Pre-recorded Vocabulary Video Clips", "📤 Upload Custom Sign Video (.MP4 / .MOV / .AVI)"],
                horizontal=True
            )

            if wlasl_mode == "🎞️ Browse & Test Pre-recorded Vocabulary Video Clips":
                c_w1, c_w2 = st.columns([1, 1])
                with c_w1:
                    selected_word = st.selectbox(
                        "1. Choose Vocabulary Word to Test:",
                        options=sorted(trained_words),
                        index=0,
                        format_func=lambda w: f"{w.upper()} ({w})"
                    )

                # Look up available video files for selected_word
                matched_video_paths = []
                if wlasl_json_path.exists():
                    try:
                        import json
                        with open(wlasl_json_path, "r", encoding="utf-8") as f:
                            wlasl_data = json.load(f)
                        for item in wlasl_data:
                            if item["gloss"].lower() == selected_word.lower():
                                for inst in item["instances"]:
                                    vid_id = inst["video_id"]
                                    p = v_dir / f"{vid_id}.mp4"
                                    if p.exists() and p.stat().st_size > 1000:
                                        matched_video_paths.append((vid_id, p))
                                break
                    except Exception as e:
                        st.warning(f"Could not load WLASL index: {e}")

                with c_w2:
                    if matched_video_paths:
                        chosen_vid_tuple = st.selectbox(
                            f"2. Choose Sample Video ({len(matched_video_paths)} available):",
                            options=matched_video_paths,
                            format_func=lambda tup: f"Video ID: {tup[0]}.mp4 ({tup[1].stat().st_size // 1024} KB)"
                        )
                        test_video_path = chosen_vid_tuple[1]
                    else:
                        st.warning(f"No video files found on disk for '{selected_word}'.")
                        test_video_path = None

                if test_video_path and test_video_path.exists():
                    v_col1, v_col2 = st.columns([1, 1])
                    with v_col1:
                        st.subheader("🎥 Source Video Clip")
                        st.video(str(test_video_path))
                        run_btn = st.button("⚡ Run AI Sign Language Recognition on this Clip", type="primary", use_container_width=True)

                    if run_btn and predictor is not None:
                        with st.spinner("Extracting 20-frame MediaPipe kinematics and evaluating WLASL Model..."):
                            v_res = predictor.predict_video(test_video_path)

                        with v_col2:
                            st.subheader("🔍 Prediction Results")
                            if v_res["success"]:
                                pred_word = v_res["predicted_word"]
                                conf = v_res["confidence_score"]
                                is_match = (pred_word.lower() == selected_word.lower())

                                if is_match:
                                    st.success(f"🎯 **PERFECT MATCH:** Predicted **{pred_word}** ({conf:.1f}% confidence)")
                                else:
                                    st.info(f"🤖 **Top Prediction:** **{pred_word}** ({conf:.1f}% confidence)")

                                st.markdown("##### 📊 Top 3 Candidate Predictions")
                                for c_idx, (cand_word, cand_prob) in enumerate(v_res["top3_candidates"]):
                                    st.write(f"**#{c_idx+1}: {cand_word}** ({cand_prob:.1%})")
                                    st.progress(min(1.0, float(cand_prob)))

                                if v_res["annotated_keyframe"] is not None:
                                    st.image(
                                        cv2.cvtColor(v_res["annotated_keyframe"], cv2.COLOR_BGR2RGB),
                                        caption="Extracted Keyframe with 21 MediaPipe Hand Landmarks",
                                        use_container_width=True
                                    )
                            else:
                                st.error(f"Prediction failed: {v_res['error_message']}")

            else:
                # Custom Video Upload
                uploaded_video = st.file_uploader(
                    "Choose an MP4, MOV, or AVI video clip to recognize:",
                    type=["mp4", "mov", "avi"],
                    key="custom_video_uploader"
                )
                if uploaded_video is not None:
                    temp_dir = Path("data")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = temp_dir / "temp_eval_video.mp4"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_video.getvalue())

                    uv_col1, uv_col2 = st.columns([1, 1])
                    with uv_col1:
                        st.subheader("🎥 Uploaded Video")
                        st.video(str(temp_path))
                        run_up_btn = st.button("⚡ Run AI Sign Language Recognition on Uploaded Video", type="primary", use_container_width=True)

                    if run_up_btn and predictor is not None:
                        with st.spinner("Extracting kinematics and predicting word gesture..."):
                            v_res = predictor.predict_video(temp_path)

                        with uv_col2:
                            st.subheader("🔍 Prediction Results")
                            if v_res["success"]:
                                pred_word = v_res["predicted_word"]
                                conf = v_res["confidence_score"]
                                st.success(f"🤖 **Predicted Sign Word:** **{pred_word}** ({conf:.1f}%)")

                                st.markdown("##### 📊 Top 3 Candidates")
                                for c_idx, (cand_word, cand_prob) in enumerate(v_res["top3_candidates"]):
                                    st.write(f"**#{c_idx+1}: {cand_word}** ({cand_prob:.1%})")
                                    st.progress(min(1.0, float(cand_prob)))

                                if v_res["annotated_keyframe"] is not None:
                                    st.image(
                                        cv2.cvtColor(v_res["annotated_keyframe"], cv2.COLOR_BGR2RGB),
                                        caption="Extracted Keyframe with 21 MediaPipe Hand Landmarks",
                                        use_container_width=True
                                    )
                            else:
                                st.error(f"Prediction failed: {v_res['error_message']}")
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

                if st.button("➕ Append to Predicted Sentence & Log to File", type="primary"):
                    char = pred.predicted_label
                    engine.add_character(char)
                    logger.log_prediction(char, tag="SIGN")
                    st.success(f"Appended & logged '{char}'! Current sentence: {engine.get_full_sentence()}")
            else:
                st.warning(f"⚠️ {pred.status_message}")

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

    # =========================================================================
    # TAB 4: INTERACTIVE SIGN TUTORIAL & PRACTICE GUIDE
    # =========================================================================
    with tab_tutorial:
        render_tutorial_tab()


if __name__ == "__main__":
    main()
