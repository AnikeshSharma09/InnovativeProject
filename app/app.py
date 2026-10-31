"""
Streamlit Web Application for Real-Time Motion-Aware Sign Language Detection & Recognition.

Features:
- Dual Recognition Paradigms:
    1. Continuous Multimodal Stream (MediaPipe Holistic, CNN-BiLSTM-CTC sequence network, non-manual facial cues)
    2. Isolated Motion-Aware Engine (1€ Anti-Shiver filter, PyTorch LSTM + Spatial Random Forest baseline)
- Non-Manual Expression & Linguistic Grammar Translation (Topic-Comment order, WH-Questions, Yes/No Questions, Negation)
- Confidence-Aware Prediction with Top-3 Candidate Ranking and Calibrated User Overrides
- Personal Vocabulary Management (Custom words, names, medical/workplace terminology)
- Accessibility Features (Adjustable caption size, high-contrast display, offline text-to-speech)
- 100% On-Device Private CPU Execution with ONNX Acceleration
- Real-Time Closed-Captions HUD for screen recording and sharing
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
import sys
import warnings
warnings.filterwarnings("ignore")
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
from utils.personalization import PersonalVocabularyManager, UserCorrectionTracker
from utils.asl_grammar import ASLGrammarTranslator
from prediction.continuous_recognizer import ContinuousSignRecognizer
from prediction.camera_manager import (
    open_robust_camera,
    scan_all_cameras,
    release_camera_safely,
)
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
if "detected_cameras" not in st.session_state:
    st.session_state.detected_cameras = None
if "vocab_mgr" not in st.session_state:
    st.session_state.vocab_mgr = PersonalVocabularyManager()
if "correction_tracker" not in st.session_state:
    st.session_state.correction_tracker = UserCorrectionTracker()
if "active_top3" not in st.session_state:
    st.session_state.active_top3 = []
if "caption_font_size" not in st.session_state:
    st.session_state.caption_font_size = "1.5rem"
if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False
if "disable_logging" not in st.session_state:
    st.session_state.disable_logging = False

# High-Contrast & Custom Theme CSS Styling
high_contrast_css = """
    .stApp { background-color: #020617 !important; color: #FFFFFF !important; }
    .text-box { background-color: #0F172A !important; border: 3px solid #FACC15 !important; color: #FACC15 !important; }
    .english-box { background-color: #052E16 !important; border: 3px solid #4ADE80 !important; color: #4ADE80 !important; }
    .cc-hud { background: #000000 !important; border-left: 6px solid #FACC15 !important; border: 2px solid #334155; }
""" if st.session_state.high_contrast else ""

st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        font-size: 1.02rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.2rem;
    }}
    .text-box {{
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        border-radius: 10px;
        padding: 1.1rem;
        font-size: {st.session_state.caption_font_size};
        font-weight: 700;
        min-height: 75px;
        color: #0F172A;
        word-wrap: break-word;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }}
    .english-box {{
        background-color: #F0FDF4;
        border: 2px solid #86EFAC;
        border-radius: 10px;
        padding: 1.1rem;
        font-size: {st.session_state.caption_font_size};
        font-weight: 700;
        min-height: 75px;
        color: #14532D;
        word-wrap: break-word;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }}
    .badge-wh-q {{
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #F59E0B;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
    }}
    .badge-yn-q {{
        background: #E0F2FE;
        color: #075985;
        border: 1px solid #0EA5E9;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
    }}
    .badge-neg {{
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #EF4444;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
    }}
    .badge-statement {{
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #10B981;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
    }}
    .file-card {{
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: #166534;
    }}
    .badge-active {{
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }}
    .badge-static {{
        background-color: #DBEAFE;
        color: #1E40AF;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }}
    .badge-fast {{
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }}
    .badge-uncertain {{
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }}
    .badge-shiver {{
        background-color: #F3E8FF;
        color: #6B21A8;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
    }}
    .badge-phrase {{
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 12px;
        display: inline-block;
        border: 1px solid #EAB308;
    }}
    .cc-hud {{
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 6px solid #38BDF8;
        margin-top: 10px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .cc-speaker {{
        color: #38BDF8;
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .cc-content {{
        font-size: {st.session_state.caption_font_size};
        font-weight: 600;
        margin-top: 4px;
        color: #FFFFFF;
    }}
    {high_contrast_css}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Continuous Multimodal ASL Pipeline...")
def load_continuous_recognizer():
    """Caches the ContinuousSignRecognizer instance."""
    try:
        return ContinuousSignRecognizer()
    except Exception as e:
        print(f"[!] Warning: Could not load ContinuousSignRecognizer: {e}")
        return None


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
    st.markdown('<div class="sub-header">Continuous Stream CTC • Non-Manual Facial Cues • ASL Grammar Translation • 100% On-Device CPU</div>', unsafe_allow_html=True)

    predictor = load_predictor()
    engine: EnterpriseSentenceEngine = st.session_state.sentence_engine
    logger: PredictionLogger = st.session_state.pred_logger
    vocab_mgr: PersonalVocabularyManager = st.session_state.vocab_mgr
    correction_tracker: UserCorrectionTracker = st.session_state.correction_tracker

    # Sidebar: System configuration and details
    with st.sidebar:
        st.header("⚙️ System Control")

        # 1. Paradigm Selector
        st.subheader("🔄 Recognition Paradigm")
        continuous_mode = st.toggle(
            "Continuous ASL Stream (CTC + Non-Manual)",
            value=False,
            help="Activates continuous stream recognition using MediaPipe Holistic, CNN-BiLSTM-CTC sequence network, non-manual facial cues, and ASL-to-English grammar translation."
        )

        c_hand1, c_hand2 = st.columns(2)
        with c_hand1:
            handedness_pref = st.selectbox(
                "Dominant Hand",
                options=["Auto", "Right", "Left"],
                index=0,
                help="Normalizes signing space (left-handed signers are mapped to canonical coordinates)."
            )
        with c_hand2:
            conf_threshold = st.slider(
                "Confidence",
                min_value=0.20,
                max_value=0.85,
                value=0.40,
                step=0.05,
                help="Predictions below this threshold will be flagged as UNCERTAIN."
            )

        if not continuous_mode and predictor is not None:
            if predictor.lstm_model is not None and getattr(predictor, "has_rf_model", False):
                st.success("🟢 Dual Base Engines: **PyTorch LSTM + Random Forest**")
            elif predictor.lstm_model is not None:
                st.success("🟢 Deep Learning: **PyTorch LSTM**")
            else:
                st.info("🟡 Baseline Model: **Random Forest**")

            if hasattr(predictor, 'wlasl_classes') and predictor.wlasl_classes:
                st.success(f"🎬 WLASL Word Engine: **{len(predictor.wlasl_classes)} Signs Active**")

            # Engine selector
            engine_choice = st.selectbox(
                "Isolated Engine Mode",
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
        elif continuous_mode:
            st.success("🌐 Continuous Stream: **CNN + BiLSTM + CTC (ONNX/Torch)**")
            st.info("🎭 Non-Manual: **Eyebrows, Eyes, Mouth & Head Pose**")

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
        # Accessibility Expander
        with st.expander("♿ Accessibility & Visual Settings", expanded=False):
            cap_size_opt = st.selectbox(
                "Caption & Text Font Size",
                options=["Small (1.2rem)", "Medium (1.5rem)", "Large (2.0rem)", "Extra Large (2.5rem)"],
                index=1
            )
            size_map = {
                "Small (1.2rem)": "1.2rem",
                "Medium (1.5rem)": "1.5rem",
                "Large (2.0rem)": "2.0rem",
                "Extra Large (2.5rem)": "2.5rem"
            }
            st.session_state.caption_font_size = size_map[cap_size_opt]

            high_c = st.checkbox(
                "High-Contrast Color Scheme",
                value=st.session_state.high_contrast,
                help="High-contrast dark mode for low-vision signers."
            )
            if high_c != st.session_state.high_contrast:
                st.session_state.high_contrast = high_c
                st.rerun()

            tts_rate = st.slider("Speech Speed (TTS)", 0.6, 1.5, 1.0, 0.1)

        # Privacy Expander
        with st.expander("🔒 Privacy & On-Device Processing", expanded=False):
            st.session_state.disable_logging = st.checkbox(
                "Disable Local File Logging",
                value=st.session_state.disable_logging,
                help="Prevents writing predictions to data/predicted_signs.txt."
            )
            st.success("🛡️ **100% On-Device Processing**\nAll AI models execute locally on CPU. No video, audio, or telemetry is ever transmitted to cloud servers.")

        # Personal Vocabulary Expander
        with st.expander("👤 Personal Vocabulary & Custom Signs", expanded=False):
            new_custom_word = st.text_input("Add Custom Word / Name / Term:", placeholder="e.g. Dr. Roberts, Metformin")
            if st.button("➕ Add Word", use_container_width=True):
                if new_custom_word.strip():
                    vocab_mgr.add_custom_word(new_custom_word.strip())
                    st.toast(f"Added '{new_custom_word.strip()}' to personal dictionary!")
                    st.rerun()
            active_custom = vocab_mgr.get_all_custom_words()
            if active_custom:
                st.caption(f"**Custom Terms ({len(active_custom)}):** " + ", ".join(active_custom[:12]))
                term_to_rm = st.selectbox("Remove Custom Term", options=["(Select)"] + active_custom)
                if st.button("🗑️ Remove") and term_to_rm != "(Select)":
                    vocab_mgr.remove_custom_word(term_to_rm)
                    st.toast(f"Removed '{term_to_rm}'")
                    st.rerun()
            else:
                st.caption("No custom words added yet.")

        st.markdown("---")
        st.subheader("📁 Prediction Storage File")
        st.code(str(logger.file_path.resolve()), language="bash")

        st.markdown("---")
        st.subheader("🎥 Camera & Performance")
        c_cam1, c_cam2 = st.columns([1, 1])
        with c_cam1:
            camera_id = st.selectbox(
                "Webcam Index",
                options=[0, 1, 2, 3],
                index=0,
                help="Try Index 1 if Index 0 is an Infrared/Windows Hello sensor or black screen."
            )
        with c_cam2:
            backend_choice = st.selectbox(
                "Capture Backend",
                options=["Auto", "DirectShow", "MSMF", "Default"],
                index=0,
                help="Auto tests DirectShow then MSMF. DirectShow offers lowest latency on Windows."
            )

        c_fps, c_auto = st.columns([1, 1])
        with c_fps:
            perf_mode = st.selectbox(
                "FPS Mode",
                options=["🚀 Turbo (60 FPS Unlocked)", "⚡ Balanced (30 FPS Standard)", "🔋 Power Saver (20 FPS)"],
                index=0,
                help="Turbo unlocks hardware 60 FPS streaming with zero-lag compressed JPEG encoding."
            )
        with c_auto:
            auto_fallback_enabled = st.checkbox(
                "Auto-Switch Camera",
                value=True,
                help="If selected camera is offline or an IR sensor, auto-switches to the first working camera."
            )

        col_cam_act1, col_cam_act2 = st.columns(2)
        with col_cam_act1:
            if st.button("🔍 Scan Cameras", use_container_width=True):
                with st.spinner("Probing camera hardware..."):
                    detected = scan_all_cameras(max_indices=3)
                    st.session_state.detected_cameras = detected
        with col_cam_act2:
            if st.button("🔄 Force Release", use_container_width=True):
                st.session_state.webcam_running = False
                release_camera_safely(None)
                st.toast("Camera hardware released.")

        if st.session_state.detected_cameras:
            with st.expander("📋 Camera Hardware Scan Results", expanded=True):
                for cam_info in st.session_state.detected_cameras:
                    if cam_info["working"]:
                        st.success(f"**Camera {cam_info['index']}:** {cam_info['note']}")
                    else:
                        st.caption(f"**Camera {cam_info['index']}:** Offline / Busy")

        st.markdown("---")
        if st.button("🧹 Reset Sentence & Sequence Buffer"):
            if predictor:
                predictor.reset_buffer()
            engine.reset()
            st.session_state.accumulated_text = ""
            st.session_state.active_top3 = []
            crec = load_continuous_recognizer()
            if crec:
                crec.reset_session()
            st.success("Buffers reset.")

    # Main Tabs
    tab_webcam, tab_upload, tab_metrics, tab_tutorial = st.tabs([
        "📹 Live Motion Webcam & Dual Output",
        "🖼️ Static Image Analysis & Dataset Explorer",
        "📊 Model Performance & Benchmarks",
        "📖 Interactive Sign Tutorial & Practice Guide"
    ])

    # =========================================================================
    # TAB 1: LIVE WEBCAM RECOGNITION & DUAL OUTPUT
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
            st.subheader("📝 Dual Real-Time ASL & English Output")

            col_gloss_h, col_eng_h = st.columns(2)
            with col_gloss_h:
                st.markdown("**Raw ASL Glosses (Sign Order):**")
                text_display_box = st.empty()
            with col_eng_h:
                st.markdown("**Translated English (Grammar Aware):**")
                english_display_box = st.empty()

            grammar_badge_box = st.empty()

            # Initial box population
            init_asl = engine.get_full_sentence(include_current=True) or "(Raw ASL glosses will appear here...)"
            init_eng = engine.get_translated_sentence() or "(English translated sentence...)"
            text_display_box.markdown(f'<div class="text-box">{init_asl}</div>', unsafe_allow_html=True)
            english_display_box.markdown(f'<div class="english-box">{init_eng}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### 💡 Top-3 Alternative Predictions & User Correction")
            alt_candidates_box = st.empty()

            # Render alternative buttons if available
            if st.session_state.active_top3:
                cols_top3 = st.columns(len(st.session_state.active_top3))
                for idx, (cand_sign, cand_prob) in enumerate(st.session_state.active_top3):
                    with cols_top3[idx]:
                        if st.button(f"👉 {cand_sign} ({cand_prob:.0%})", key=f"btn_alt_{idx}_{cand_sign}", use_container_width=True):
                            correction_tracker.record_correction(
                                predicted=st.session_state.last_stable_char or cand_sign,
                                corrected=cand_sign
                            )
                            if continuous_mode:
                                crec = load_continuous_recognizer()
                                if crec:
                                    crec.add_manual_sign(cand_sign)
                            else:
                                engine.add_phrase(cand_sign)
                            st.toast(f"Calibrated! Selected: '{cand_sign}'")

            # Manual Sentence Controls
            st.markdown("##### 🛠️ Transcript Controls")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("➕ Add Current", use_container_width=True):
                    if st.session_state.last_stable_char:
                        engine.add_character(st.session_state.last_stable_char)
                        if not st.session_state.disable_logging:
                            logger.log_prediction(st.session_state.last_stable_char, tag="SIGN")
                        text_display_box.markdown(
                            f'<div class="text-box">{engine.get_full_sentence(include_current=True)}</div>',
                            unsafe_allow_html=True
                        )
                        english_display_box.markdown(
                            f'<div class="english-box">{engine.get_translated_sentence()}</div>',
                            unsafe_allow_html=True
                        )
            with c2:
                if st.button("⌫ Undo Sign", use_container_width=True):
                    if continuous_mode:
                        crec = load_continuous_recognizer()
                        if crec:
                            removed = crec.undo_last_sign()
                            if removed:
                                st.toast(f"Undid sign: {removed}")
                    else:
                        removed = engine.delete_last_word()
                        if removed:
                            st.toast(f"Removed word: {removed}")
                    text_display_box.markdown(
                        f'<div class="text-box">{engine.get_full_sentence(include_current=True)}</div>',
                        unsafe_allow_html=True
                    )
                    english_display_box.markdown(
                        f'<div class="english-box">{engine.get_translated_sentence()}</div>',
                        unsafe_allow_html=True
                    )
            with c3:
                if st.button("🗑️ Delete Word", use_container_width=True):
                    removed = engine.delete_last_word()
                    if removed:
                        st.toast(f"Deleted word: {removed}")
                    text_display_box.markdown(
                        f'<div class="text-box">{engine.get_full_sentence(include_current=True)}</div>',
                        unsafe_allow_html=True
                    )
                    english_display_box.markdown(
                        f'<div class="english-box">{engine.get_translated_sentence()}</div>',
                        unsafe_allow_html=True
                    )
            with c4:
                if st.button("🧹 Clear All", use_container_width=True):
                    if continuous_mode:
                        crec = load_continuous_recognizer()
                        if crec:
                            crec.reset_session()
                    engine.reset()
                    st.session_state.accumulated_text = ""
                    st.session_state.active_top3 = []
                    text_display_box.markdown('<div class="text-box">(Cleared)</div>', unsafe_allow_html=True)
                    english_display_box.markdown('<div class="english-box">(Cleared)</div>', unsafe_allow_html=True)
                    st.toast("Sentence buffers cleared.")

            c5, c6 = st.columns(2)
            with c5:
                if st.button("💾 Save Sentence Now", use_container_width=True):
                    full_txt = engine.get_full_sentence(include_current=False)
                    if continuous_mode:
                        crec = load_continuous_recognizer()
                        if crec and crec.last_translation.english_translation:
                            full_txt = crec.last_translation.english_translation
                    if full_txt.strip():
                        if not st.session_state.disable_logging:
                            logger.log_prediction(full_txt, tag="SENTENCE")
                            st.success(f"Saved: '{full_txt}' to file!")
                        else:
                            st.info("File logging is disabled in Privacy settings.")
                    else:
                        st.warning("Current sentence is empty.")
            with c6:
                if st.button("🔊 Speak English (TTS)", use_container_width=True, type="primary"):
                    speak_target = engine.get_translated_sentence() or engine.get_full_sentence(include_current=False)
                    if continuous_mode:
                        crec = load_continuous_recognizer()
                        if crec and crec.last_translation.english_translation:
                            speak_target = crec.last_translation.english_translation
                    if speak_target.strip():
                        speak_text(speak_target)
                        st.toast(f"Speaking: '{speak_target}'")
                    else:
                        st.warning("Sentence is empty! Perform signs first.")

            # File Management Section
            st.markdown("---")
            st.markdown("#### 📄 Prediction Log Actions")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.download_button(
                    "📥 Download Log (.txt)",
                    data=logger.get_file_content(),
                    file_name="predicted_signs.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with f_col2:
                if st.button("🗑️ Clear Log File", use_container_width=True):
                    logger.clear()
                    st.toast("Log file cleared.")

            with st.expander("📜 View File Contents (`data/predicted_signs.txt`)", expanded=False):
                st.text_area("Recent File Log", value=logger.get_file_content(), height=160)

        with col_video:
            # Mode Toggle: Continuous Live Stream vs In-Browser Snapshot
            cam_mode = st.radio(
                "Camera Input Mode:",
                options=["🎥 Live Stream (OpenCV)", "📸 In-Browser Webcam (Instant Snapshot)"],
                horizontal=True,
                help="Switch to 'In-Browser Webcam' if live stream is blocked by OS camera permissions or virtual drivers."
            )

            # Live Closed-Captions HUD Banner
            cc_placeholder = st.empty()
            cc_placeholder.markdown(
                f'<div class="cc-hud">'
                f'<div class="cc-speaker">🎙️ LIVE ASL CLOSED CAPTIONS</div>'
                f'<div class="cc-content">{engine.get_translated_sentence() or engine.get_full_sentence() or "Awaiting sign language input..."}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            if cam_mode == "📸 In-Browser Webcam (Instant Snapshot)":
                st.info("💡 **In-Browser Webcam Mode**: Uses your browser directly via WebRTC. Works seamlessly on any OS with zero driver conflicts.")
                browser_snap = st.camera_input("Hold up your ASL sign and click 'Take Photo'", key="browser_cam_snap")
                if browser_snap is not None:
                    bytes_data = browser_snap.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None and predictor is not None:
                        with st.spinner("Analyzing ASL sign hand kinematics..."):
                            pred = predictor.predict_frame(frame)

                        annotated = frame.copy()
                        if pred.hand_detected:
                            if pred.dual_result is not None:
                                draw_dual_styled_landmarks(annotated, pred.dual_result)
                            elif pred.hand_landmarks_proto:
                                draw_styled_landmarks(annotated, pred.hand_landmarks_proto)

                            disp_sign = pred.display_label
                            st.success(f"🎯 **Predicted Sign:** `{disp_sign}` ({pred.confidence_score:.1f}% confidence)")

                            # Update HUD
                            cc_placeholder.markdown(
                                f'<div class="cc-hud">'
                                f'<div class="cc-speaker">🎙️ LIVE ASL CLOSED CAPTIONS</div>'
                                f'<div class="cc-content">{disp_sign}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            snap_col1, snap_col2 = st.columns(2)
                            with snap_col1:
                                if st.button("➕ Add to Sentence", key="btn_add_snap", use_container_width=True):
                                    sign_val = pred.phrase_text if (pred.is_phrase and pred.phrase_text) else (pred.stable_label or pred.predicted_label or pred.label)
                                    if pred.is_phrase or len(sign_val) > 1:
                                        engine.add_phrase(sign_val)
                                        if not st.session_state.disable_logging:
                                            logger.log_prediction(sign_val, tag="WORD")
                                    else:
                                        engine.add_character(sign_val)
                                    text_display_box.markdown(
                                        f'<div class="text-box">{engine.get_full_sentence(include_current=True)}</div>',
                                        unsafe_allow_html=True
                                    )
                                    english_display_box.markdown(
                                        f'<div class="english-box">{engine.get_translated_sentence()}</div>',
                                        unsafe_allow_html=True
                                    )
                                    st.toast(f"Added '{sign_val}' to sentence!")
                            with snap_col2:
                                if st.button("🔊 Speak Sign", key="btn_speak_snap", use_container_width=True):
                                    sign_val = pred.phrase_text if (pred.is_phrase and pred.phrase_text) else (pred.stable_label or pred.predicted_label or pred.label)
                                    speak_text(sign_val)
                                    st.toast(f"Spoke: {sign_val}")
                        else:
                            st.warning("⚠️ No hand landmarks detected. Hold your hand clearly in front of the camera and retake.")

                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Landmark Annotated Snapshot", use_container_width=True)

            else:
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

                if not st.session_state.webcam_running:
                    video_placeholder.info("Webcam is idle. Click **'Start Webcam'** above to begin live motion detection and captions.")
                else:
                    # Robust multi-backend camera initialization
                    cap, actual_cam_id, b_used, switch_notif = open_robust_camera(
                        requested_index=camera_id,
                        preferred_backend=backend_choice,
                        auto_fallback=auto_fallback_enabled
                    )

                    if switch_notif:
                        st.info(switch_notif)

                    if cap is None:
                        st.error(f"❌ Could not open webcam index {camera_id}.")
                        st.session_state.webcam_running = False
                    else:
                        auto_commit_counter = 0
                        last_auto_char = ""
                        last_rendered_phrase = ""
                        frame_idx = 0
                        consecutive_dropped = 0

                        is_turbo = "Turbo" in perf_mode or "60 FPS" in perf_mode
                        is_power_saver = "Power Saver" in perf_mode

                        metric_step = 6 if is_power_saver else (5 if is_turbo else 4)
                        sleep_time = 0.025 if is_power_saver else 0.0
                        jpeg_qual = 65 if is_power_saver else 72

                        fps_history = []
                        last_fps_time = time.perf_counter()
                        live_fps = 30.0

                        crec = load_continuous_recognizer() if continuous_mode else None

                        try:
                            while st.session_state.webcam_running:
                                ret, frame = cap.read()
                                if not ret or frame is None or frame.size == 0:
                                    consecutive_dropped += 1
                                    if consecutive_dropped <= 15:
                                        time.sleep(0.03)
                                        continue
                                    else:
                                        st.warning(f"Webcam (Camera {actual_cam_id}) stopped transmitting video frames.")
                                        break
                                consecutive_dropped = 0

                                frame_idx += 1
                                frame = cv2.flip(frame, 1)
                                h, w, _ = frame.shape

                                now_t = time.perf_counter()
                                dt = now_t - last_fps_time
                                last_fps_time = now_t
                                if dt > 0:
                                    fps_history.append(1.0 / dt)
                                    if len(fps_history) > 15:
                                        fps_history.pop(0)
                                    live_fps = sum(fps_history) / len(fps_history)

                                if w > 640:
                                    frame = cv2.resize(frame, (640, int(640 * h / w)))
                                    h, w, _ = frame.shape

                                # =============================================
                                # PARADIGM 1: CONTINUOUS STREAM (CTC + HOLISTIC)
                                # =============================================
                                if continuous_mode and crec is not None:
                                    crec.confidence_threshold = conf_threshold
                                    crec.bimanual_engine.dominant_hand = handedness_pref

                                    c_res = crec.process_frame(frame)
                                    annotated_frame = crec.draw_multimodal_hud(frame, c_res)

                                    if c_res.top3_candidates:
                                        st.session_state.active_top3 = c_res.top3_candidates

                                    # Live FPS overlay badge on frame
                                    fps_color = (0, 255, 127) if live_fps >= 28 else ((0, 215, 255) if live_fps >= 20 else (0, 100, 255))
                                    cv2.rectangle(annotated_frame, (w - 145, 10), (w - 10, 42), (15, 23, 42), -1)
                                    cv2.putText(annotated_frame, f"FPS: {live_fps:.1f}", (w - 135, 33),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)

                                    # Fast SIMD JPEG byte stream for instant zero-lag rendering
                                    ret_enc, jpeg_buf = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qual])
                                    if ret_enc:
                                        video_placeholder.image(jpeg_buf.tobytes(), format="JPEG", use_container_width=True)

                                    if c_res.new_sign_emitted and c_res.emitted_sign_text:
                                        if not st.session_state.disable_logging:
                                            logger.log_prediction(c_res.emitted_sign_text, tag="CONTINUOUS")
                                        st.toast(f"🎯 Recognized: {c_res.emitted_sign_text}")

                                    # Throttled status updates
                                    if frame_idx % metric_step == 0:
                                        metric_hands_box.markdown(f"🙌 **Mode:** `Holistic Stream ({c_res.num_hands} Hands, Pose={c_res.has_pose}, Face={c_res.has_face})`")
                                        metric_pred_box.markdown(f"### Active Sign: **{c_res.display_label}**")
                                        metric_prob_box.markdown(f"**Confidence:** `{c_res.confidence_score:.1f}%` ({c_res.confidence_level}) | **Motion:** `{c_res.motion_state}`")
                                        metric_motion_box.markdown(f"**Kinetic Energy:** `{c_res.kinetic_energy:.4f}` | **Signing Space:** `{c_res.signing_space}`")
                                        metric_phrase_box.markdown(f"**Non-Manual Cues:** `{c_res.non_manual_summary}`")
                                        metric_buffer_box.markdown(f"**Stream Tokens:** `{len(c_res.asl_gloss_history)} signs recorded`")

                                        text_display_box.markdown(
                                            f'<div class="text-box">{c_res.raw_asl_string or "(Signing...)"}</div>',
                                            unsafe_allow_html=True
                                        )
                                        english_display_box.markdown(
                                            f'<div class="english-box">{c_res.translated_english or "(English translation will appear as you sign...)"}</div>',
                                            unsafe_allow_html=True
                                        )

                                        # Linguistic Grammar Badge
                                        if c_res.sentence_type == "WH_QUESTION":
                                            b_html = '<span class="badge-wh-q">🔍 WH-Question (Information Query)</span>'
                                        elif c_res.sentence_type == "YES_NO_QUESTION":
                                            b_html = '<span class="badge-yn-q">❓ Yes/No Question (Inquiry)</span>'
                                        elif c_res.sentence_type == "NEGATED_STATEMENT":
                                            b_html = '<span class="badge-neg">🚫 Negated Statement</span>'
                                        else:
                                            b_html = '<span class="badge-statement">🗣️ Declarative / Statement</span>'
                                        grammar_badge_box.markdown(b_html, unsafe_allow_html=True)

                                        if c_res.top3_candidates:
                                            t3_str = " &nbsp;|&nbsp; ".join([f"**#{i+1}: {g}** ({p:.0%})" for i, (g, p) in enumerate(c_res.top3_candidates)])
                                            alt_candidates_box.markdown(t3_str)

                                        cc_placeholder.markdown(
                                            f'<div class="cc-hud">'
                                            f'<div class="cc-speaker">🎙️ CONTINUOUS ASL TRANSLATION</div>'
                                            f'<div class="cc-content">{c_res.translated_english or c_res.raw_asl_string or "(Awaiting sign language...)"}</div>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )

                                # =============================================
                                # PARADIGM 2: ISOLATED MOTION-AWARE (LSTM/RF)
                                # =============================================
                                elif predictor is not None:
                                    pred = predictor.predict_frame(frame)

                                    if pred.hand_detected:
                                        if pred.dual_result is not None:
                                            draw_dual_styled_landmarks(frame, pred.dual_result)
                                        elif pred.hand_landmarks_proto:
                                            draw_styled_landmarks(frame, pred.hand_landmarks_proto)

                                    if pred.top3_candidates:
                                        st.session_state.active_top3 = pred.top3_candidates

                                    # Whole-word / phrase detection
                                    if pred.is_phrase and pred.phrase_text:
                                        if pred.phrase_text != last_rendered_phrase:
                                            engine.add_phrase(pred.phrase_text)
                                            if not st.session_state.disable_logging:
                                                logger.log_prediction(pred.phrase_text, tag="2-HAND" if pred.is_two_handed else "WORD")
                                            last_rendered_phrase = pred.phrase_text
                                            st.toast(f"{'🙌 2-Hand' if pred.is_two_handed else '📖 Word'}: {pred.phrase_text}")

                                    # Static character accumulation
                                    if pred.is_stable and pred.stable_label and not pred.is_uncertain:
                                        st.session_state.last_stable_char = pred.stable_label
                                        if pred.stable_label == last_auto_char:
                                            auto_commit_counter = 0
                                        else:
                                            auto_commit_counter += 1
                                            if auto_commit_counter >= 12:
                                                ch = pred.stable_label
                                                engine.add_character(ch)
                                                last_auto_char = ch
                                                auto_commit_counter = 0
                                    else:
                                        auto_commit_counter = 0
                                        if not pred.hand_detected or pred.kinetic_energy > 0.015:
                                            last_auto_char = ""
                                        if not pred.hand_detected:
                                            last_rendered_phrase = ""

                                    # Pause autospace
                                    if engine.current_word and (time.time() - engine.last_char_timestamp) > engine.pause_autospace_seconds:
                                        w_saved = engine.commit_current_word(autocorrect=True)
                                        if w_saved:
                                            if not st.session_state.disable_logging:
                                                logger.log_prediction(w_saved, tag="WORD")
                                            st.toast(f"✨ Word Formed: {w_saved}")

                                    # Native Closed Caption overlay
                                    cv2.rectangle(frame, (0, h - 70), (w, h), (15, 23, 42), -1)
                                    disp_cc = engine.get_translated_sentence() or engine.get_full_sentence() or "(Listening for signs...)"
                                    cv2.putText(frame, f"[ASL AI]: {disp_cc[:60]}", (15, h - 25),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

                                    # Live FPS overlay badge on frame
                                    fps_color = (0, 255, 127) if live_fps >= 28 else ((0, 215, 255) if live_fps >= 20 else (0, 100, 255))
                                    cv2.rectangle(frame, (w - 145, 10), (w - 10, 42), (15, 23, 42), -1)
                                    cv2.putText(frame, f"FPS: {live_fps:.1f}", (w - 135, 33),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)

                                    # Fast SIMD JPEG byte stream for instant zero-lag rendering
                                    ret_enc, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qual])
                                    if ret_enc:
                                        video_placeholder.image(jpeg_buf.tobytes(), format="JPEG", use_container_width=True)

                                    # Throttled status updates
                                    if frame_idx % metric_step == 0:
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

                                        if pred.is_phrase and pred.phrase_text:
                                            icon = "🙌 2-HAND" if pred.is_two_handed else "✨ PHRASE"
                                            metric_phrase_box.markdown(f'<div class="badge-phrase">{icon}: {pred.phrase_text}</div>', unsafe_allow_html=True)
                                        else:
                                            metric_phrase_box.empty()

                                        buffer_ratio = pred.buffer_fill / max(1, predictor.sequence_length)
                                        metric_buffer_box.markdown(f"**LSTM Window:** `[{pred.buffer_fill}/{predictor.sequence_length}]` ({buffer_ratio:.0%})")

                                        text_display_box.markdown(
                                            f'<div class="text-box">{engine.get_full_sentence(include_current=True) or "(Signing...)"}</div>',
                                            unsafe_allow_html=True
                                        )
                                        english_display_box.markdown(
                                            f'<div class="english-box">{engine.get_translated_sentence() or "(Translated English sentence...)"}</div>',
                                            unsafe_allow_html=True
                                        )

                                        if pred.top3_candidates:
                                            t3_str = " &nbsp;|&nbsp; ".join([f"**#{i+1}: {g}** ({p:.0%})" for i, (g, p) in enumerate(pred.top3_candidates)])
                                            alt_candidates_box.markdown(t3_str)

                                        cc_placeholder.markdown(
                                            f'<div class="cc-hud">'
                                            f'<div class="cc-speaker">🎙️ LIVE ASL CLOSED CAPTIONS</div>'
                                            f'<div class="cc-content">{disp_cc}</div>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )

                                if sleep_time > 0:
                                    time.sleep(sleep_time)

                        finally:
                            release_camera_safely(cap)

    # =========================================================================
    # TAB 2: STATIC IMAGE ANALYSIS & DATASET EXPLORER
    # =========================================================================
    with tab_upload:
        st.subheader("🖼️ High-Precision Gesture Image Analysis & Multi-Dataset Explorer")
        st.markdown(
            "Test hand landmark detection, 1€ anti-shiver stabilization, and sign classification on any uploaded image "
            "or explore samples from the **ASL Alphabet** dataset."
        )

        dataset_dir = Path("dataset")
        if not dataset_dir.exists():
            dataset_dir = Path("archive/asl_dataset")
        if not dataset_dir.exists():
            dataset_dir = Path("archive/asl_alphabet_test/asl_alphabet_test")

        tab2_col_mode, tab2_col_model = st.columns(2)
        with tab2_col_mode:
            options_source = ["📁 Upload Image from Computer", "📦 Select Sample from ASL Alphabet (39 Classes: 0-9, A-Z)"]

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

                if pred.top3_candidates:
                    st.markdown("##### 📊 Top 3 Candidate Predictions")
                    t_cols = st.columns(len(pred.top3_candidates))
                    for c_idx, (cand_name, p_val) in enumerate(pred.top3_candidates):
                        with t_cols[c_idx]:
                            st.write(f"**#{c_idx+1}: {cand_name}** ({p_val:.1%})")
                            st.progress(float(p_val))

                if st.button("➕ Append to Predicted Sentence & Log to File", type="primary"):
                    char = pred.predicted_label
                    engine.add_character(char)
                    if not st.session_state.disable_logging:
                        logger.log_prediction(char, tag="SIGN")
                    st.success(f"Appended '{char}'! Current sentence: {engine.get_full_sentence()}")
            else:
                st.warning(f"⚠️ {pred.status_message}")

    # =========================================================================
    # TAB 3: MODEL PERFORMANCE & BENCHMARKS
    # =========================================================================
    with tab_metrics:
        st.subheader("📊 System Performance & Comprehensive Benchmark Suite")

        eval_v2_path = Path("models/evaluation_report_v2.json")
        bench_md_path = Path("models/benchmark_report.md")
        eval_json_path = Path("models/evaluation_results.json")
        cm_img_path = Path("models/confusion_matrix.png")

        # 1. Continuous CTC & Sequence Benchmarks
        if eval_v2_path.exists():
            with open(eval_v2_path, "r", encoding="utf-8") as f:
                v2_data = json.load(f)

            st.markdown("### ⚡ Continuous CTC & Latency Benchmarks")
            s_met = v2_data.get("sequence_metrics", {})
            p_prof = v2_data.get("performance_profile", {})

            bm1, bm2, bm3, bm4 = st.columns(4)
            with bm1:
                st.metric("Median Latency (p50)", f"{p_prof.get('latency_p50_ms', 0):.1f} ms")
            with bm2:
                st.metric("Tail Latency (p95)", f"{p_prof.get('latency_p95_ms', 0):.1f} ms")
            with bm3:
                st.metric("Character Error Rate (CER)", f"{s_met.get('mean_character_error_rate_cer', 0):.1%}")
            with bm4:
                st.metric("Word Error Rate (WER)", f"{s_met.get('mean_word_error_rate_wer', 0):.1%}")

            bm5, bm6, bm7, bm8 = st.columns(4)
            with bm5:
                st.metric("Transition False-Positives (TFPR)", f"{s_met.get('transition_false_positive_rate', 0):.1%}")
            with bm6:
                st.metric("Sentence Accuracy", f"{s_met.get('sentence_accuracy_percentage', 0):.1f}%")
            with bm7:
                st.metric("p99 Latency", f"{p_prof.get('latency_p99_ms', 0):.1f} ms")
            with bm8:
                st.metric("Hardware Architecture", "100% CPU Local (ONNX)")

            # Robustness Scorecard
            st.markdown("#### 🛡️ Robustness & Invariance Scorecard")
            rob = v2_data.get("robustness_scorecard", {})
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            with r_c1:
                st.info(f"**Occlusion (20% Dropout):**\n\n{rob.get('occlusion_resistance_20pct', 'N/A')}")
            with r_c2:
                st.info(f"**Scale Invariance (0.6x-1.5x):**\n\n{rob.get('scale_invariance_0.6x_to_1.5x', 'N/A')}")
            with r_c3:
                st.info(f"**Rotation Tolerance (±25°):**\n\n{rob.get('rotation_tolerance_pm25deg', 'N/A')}")
            with r_c4:
                st.info(f"**Left-Handed Parity:**\n\n{rob.get('left_handed_mirror_accuracy', 'N/A')}")

        st.markdown("---")

        # 2. Isolated Baseline LSTM / RF Metrics
        st.markdown("### 📈 Motion-Aware LSTM Held-Out Test Evaluation (39 Classes)")
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

            m_col1, m_col2 = st.columns([3, 2])
            with m_col1:
                st.markdown("#### 🎯 LSTM Confusion Matrix Heatmap")
                if cm_img_path.exists():
                    st.image(str(cm_img_path), use_container_width=True)
                else:
                    st.info("Confusion matrix image not yet generated.")
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

        # 3. View Full Markdown Report
        if bench_md_path.exists():
            with st.expander("📄 View Full Comprehensive Benchmark Report (`models/benchmark_report.md`)", expanded=False):
                with open(bench_md_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())

    # =========================================================================
    # TAB 4: INTERACTIVE SIGN TUTORIAL & PRACTICE GUIDE
    # =========================================================================
    with tab_tutorial:
        render_tutorial_tab()


if __name__ == "__main__":
    main()
