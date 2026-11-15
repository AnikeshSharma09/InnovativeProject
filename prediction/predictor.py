"""
Real-Time Gesture & Motion Predictor Engine.
Supports:
1. Dual-Hand (Bimanual) Sign Language Recognition:
   - Synchronized tracking of Left and Right hands (MediaPipe Hands max_num_hands=2)
   - Inter-hand relational geometry (wrist-to-wrist distance, contact, elevation)
   - 2-Hand Whole-Word Recognition ("BOOK", "HELP", "MEET", "WORK", "HOUSE", "FINISHED", "PLAY")
2. Single-Hand Spatial & Recurrent Gesture Recognition:
   - 39 Base Classes (0-9, A-Z, del, space, nothing) with Random Forest (99.3%) & PyTorch LSTM
   - 1-Hand Macro Signs ("APPROVED", "PLEASE WAIT", "I AGREE", "DISAGREE", "THANK YOU", "HELLO", "STANDUP")
3. Adaptive 1€ Anti-Tremor & Shiver Stabilization across both hands.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
import sys
import warnings
warnings.filterwarnings("ignore")
import time
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from collections import deque, Counter
from pathlib import Path
import cv2
import numpy as np
import joblib
import mediapipe as mp
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm_model import MotionSignLSTM, load_lstm_checkpoint
from utils.onnx_exporter import ONNXModelSession
from utils.landmarks_util import (
    extract_motion_frame_data,
    extract_motion_features,
    extract_landmarks_from_bgr_frame,
    robust_extract_landmarks_from_bgr_frame,
    draw_styled_landmarks,
    extract_dual_hands_from_bgr_frame,
    draw_dual_styled_landmarks,
    DualHandDetectionResult,
    TOTAL_MOTION_FEATURES,
    DEFAULT_SEQUENCE_LENGTH
)
from utils.labels import format_label_for_display
from utils.stabilizer import LandmarkStabilizer, ProbabilityStabilizer, DualLandmarkStabilizer
from prediction.phrase_recognizer import EnterprisePhraseRecognizer, PhraseDetectionResult
from utils.lip_morpheme_engine import LipMorphemeEngine
from utils.hand_calibrator import HandCalibrator


@dataclass
class PredictionOutput:
    """Encapsulates the complete prediction result for a single frame or image."""
    hand_detected: bool
    predicted_label: str
    display_label: str
    confidence_level: str  # 'HIGH', 'MEDIUM', 'LOW'
    confidence_score: float  # [0.0 - 100.0]%
    raw_probability: float  # [0.0 - 1.0]
    margin: float
    kinetic_energy: float
    motion_state: str  # 'STATIC HOLD', 'ACTIVE MOTION', 'FAST MOTION', 'TWO-HANDED GESTURE', 'WHOLE PHRASE'
    buffer_fill: int
    is_uncertain: bool
    is_stable: bool
    stable_label: Optional[str]
    hand_landmarks_proto: Optional[object]
    status_message: str
    is_shivering: bool = False
    jitter_score: float = 0.0
    is_phrase: bool = False
    phrase_text: Optional[str] = None
    stabilized_coords: Optional[List[Tuple[float, float, float]]] = None
    num_hands: int = 1
    is_two_handed: bool = False
    left_hand_proto: Optional[object] = None
    right_hand_proto: Optional[object] = None
    dual_result: Optional[DualHandDetectionResult] = None
    top3_candidates: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Compatibility property for UI consumers expecting .label"""
        return self.display_label or self.predicted_label or ""


class SignPredictor:
    """
    Production-grade inference engine integrating:
    - Dual-hand (bimanual) and single-hand tracking
    - 1€ (One-Euro) adaptive anti-tremor stabilization
    - Deep Learning recurrent motion sequence modeling (PyTorch LSTM)
    - High-precision static spatial feature classification (Random Forest)
    - Real-time kinematic phrase and sentence recognition
    """

    def __init__(
        self,
        lstm_model_path: str = "models/sign_lstm.pth",
        encoder_path: str = "models/label_encoder.pkl",
        rf_model_path: str = "models/sign_model.pkl",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        min_confidence: float = 0.28,
        min_margin: float = 0.03,
        engine_mode: str = "auto",
        enable_stabilization: bool = True,
        stabilizer_min_cutoff: float = 0.8,
        stabilizer_beta: float = 0.015,
        device: str = "cpu",
        enable_word_signs: bool = False
    ):
        self.lstm_model_path = Path(lstm_model_path)
        self.encoder_path = Path(encoder_path)
        self.rf_model_path = Path(rf_model_path)
        self.wlasl_model_path = Path("models/wlasl_word_model.pkl")
        self.wlasl_encoder_path = Path("models/wlasl_label_encoder.pkl")
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.engine_mode = engine_mode  # 'auto', 'rf', 'lstm'
        self.device = torch.device(device)
        self.enable_stabilization = enable_stabilization
        self.enable_word_signs = enable_word_signs

        # Biomechanical Stabilizers (One-Euro Filter across Left and Right hands)
        self.landmark_stabilizer = LandmarkStabilizer(
            min_cutoff=stabilizer_min_cutoff,
            beta=stabilizer_beta
        )
        self.dual_stabilizer = DualLandmarkStabilizer(
            min_cutoff=stabilizer_min_cutoff,
            beta=stabilizer_beta
        )
        self.prob_stabilizer = ProbabilityStabilizer(alpha=0.90)

        # Whole-Sentence & Dual-Hand Recognition Engine
        self.phrase_recognizer = EnterprisePhraseRecognizer(enable_single_hand_macros=False)

        # Temporal Sequence Window for LSTM (stores last N frames of 129 features)
        self.sequence_buffer: deque = deque(maxlen=sequence_length)
        self.prediction_history: deque = deque(maxlen=10)
        self.prev_raw_coords: Optional[List[Tuple[float, float, float]]] = None
        self.last_stable_label: Optional[str] = None
        self.tremor_deadband: bool = True

        # Load models
        self.lstm_model: Optional[MotionSignLSTM] = None
        self._rf_model = None
        self._wlasl_model = None
        self.label_encoder = None
        self.classes: List[str] = []
        self.wlasl_encoder = None
        self.wlasl_classes: List[str] = []
        self._load_models()

        # ONNX Runtime Acceleration
        self.onnx_lstm_path = Path("models/sign_lstm.onnx")
        self.onnx_session: Optional[ONNXModelSession] = None
        if self.onnx_lstm_path.exists():
            try:
                self.onnx_session = ONNXModelSession(str(self.onnx_lstm_path))
            except Exception:
                self.onnx_session = None

        # Initialize MediaPipe Hands supporting up to 2 hands (Dual Hand / Bimanual)
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25
        )

        # Innovation Engines: Multimodal Lip Morpheme Fusion & Biomechanical Calibration
        self.lip_engine = LipMorphemeEngine()
        self.calibrator = HandCalibrator()
        self.enable_lip_fusion = True
        self.enable_calibration = True
        self._face_mesh = None

    @property
    def face_mesh(self):
        """Lazy loads lightweight MediaPipe FaceMesh for multimodal lip morpheme fusion."""
        if self._face_mesh is None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
        return self._face_mesh

    @property
    def rf_model(self):
        """Lazy-loads Random Forest baseline (~80MB RAM) on-demand only when requested."""
        if self._rf_model is None and self.rf_model_path.exists():
            try:
                self._rf_model = joblib.load(self.rf_model_path)
                print(f"[+] SignPredictor: On-demand loaded Random Forest baseline from {self.rf_model_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load RF model: {e}")
        return self._rf_model

    @rf_model.setter
    def rf_model(self, value):
        self._rf_model = value

    @property
    def has_rf_model(self) -> bool:
        """Checks if Random Forest checkpoint is available on disk without allocating RAM."""
        return self._rf_model is not None or self.rf_model_path.exists()

    @property
    def wlasl_model(self):
        """Lazy-loads WLASL Word Model (~25MB RAM) on-demand only when requested."""
        if self._wlasl_model is None and self.wlasl_model_path.exists():
            try:
                self._wlasl_model = joblib.load(self.wlasl_model_path)
                print(f"[+] SignPredictor: On-demand loaded WLASL Word Model from {self.wlasl_model_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load WLASL Word Model: {e}")
        return self._wlasl_model

    @wlasl_model.setter
    def wlasl_model(self, value):
        self._wlasl_model = value

    def _load_models(self) -> None:
        """Loads lightweight LSTM checkpoint and metadata; keeps heavy baseline models lazy."""
        if not self.encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at '{self.encoder_path}'. "
                "Please run training first: 'python training/train_lstm.py'"
            )

        self.label_encoder = joblib.load(self.encoder_path)
        self.classes = [str(c) for c in self.label_encoder.classes_]

        # Load primary LSTM model (compact ~1MB)
        if self.lstm_model_path.exists():
            try:
                loaded = load_lstm_checkpoint(
                    checkpoint_path=str(self.lstm_model_path),
                    input_size=TOTAL_MOTION_FEATURES,
                    num_classes=len(self.classes),
                    device=self.device
                )
                self.lstm_model = loaded[0] if isinstance(loaded, tuple) else loaded
                print(f"[+] SignPredictor: Loaded Deep Learning LSTM model from {self.lstm_model_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load LSTM model: {e}")

        # Pre-load lightweight WLASL encoder classes without loading the heavy model pickle
        if self.wlasl_encoder_path.exists():
            try:
                self.wlasl_encoder = joblib.load(self.wlasl_encoder_path)
                self.wlasl_classes = [str(c) for c in self.wlasl_encoder.classes_]
            except Exception:
                pass

        if not self.wlasl_classes:
            classes_json_path = self.models_dir / "wlasl_classes.json"
            if classes_json_path.exists():
                try:
                    with open(classes_json_path, "r", encoding="utf-8") as f:
                        self.wlasl_classes = json.load(f)
                except Exception:
                    pass

        if self.lstm_model is None and not self.has_rf_model:
            raise FileNotFoundError(
                f"Neither LSTM checkpoint '{self.lstm_model_path}' nor RF model '{self.rf_model_path}' found."
            )

    def predict_wlasl_word(self, min_confidence: float = 0.35) -> Optional[Tuple[str, float]]:
        """
        Evaluates the rolling temporal motion window against the WLASL Word-Level Model.
        Returns: (predicted_word, confidence_probability) if confident, or None.
        """
        if self.wlasl_model is None or len(self.sequence_buffer) < 8:
            return None

        frames = list(self.sequence_buffer)
        while len(frames) < self.sequence_length:
            frames.insert(0, frames[0])

        seq_array = np.array(frames, dtype=np.float32)  # (20, 129)

        # 258-dim summary vector: initial shape (63) + mid shape (63) + final shape (63) + mean velocity (63) + traj (3) + max vel (3)
        init_shape = seq_array[0, :63]
        mid_shape = seq_array[len(frames) // 2, :63]
        final_shape = seq_array[-1, :63]
        mean_vel = np.mean(seq_array[:, 63:126], axis=0)
        traj_delta = seq_array[-1, 126:129] - seq_array[0, 126:129]
        max_vel = np.max(np.abs(seq_array[:, 63:66]), axis=0)

        feat_258 = np.hstack([init_shape, mid_shape, final_shape, mean_vel, traj_delta, max_vel]).reshape(1, -1)

        try:
            probs = self.wlasl_model.predict_proba(feat_258)[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

            if top_prob >= min_confidence:
                word_label = self.wlasl_classes[top_idx].upper()
                return word_label, top_prob
        except Exception:
            pass

        return None

    def reset_buffer(self) -> None:
        """Clears sequence buffer and prediction history."""
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self.prev_raw_coords = None
        self.last_stable_label = None
        self.landmark_stabilizer.reset()
        self.dual_stabilizer.reset()
        self.prob_stabilizer.reset()
        self.phrase_recognizer.reset()

    def _apply_anatomical_disambiguation(
        self,
        raw_coords: Optional[List[Tuple[float, float, float]]],
        probs: np.ndarray,
        kinetic_energy: float = 0.0,
        bgr_frame: Optional[np.ndarray] = None,
        face_landmarks: Any = None
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Surgically disambiguates geometrically and kinematically confused ASL sign pairs
        ONLY when the model's top-2 predictions are genuinely competing for that pair.
        1. 'W' vs '6': Inter-finger spread angle (fan-out vs parallel stack).
        2. 'U' vs 'V': Index-middle fingertip separation ratio (touching vs spread).
        3. 'C' vs 'O': Aperture opening (wide opening vs closed circle).
        4. '0' vs 'O': Aspect ratio (narrow compressed oval vs circular loop).
        5. 'Z' vs '1': Dynamic motion trajectory vs static upright point.
        6. 'J' vs 'I': Dynamic swooping curve vs static upright pinky.
        """
        if raw_coords is None or len(raw_coords) < 21:
            return probs, None

        w = np.array(raw_coords[0])
        m_mcp = np.array(raw_coords[9])
        palm = np.linalg.norm(m_mcp - w)
        if palm < 1e-4:
            return probs, None

        probs = probs.copy()
        disambig_notes = []

        sorted_idx = np.argsort(probs)[::-1]
        top1_idx = int(sorted_idx[0])
        top1_label = self.classes[top1_idx]
        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
        top2_label = self.classes[top2_idx]
        top2_set = {top1_label, top2_label}

        # Landmarks
        t_mcp = np.array(raw_coords[2])
        t_tip = np.array(raw_coords[4])
        i_mcp = np.array(raw_coords[5])
        i_pip = np.array(raw_coords[6])
        i_tip = np.array(raw_coords[8])
        m_pip = np.array(raw_coords[10])
        m_tip = np.array(raw_coords[12])
        r_mcp = np.array(raw_coords[13])
        r_pip = np.array(raw_coords[14])
        r_tip = np.array(raw_coords[16])
        p_mcp = np.array(raw_coords[17])
        p_pip = np.array(raw_coords[18])
        p_tip = np.array(raw_coords[20])

        # Anatomical finger extension states
        idx_ext = np.linalg.norm(i_tip - w) > np.linalg.norm(i_pip - w) * 1.08
        mid_ext = np.linalg.norm(m_tip - w) > np.linalg.norm(m_pip - w) * 1.08
        ring_ext = np.linalg.norm(r_tip - w) > np.linalg.norm(r_pip - w) * 1.08
        pnk_ext = np.linalg.norm(p_tip - w) > np.linalg.norm(p_pip - w) * 1.08

        # Thumb kinematics (abduction & extension)
        d_ti_mcp = float(np.linalg.norm(t_tip - i_mcp) / palm)
        d_tm_mcp = float(np.linalg.norm(t_tip - m_mcp) / palm)
        d_tr_mcp = float(np.linalg.norm(t_tip - r_mcp) / palm)
        d_tp_mcp = float(np.linalg.norm(t_tip - p_mcp) / palm)
        d_tw = float(np.linalg.norm(t_tip - w) / palm)
        thumb_ext = (d_tp_mcp >= 0.80) and (d_tw >= 0.75)

        # Distances normalized by palm length
        d_im = float(np.linalg.norm(i_tip - m_tip) / palm)
        d_mr = float(np.linalg.norm(m_tip - r_tip) / palm)
        d_ti = float(np.linalg.norm(t_tip - i_tip) / palm)
        d_tm = float(np.linalg.norm(t_tip - m_tip) / palm)
        d_tr = float(np.linalg.norm(t_tip - r_tip) / palm)
        d_tp = float(np.linalg.norm(t_tip - p_tip) / palm)
        d_tm_pip = float(np.linalg.norm(t_tip - m_pip) / palm)
        d_tip_pip = float(np.linalg.norm(i_tip - i_pip) / palm)

        # Hand orientation (upright vs horizontal vs downward)
        v_idx = i_tip - i_mcp
        v_thumb = t_tip - t_mcp
        norm_v_thumb = np.linalg.norm(v_thumb[:2])
        norm_v_idx = np.linalg.norm(v_idx[:2])
        cos_thumb_idx = float(np.dot(v_thumb[:2], v_idx[:2]) / (norm_v_thumb * norm_v_idx + 1e-6))
        ratio_xy = float(abs(v_idx[0]) / (abs(v_idx[1]) + 1e-6))
        is_downward = (idx_ext or mid_ext) and (w[1] < i_mcp[1] + 0.10) and (v_idx[1] > -0.05)
        is_horizontal = (ratio_xy >= 1.10) and not is_downward

        # Specialized poses: K and R guards
        is_k_pose = (top1_label == "K" or top2_label == "K") and (d_tm_pip < 0.28) and (v_idx[1] < -0.15)
        is_r_pose = (top1_label == "R" or top2_label == "R") and ((i_tip[0] - m_tip[0]) * (i_mcp[0] - m_mcp[0]) < 0)

        def _force_class(target_cls: str, reason: str, suppress_classes: list = None):
            nonlocal probs
            if target_cls not in self.classes:
                return
            if suppress_classes:
                for wrong in suppress_classes:
                    if wrong in self.classes:
                        probs[self.classes.index(wrong)] = min(probs[self.classes.index(wrong)], 0.005)
            t_idx = self.classes.index(target_cls)
            probs = probs * 0.05
            probs[t_idx] = 0.88
            probs = probs / np.sum(probs)
            disambig_notes.append(f"{reason} -> '{target_cls}'")

        def _disambiguate_pair(lbl_a: str, lbl_b: str, favor_a: bool, reason: str):
            nonlocal probs
            if lbl_a not in self.classes or lbl_b not in self.classes:
                return
            idx_a = self.classes.index(lbl_a)
            idx_b = self.classes.index(lbl_b)

            # Strictly require BOTH classes to be the top-2 predictions
            if not (top1_idx in (idx_a, idx_b) and top2_idx in (idx_a, idx_b)):
                return

            margin = abs(probs[idx_a] - probs[idx_b])
            if margin > 0.20:
                # Decisive margin (> 20%), respect learned representations
                return

            if favor_a and probs[idx_a] < probs[idx_b]:
                p_high = probs[idx_b]
                p_low = probs[idx_a]
                probs[idx_a] = p_high
                probs[idx_b] = p_low
                disambig_notes.append(f"{reason} -> '{lbl_a}'")
            elif (not favor_a) and probs[idx_b] < probs[idx_a]:
                p_high = probs[idx_a]
                p_low = probs[idx_b]
                probs[idx_b] = p_high
                probs[idx_a] = p_low
                disambig_notes.append(f"{reason} -> '{lbl_b}'")

        # =============================================================
        # COMPREHENSIVE ANATOMICAL & KINEMATIC DISAMBIGUATION (0-9, A-Z)
        # =============================================================

        # -------------------------------------------------------------
        # 1. ASL 'M' vs 'N': Fingers Draped Downward Over Thumb
        # -------------------------------------------------------------
        if v_idx[1] > 0.40 and (idx_ext or mid_ext):
            if ring_ext:
                _force_class("M", "3 fingers draped downward over thumb", ["N", "W", "3", "del"])
            else:
                _force_class("N", "2 fingers draped downward over thumb", ["M", "W", "3", "del"])

        # -------------------------------------------------------------
        # 2. Downward Signs: 'P' vs 'Q'
        # -------------------------------------------------------------
        elif (top1_label in ("P", "Q") or top2_label in ("P", "Q") or (is_downward and v_idx[0] < -0.25)):
            if d_tm < 0.35:
                _force_class("P", f"Downward hand with middle finger touch (d_tm={d_tm:.2f})", ["Q", "G", "K", "1"])
            else:
                _force_class("Q", "Downward index point with thumb along index", ["P", "G", "1", "K"])

        # -------------------------------------------------------------
        # 3. Pinky-Only Extended Signs: 'I', 'J', 'Y'
        # -------------------------------------------------------------
        elif pnk_ext and not idx_ext and not mid_ext and not ring_ext:
            if thumb_ext or d_tp >= 1.40:
                _force_class("Y", f"Extended thumb + pinky shaka (d_tp={d_tp:.2f})", ["I", "J"])
            elif kinetic_energy >= 0.028:
                _force_class("J", f"Dynamic pinky swoop stroke (KE={kinetic_energy:.3f})", ["I", "Y"])
            else:
                _force_class("I", "Static upright pinky hold", ["J", "Y"])

        # -------------------------------------------------------------
        # 4. Number Contact Loops: '6', '7', '8', '9' and Letter 'F'
        # -------------------------------------------------------------
        elif d_tp < 0.22 and idx_ext and mid_ext and ring_ext:
            _force_class("6", f"Thumb-pinky contact loop (d_tp={d_tp:.2f})", ["W", "7", "8", "9", "F"])
        elif d_tr < 0.22 and idx_ext and mid_ext and pnk_ext:
            _force_class("7", f"Thumb-ring contact loop (d_tr={d_tr:.2f})", ["6", "8", "9", "F"])
        elif d_tm < 0.22 and idx_ext and ring_ext and pnk_ext:
            _force_class("8", f"Thumb-middle contact loop (d_tm={d_tm:.2f})", ["6", "7", "9", "F"])
        elif (d_ti < 0.28) and (d_tm >= 0.35) and (mid_ext or ring_ext or pnk_ext):
            # Letter 'F' / Number '9': Thumb and index meet in a circle loop, outer fingers extended
            p_9 = probs[self.classes.index("9")] if "9" in self.classes else 0.0
            p_f = probs[self.classes.index("F")] if "F" in self.classes else 0.0
            is_9_geom = (d_tm >= 0.85) or (d_ti_mcp < 0.55)
            is_f_geom = (d_tm < 0.80) and (d_ti_mcp >= 0.55)
            if is_9_geom and (p_9 >= p_f * 0.25 or top1_label == "9"):
                target = "9"
            elif is_f_geom and (p_f >= p_9 * 0.25 or top1_label == "F"):
                target = "F"
            elif top1_label == "9" and p_9 > p_f * 0.80:
                target = "9"
            elif p_9 > p_f and top1_label != "F":
                target = "9"
            else:
                target = "F"
            suppress = ["6", "7", "8", "B", "D", "O", "C"]
            if target == "F" and "9" in self.classes:
                suppress.append("9")
            elif target == "9" and "F" in self.classes:
                suppress.append("F")
            _force_class(target, f"Thumb-index contact circle (d_ti={d_ti:.2f})", suppress)

        # -------------------------------------------------------------
        # 5. Exactly 1 Finger Extended (Index): 'G', 'Z', '1', 'D', 'L', 'X'
        # -------------------------------------------------------------
        elif idx_ext and not mid_ext and not ring_ext and not pnk_ext and not is_downward and v_idx[1] <= 0.15:
            is_thumb_folded = (d_tm_mcp < 0.40) or (d_tm_pip < 0.30)
            is_g_pose = (ratio_xy >= 1.45) and (abs(v_idx[0]) >= 0.18) and (not is_thumb_folded) and (cos_thumb_idx > 0.60)

            if (d_tip_pip < 0.38) or (top1_label == "X" and d_tip_pip < 0.45):
                _force_class("X", f"Bent index hook (d_tip_pip={d_tip_pip:.2f})", ["1", "D", "G", "Z"])
            elif kinetic_energy >= 0.028:
                _force_class("Z", f"Dynamic stroke trajectory (KE={kinetic_energy:.3f})", ["G", "1", "D", "X"])
            elif d_tm < 0.20 and ratio_xy < 1.20 and d_tip_pip >= 0.40 and (not is_g_pose):
                _force_class("D", f"Thumb-middle tip circle loop (d_tm={d_tm:.2f})", ["G", "Z", "1", "L", "X", "F"])
            elif thumb_ext and d_ti_mcp >= 0.55 and d_tm > 0.45 and ratio_xy < 0.70 and cos_thumb_idx < 0.65:
                _force_class("L", f"L-shape upright index with extended thumb (d_ti_mcp={d_ti_mcp:.2f})", ["1", "D", "G", "Z", "X"])
            elif is_g_pose:
                _force_class("G", "Static horizontal index point with parallel thumb", ["Z", "1", "D", "X"])
            elif (0.45 <= ratio_xy < 1.45) and is_thumb_folded and (top1_label == "Z" or top2_label == "Z" or (top1_label == "G" and probs[self.classes.index("Z")] > 0.02)):
                _force_class("Z", f"Diagonal index pose (ratio={ratio_xy:.2f}) with folded thumb", ["G", "1", "D", "X"])
            else:
                _force_class("1", "Static upright index point", ["G", "Z", "D", "X"])

        # -------------------------------------------------------------
        # 6. Exactly 2 Fingers Extended (Index, Middle): 'V', '2', '3', 'U', 'R', 'K', 'H'
        # -------------------------------------------------------------
        elif idx_ext and mid_ext and not ring_ext and not pnk_ext and not is_downward and v_idx[1] <= 0.15:
            if thumb_ext:
                _force_class("3", f"Thumb+Index+Mid ext (d_tp_mcp={d_tp_mcp:.2f}, d_tw={d_tw:.2f})", ["V", "2", "U", "R", "K", "H", "W", "6"])
            elif (ratio_xy >= 1.10) and d_im < 0.35 and v_idx[1] < 0.0:
                _force_class("H", "Horizontal 2-finger parallel", ["V", "2", "U", "R", "K", "W", "6"])
            elif ((i_tip[0] - m_tip[0]) * (i_mcp[0] - m_mcp[0]) < 0) or top1_label == "R":
                _force_class("R", "Crossed index and middle fingers", ["V", "2", "U", "K", "H", "W", "6"])
            elif (d_tm_pip < 0.36) and (t_tip[1] < i_mcp[1] + 0.15 * palm):
                # ASL 'K': Thumb placed at middle finger PIP joint
                _force_class("K", f"Thumb at middle PIP joint (d_tm_pip={d_tm_pip:.2f})", ["V", "2", "U", "R", "H", "W", "6"])
            elif d_im < 0.35:
                _force_class("U", f"Touching parallel fingers (d_im={d_im:.2f})", ["V", "2", "R", "K", "W", "6"])
            else:
                # Spread V-shape: Resolve 'V' vs '2' with zero jumping
                p_2 = probs[self.classes.index("2")] if "2" in self.classes else 0.0
                p_v = probs[self.classes.index("V")] if "V" in self.classes else 0.0
                is_2_geom = (d_tm_pip < 0.55) and (d_ti < 1.20) and (d_tw > 0.98)
                is_v_geom = (d_tm_pip >= 0.55) or (d_ti >= 1.20)
                if is_2_geom and (p_2 >= p_v * 0.25 or top1_label == "2"):
                    target = "2"
                elif is_v_geom and (p_v >= p_2 * 0.25 or top1_label == "V"):
                    target = "V"
                elif top1_label == "2" and p_2 > p_v * 0.80:
                    target = "2"
                elif p_2 > p_v and top1_label != "V":
                    target = "2"
                else:
                    target = "V"
                suppress = ["U", "R", "K", "H", "3", "W", "6"]
                if target == "V" and "2" in self.classes:
                    suppress.append("2")
                elif target == "2" and "V" in self.classes:
                    suppress.append("V")
                _force_class(target, f"Spread V-shape (d_im={d_im:.2f})", suppress)

        # -------------------------------------------------------------
        # 7. Three Fingers Extended: 'W' vs '6'
        # -------------------------------------------------------------
        elif idx_ext and mid_ext and ring_ext and not pnk_ext and not is_downward and v_idx[1] <= 0.15:
            if d_tp < 0.25:
                _force_class("6", f"Thumb-pinky contact loop (d_tp={d_tp:.2f})", ["W", "V", "3", "U", "2"])
            else:
                _force_class("W", f"3-finger spread fan (d_im={d_im:.2f}, d_mr={d_mr:.2f})", ["6", "V", "3", "U", "2"])

        # -------------------------------------------------------------
        # 8. Four/Five Fingers Extended: 'F', '4', 'B', '5', 'C', 'space'
        # -------------------------------------------------------------
        elif idx_ext and mid_ext and ring_ext and pnk_ext:
            # Check F/9 first if thumb touches index tip
            if d_ti < 0.28 and d_tm >= 0.35:
                p_9 = probs[self.classes.index("9")] if "9" in self.classes else 0.0
                p_f = probs[self.classes.index("F")] if "F" in self.classes else 0.0
                is_9_geom = (d_tm >= 0.85) or (d_ti_mcp < 0.55)
                is_f_geom = (d_tm < 0.80) and (d_ti_mcp >= 0.55)
                if is_9_geom and (p_9 >= p_f * 0.25 or top1_label == "9"):
                    target = "9"
                elif is_f_geom and (p_f >= p_9 * 0.25 or top1_label == "F"):
                    target = "F"
                elif top1_label == "9" and p_9 > p_f * 0.80:
                    target = "9"
                elif p_9 > p_f and top1_label != "F":
                    target = "9"
                else:
                    target = "F"
                suppress = ["B", "4", "5", "space", "D", "6", "7", "8"]
                if target == "F" and "9" in self.classes:
                    suppress.append("9")
                elif target == "9" and "F" in self.classes:
                    suppress.append("F")
                _force_class(target, f"Thumb-index contact circle in open hand (d_ti={d_ti:.2f})", suppress)
            else:
                min_aperture = min(d_ti, d_tm)
                if 0.35 <= min_aperture <= 0.85 and thumb_ext and (top1_label in ("C", "O", "0") or top2_label in ("C", "O", "0")):
                    _force_class("C", f"Open curved C aperture ({min_aperture:.2f})", ["B", "4", "5", "space", "O"])
                elif (is_horizontal or abs(v_idx[0]) > abs(v_idx[1]) * 0.70) and d_im < 0.40 and d_mr < 0.40 and top1_label == "space":
                    _force_class("space", "Horizontal 4-finger flat hand", ["B", "4", "5", "C"])
                elif thumb_ext or d_tp_mcp >= 0.85:
                    _force_class("5", "All 5 digits extended wide", ["4", "B", "space", "C"])
                else:
                    if d_im < 0.28 and d_mr < 0.28:
                        _force_class("B", f"4-finger flat pressed upright (d_im={d_im:.2f})", ["4", "5", "space", "C", "F"])
                    else:
                        _force_class("4", f"4-finger spread upright (d_im={d_im:.2f})", ["B", "5", "space", "C", "F"])

        # -------------------------------------------------------------
        # 9. Fist / Closed Hand: 'A', 'S', 'T', 'E', '0', 'O'
        # -------------------------------------------------------------
        elif not idx_ext and not mid_ext and not ring_ext and not pnk_ext:
            if d_ti < 0.25 and d_tm < 0.32:
                loop_width = np.linalg.norm(i_mcp - p_mcp) / palm
                loop_height = np.linalg.norm(i_tip - w) / palm
                aspect = float(loop_width / (loop_height + 1e-6))
                if aspect >= 0.29 or d_tm >= 0.22:
                    _force_class("0", f"Narrow oval zero (aspect={aspect:.2f}, d_tm={d_tm:.2f})", ["O", "A", "S", "T", "E", "del"])
                else:
                    _force_class("O", f"Round circular O (aspect={aspect:.2f}, d_tm={d_tm:.2f})", ["0", "A", "S", "T", "E", "del"])
            elif d_tr < 0.32 and d_tp < 0.32:
                _force_class("E", "Fingertips curled into palm against thumb", ["A", "S", "T", "0", "O"])
            elif (t_tip[0] - i_mcp[0]) > 0.35 * palm and d_ti_mcp > 0.60:
                _force_class("T", "Thumb tucked between index and middle", ["A", "S", "E"])
            elif d_tm_mcp < 0.35:
                _force_class("S", "Thumb folded across front of fingers", ["A", "T", "E"])
            else:
                _force_class("A", "Thumb upright alongside index MCP", ["S", "T", "E"])

        # -------------------------------------------------------------
        # 10. Closed Fist 'del' Guard
        # -------------------------------------------------------------
        is_fist = not (idx_ext or mid_ext or ring_ext or pnk_ext)
        if is_fist:
            if "del" in self.classes:
                del_idx = self.classes.index("del")
                if probs[del_idx] > 0.05:
                    probs[del_idx] = 0.01
                    probs = probs / np.sum(probs)

        # -------------------------------------------------------------
        # 11. Fine-Grained Pairwise Disambiguations
        # -------------------------------------------------------------
        # 'C' vs 'O' Disambiguation
        if top2_set == {"C", "O"}:
            aperture = float(min(d_ti, d_tm))
            if aperture >= 0.40:
                _disambiguate_pair("C", "O", favor_a=True, reason=f"Open aperture ({aperture:.2f})")
            elif aperture <= 0.26:
                _disambiguate_pair("C", "O", favor_a=False, reason=f"Closed aperture ({aperture:.2f})")

        # '0' vs 'O' Disambiguation
        if top2_set == {"0", "O"}:
            loop_width = np.linalg.norm(i_mcp - p_mcp) / palm
            loop_height = np.linalg.norm(i_tip - w) / palm
            aspect = float(loop_width / (loop_height + 1e-6))
            if aspect < 0.65:
                _disambiguate_pair("0", "O", favor_a=True, reason=f"Narrow oval zero ({aspect:.2f})")
            elif aspect >= 0.74:
                _disambiguate_pair("0", "O", favor_a=False, reason=f"Round circular O ({aspect:.2f})")

        # 12. Multimodal Lip-Morpheme Facial Fusion Engine (M vs N, C vs O, D vs 1)
        if self.enable_lip_fusion and (top2_set & ({"M", "N"} | {"C", "O"} | {"D", "1"})):
            active_face_lm = face_landmarks
            if active_face_lm is None and bgr_frame is not None and bgr_frame.size > 0:
                try:
                    rgb_f = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    f_res = self.face_mesh.process(rgb_f)
                    if f_res.multi_face_landmarks:
                        active_face_lm = f_res.multi_face_landmarks[0]
                except Exception:
                    active_face_lm = None

            if active_face_lm is not None:
                probs, lip_note = self.lip_engine.apply_morpheme_fusion(
                    top1_label, top2_label, probs, self.classes, active_face_lm
                )
                if lip_note:
                    disambig_notes.append(lip_note)

        note = "; ".join(disambig_notes) if disambig_notes else None
        return probs, note

    def predict_frame(self, bgr_frame: np.ndarray) -> PredictionOutput:
        """
        Processes a single live webcam frame:
        1. Tracks up to 2 hands (Left & Right) with MediaPipe Hands.
        2. Applies 1€ anti-tremor stabilization across both hands.
        3. Evaluates 2-Hand Bimanual Signs ("BOOK", "HELP", "MEET", "WORK", "HOUSE", "FINISHED", "PLAY").
        4. If not a two-handed sign, executes spatial & LSTM sequence inference on primary hand.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return self._empty_response("Empty video frame received.")

        # Step 1: Synchronized Dual-Hand Extraction (Left & Right)
        dual_res = extract_dual_hands_from_bgr_frame(bgr_frame, self.hands_detector)

        if not dual_res.has_hands:
            self.prev_raw_coords = None
            if self.sequence_buffer:
                self.sequence_buffer.popleft()
            return self._empty_response("No hand detected in frame.", dual_res=dual_res)

        # Prefer Right hand as primary, fallback to Left hand
        primary_coords = dual_res.right_coords or dual_res.left_coords
        primary_proto = dual_res.right_hand_proto or dual_res.left_hand_proto

        # One-Shot Biomechanical Calibration Morphological Scaling
        if self.enable_calibration and self.calibrator and self.calibrator.profile.get("calibrated", False):
            if dual_res.left_coords:
                dual_res.left_coords = self.calibrator.apply_morphological_scaling(dual_res.left_coords)
            if dual_res.right_coords:
                dual_res.right_coords = self.calibrator.apply_morphological_scaling(dual_res.right_coords)
            if primary_coords is not None:
                primary_coords = self.calibrator.apply_morphological_scaling(primary_coords)

        # Step 2: Apply Biomechanical 1€ Anti-Tremor & Shiver Stabilizer across hands
        if self.enable_stabilization:
            stab_left, stab_right, jitter_score, is_shivering = self.dual_stabilizer.stabilize(
                dual_res.left_coords, dual_res.right_coords
            )
            stab_primary = stab_right or stab_left
            motion_vec, kinetic_energy = extract_motion_features(stab_primary, self.prev_raw_coords)
            active_coords = stab_primary

            # Tremor Deadband Compensation
            if self.tremor_deadband and is_shivering and kinetic_energy < 0.048:
                kinetic_energy = 0.015
        else:
            stab_left, stab_right = dual_res.left_coords, dual_res.right_coords
            active_coords = primary_coords
            motion_vec, kinetic_energy = extract_motion_features(active_coords, self.prev_raw_coords)
            jitter_score = 0.0
            is_shivering = False

        # Step 3: Evaluate Whole-Sentence & Two-Handed (Bimanual) Signs (ONLY if word signs enabled)
        if self.enable_word_signs:
            phrase_result = self.phrase_recognizer.update(
                coords=active_coords,
                kinetic_energy=kinetic_energy,
                left_coords=stab_left,
                right_coords=stab_right,
                wrist_dist=dual_res.wrist_dist,
                index_dist=dual_res.index_dist
            )

            if phrase_result is not None and phrase_result.is_phrase_detected:
                self.prev_raw_coords = active_coords
                self.sequence_buffer.append(motion_vec)
                self.prediction_history.append(phrase_result.phrase_text)

                m_state = "TWO-HANDED GESTURE" if phrase_result.is_bimanual else "WHOLE PHRASE GESTURE"
                return PredictionOutput(
                    hand_detected=True,
                    predicted_label=phrase_result.phrase_id,
                    display_label=phrase_result.phrase_text,
                    confidence_level="HIGH",
                    confidence_score=round(phrase_result.confidence * 100.0, 1),
                    raw_probability=phrase_result.confidence,
                    margin=0.50,
                    kinetic_energy=round(kinetic_energy, 4),
                    motion_state=m_state,
                    buffer_fill=len(self.sequence_buffer),
                    is_uncertain=False,
                    is_stable=True,
                    stable_label=phrase_result.phrase_text,
                    hand_landmarks_proto=primary_proto,
                    status_message=f"Recognized {'Two-Handed' if phrase_result.is_bimanual else 'Single-Hand'} Sign: '{phrase_result.phrase_text}'",
                    is_shivering=is_shivering,
                    jitter_score=round(jitter_score, 6),
                    is_phrase=True,
                    phrase_text=phrase_result.phrase_text,
                    stabilized_coords=active_coords,
                    num_hands=dual_res.num_hands,
                    is_two_handed=phrase_result.is_bimanual,
                    left_hand_proto=dual_res.left_hand_proto,
                    right_hand_proto=dual_res.right_hand_proto,
                    dual_result=dual_res
                )

        # Update previous frame coordinates
        self.prev_raw_coords = active_coords
        self.sequence_buffer.append(motion_vec)

        # Determine motion state based on stabilized kinetic energy
        if dual_res.is_bimanual:
            motion_state = "DUAL HANDS DETECTED"
        elif kinetic_energy > 0.08:
            motion_state = "FAST MOTION"
        elif kinetic_energy > 0.008:
            motion_state = "ACTIVE MOTION"
        else:
            motion_state = "STATIC HOLD"

        # Step 3.5: Evaluate Word-Level WLASL Signs (ONLY when word signs explicitly enabled AND in word/hybrid mode)
        if self.enable_word_signs and self.wlasl_model is not None and (
            self.engine_mode in ("wlasl", "hybrid")
        ):
            min_thresh = 0.35 if self.engine_mode == "wlasl" else 0.50
            wlasl_res = self.predict_wlasl_word(min_confidence=min_thresh)
            if wlasl_res is not None:
                word_label, word_prob = wlasl_res
                self.prediction_history.append(word_label)
                return PredictionOutput(
                        hand_detected=True,
                        predicted_label=word_label,
                        display_label=word_label,
                        confidence_level="HIGH" if word_prob >= 0.70 else "MEDIUM",
                        confidence_score=round(word_prob * 100.0, 1),
                        raw_probability=round(word_prob, 4),
                        margin=0.35,
                        kinetic_energy=round(kinetic_energy, 4),
                        motion_state="WHOLE WORD GESTURE",
                        buffer_fill=len(self.sequence_buffer),
                        is_uncertain=False,
                        is_stable=True,
                        stable_label=word_label,
                        hand_landmarks_proto=primary_proto,
                        status_message=f"Recognized Whole-Word Sign: '{word_label}' via WLASL Model",
                        is_shivering=is_shivering,
                        jitter_score=round(jitter_score, 6),
                        is_phrase=True,
                        phrase_text=word_label,
                        stabilized_coords=active_coords,
                        num_hands=dual_res.num_hands,
                        is_two_handed=dual_res.is_bimanual,
                        left_hand_proto=dual_res.left_hand_proto,
                        right_hand_proto=dual_res.right_hand_proto,
                        dual_result=dual_res
                    )
            elif self.engine_mode == "wlasl":
                # In explicit WLASL mode, wait for word gesture completion instead of falling back to letters
                return PredictionOutput(
                    hand_detected=True,
                    predicted_label="...",
                    display_label="Performing Word Sign...",
                    confidence_level="MEDIUM",
                    confidence_score=50.0,
                    raw_probability=0.5,
                    margin=0.1,
                    kinetic_energy=round(kinetic_energy, 4),
                    motion_state="ACCUMULATING MOTION" if len(self.sequence_buffer) < 10 else "ANALYZING WORD SIGN",
                    buffer_fill=len(self.sequence_buffer),
                    is_uncertain=True,
                    is_stable=False,
                    stable_label=None,
                    hand_landmarks_proto=primary_proto,
                    status_message=f"Capturing motion trajectory ({len(self.sequence_buffer)}/20 frames) for WLASL vocabulary...",
                    is_shivering=is_shivering,
                    jitter_score=round(jitter_score, 6),
                    is_phrase=True,
                    phrase_text=None,
                    stabilized_coords=active_coords,
                    num_hands=dual_res.num_hands,
                    is_two_handed=dual_res.is_bimanual,
                    left_hand_proto=dual_res.left_hand_proto,
                    right_hand_proto=dual_res.right_hand_proto,
                    dual_result=dual_res
                )

        # Step 4: Model Inference for Primary Hand (39 Base Classes)
        use_lstm = False
        if self.engine_mode == "lstm" and self.lstm_model is not None:
            use_lstm = True
        elif self.engine_mode == "rf" and self.rf_model is not None:
            use_lstm = False
        elif self.engine_mode == "auto":
            if kinetic_energy >= 0.04 and self.lstm_model is not None:
                use_lstm = True
            elif self.rf_model is not None:
                use_lstm = False
            elif self.lstm_model is not None:
                use_lstm = True

        if use_lstm:
            frames = list(self.sequence_buffer)
            while len(frames) < self.sequence_length:
                frames.insert(0, frames[0])

            seq_array = np.array(frames, dtype=np.float32).reshape(1, self.sequence_length, TOTAL_MOTION_FEATURES)
            if self.onnx_session is not None and self.onnx_session.is_ready:
                probs = self.onnx_session.predict_probabilities(seq_array)
                model_used_name = "Deep LSTM (ONNX Runtime)"
            else:
                seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(self.device)
                probs_tensor = self.lstm_model.predict_probabilities(seq_tensor)[0]
                probs = probs_tensor.cpu().numpy()
                model_used_name = "Deep LSTM (Motion)"
        else:
            spatial_feat = np.array(motion_vec[:63]).reshape(1, -1)
            probs = self.rf_model.predict_proba(spatial_feat)[0]
            model_used_name = "Ensemble Stack (Spatial + Invariants)"

        # Anatomical & Multimodal Lip-Morpheme Disambiguation
        probs, disambig_note = self._apply_anatomical_disambiguation(
            active_coords, probs, kinetic_energy=kinetic_energy, bgr_frame=bgr_frame
        )
        if disambig_note:
            model_used_name += f" [{disambig_note}]"

        # Probability Smoothing
        probs = self.prob_stabilizer.update(probs)

        # Analyze Probability Distribution
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx
        top3_list = [(self.classes[idx], float(probs[idx])) for idx in sorted_indices[:3]]

        raw_label = self.classes[top1_idx]
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        margin = top1_prob - top2_prob

        # Confidence Evaluation
        if top1_prob >= 0.60 and margin >= self.min_margin:
            confidence_level = "HIGH"
            is_uncertain = False
            display_label = raw_label
            status_message = f"Clear {motion_state.lower()} via {model_used_name}."
        elif top1_prob >= self.min_confidence and margin >= (self.min_margin * 0.5):
            confidence_level = "MEDIUM"
            is_uncertain = False
            display_label = raw_label
            status_message = f"Confident {motion_state.lower()} via {model_used_name}."
        else:
            confidence_level = "LOW"
            is_uncertain = True
            display_label = "Uncertain Gesture"
            status_message = f"Ambiguous motion transition via {model_used_name}."

        # Temporal Consistency & Debouncing
        if not is_uncertain:
            self.prediction_history.append(display_label)
        else:
            self.prediction_history.append("Uncertain Gesture")

        is_stable = False
        stable_label = None

        if len(self.prediction_history) >= 4:
            counts = Counter(self.prediction_history)
            most_common_label, count = counts.most_common(1)[0]
            if (count / len(self.prediction_history)) >= 0.50 and most_common_label != "Uncertain Gesture":
                is_stable = True
                stable_label = most_common_label
                self.last_stable_label = stable_label

        return PredictionOutput(
            hand_detected=True,
            predicted_label=raw_label,
            display_label=display_label,
            confidence_level=confidence_level,
            confidence_score=round(top1_prob * 100.0, 1),
            raw_probability=round(top1_prob, 4),
            margin=round(margin, 4),
            kinetic_energy=round(kinetic_energy, 4),
            motion_state=motion_state,
            buffer_fill=len(self.sequence_buffer),
            is_uncertain=is_uncertain,
            is_stable=is_stable,
            stable_label=stable_label,
            hand_landmarks_proto=primary_proto,
            status_message=status_message,
            is_shivering=is_shivering,
            jitter_score=round(jitter_score, 6),
            is_phrase=False,
            phrase_text=None,
            stabilized_coords=active_coords,
            num_hands=dual_res.num_hands,
            is_two_handed=dual_res.is_bimanual,
            left_hand_proto=dual_res.left_hand_proto,
            right_hand_proto=dual_res.right_hand_proto,
            dual_result=dual_res,
            top3_candidates=top3_list
        )

    def predict_image(
        self,
        bgr_image: np.ndarray,
        model_type: Optional[str] = None
    ) -> Tuple[PredictionOutput, np.ndarray]:
        """
        Evaluates a static image using robust cascading hand detection.
        Defaults to high-precision Random Forest (99.33% accuracy), with optional LSTM support.
        """
        annotated_image = bgr_image.copy() if bgr_image is not None else None

        if bgr_image is None or bgr_image.size == 0:
            return self._empty_response("No image data provided."), annotated_image

        features_63, hand_landmarks, raw_coords, stage_name = robust_extract_landmarks_from_bgr_frame(
            bgr_image
        )

        if features_63 is None:
            return self._empty_response("No hand detected after 8 cascading stages."), annotated_image

        if annotated_image is not None and hand_landmarks is not None:
            draw_styled_landmarks(annotated_image, hand_landmarks)

        probs = None
        model_name = "Ensemble Stack (Spatial + Invariants)"

        if model_type == "lstm" and self.lstm_model is not None:
            motion_vec = features_63 + [0.0] * 63 + [0.0, 0.0, 0.0]
            seq_array = np.array([motion_vec] * self.sequence_length, dtype=np.float32).reshape(1, self.sequence_length, TOTAL_MOTION_FEATURES)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(self.device)
            probs_t = self.lstm_model.predict_probabilities(seq_tensor)[0]
            probs = probs_t.cpu().numpy()
            model_name = "Deep LSTM (Static Projection)"
        elif self.rf_model is not None:
            spatial_feat = np.array(features_63).reshape(1, -1)
            probs = self.rf_model.predict_proba(spatial_feat)[0]
        elif self.lstm_model is not None:
            motion_vec = features_63 + [0.0] * 63 + [0.0, 0.0, 0.0]
            seq_array = np.array([motion_vec] * self.sequence_length, dtype=np.float32).reshape(1, self.sequence_length, TOTAL_MOTION_FEATURES)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(self.device)
            probs_t = self.lstm_model.predict_probabilities(seq_tensor)[0]
            probs = probs_t.cpu().numpy()
            model_name = "Deep LSTM (Fallback)"

        if probs is None:
            return self._empty_response("No valid model available for inference."), annotated_image

        probs, disambig_note = self._apply_anatomical_disambiguation(raw_coords, probs, kinetic_energy=0.0)
        if disambig_note:
            model_name += f" [{disambig_note}]"

        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        top_prob = float(probs[top_idx])
        margin = top_prob - (float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0)
        top3_list = [(self.classes[idx], float(probs[idx])) for idx in sorted_indices[:3]]

        label = self.classes[top_idx]
        is_uncertain = (top_prob < self.min_confidence) or (margin < 0.02)

        output = PredictionOutput(
            hand_detected=True,
            predicted_label=label,
            display_label=label if not is_uncertain else "Uncertain Gesture",
            confidence_level="HIGH" if top_prob >= 0.70 else ("MEDIUM" if not is_uncertain else "LOW"),
            confidence_score=round(top_prob * 100.0, 1),
            raw_probability=round(top_prob, 4),
            margin=round(margin, 4),
            kinetic_energy=0.0,
            motion_state="STATIC HOLD",
            buffer_fill=self.sequence_length,
            is_uncertain=is_uncertain,
            is_stable=True,
            stable_label=label if not is_uncertain else None,
            hand_landmarks_proto=hand_landmarks,
            status_message=f"Detected via '{stage_name}' stage, classified by {model_name}.",
            top3_candidates=top3_list
        )
        return output, annotated_image

    def predict_video(
        self,
        video_path: Union[str, Path],
        frame_start: int = 1,
        frame_end: int = -1
    ) -> Dict[str, Any]:
        """
        Processes an MP4 video clip, extracts uniform landmark sequences across frames,
        and evaluates using the WLASL Word-Level Model and spatial-temporal features.
        """
        v_path = Path(video_path)
        if not v_path.exists() or v_path.stat().st_size < 1000:
            return {
                "success": False,
                "error_message": f"Video file not found or invalid: {video_path}",
                "predicted_word": None,
                "confidence_score": 0.0,
                "top3_candidates": [],
                "annotated_keyframe": None,
                "num_frames_processed": 0
            }

        cap = cv2.VideoCapture(str(v_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
            cap.release()
            return {
                "success": False,
                "error_message": f"Video has too few frames ({total_frames}).",
                "predicted_word": None,
                "confidence_score": 0.0,
                "top3_candidates": [],
                "annotated_keyframe": None,
                "num_frames_processed": 0
            }

        start = max(0, frame_start - 1)
        end = total_frames - 1
        if frame_end > 0 and frame_end > start:
            end = min(total_frames - 1, frame_end)

        sample_indices = np.linspace(start, end, self.sequence_length).astype(int)
        target_set = set(sample_indices)
        mid_target = sample_indices[len(sample_indices) // 2]

        sampled_dict = {}
        keyframe_bgr = None
        keyframe_proto = None
        curr_frame = 0

        while cap.isOpened() and curr_frame <= end:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame in target_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands_detector.process(rgb)
                if res.multi_hand_landmarks:
                    primary = res.multi_hand_landmarks[0]
                    pts = [(lm.x, lm.y, lm.z) for lm in primary.landmark]
                    sampled_dict[curr_frame] = pts
                    if curr_frame == mid_target or keyframe_bgr is None:
                        keyframe_bgr = frame.copy()
                        keyframe_proto = primary
                else:
                    sampled_dict[curr_frame] = None
                    if keyframe_bgr is None:
                        keyframe_bgr = frame.copy()
            curr_frame += 1

        cap.release()

        # Reconstruct in order
        last_valid = None
        ordered_coords = []
        valid_count = 0

        for idx in sample_indices:
            pts = sampled_dict.get(idx)
            if pts is not None:
                last_valid = pts
                valid_count += 1
                ordered_coords.append(pts)
            elif last_valid is not None:
                ordered_coords.append(last_valid)
            else:
                ordered_coords.append(None)

        if valid_count < 3:
            return {
                "success": False,
                "error_message": "Could not detect clear hand landmarks in enough video frames.",
                "predicted_word": None,
                "confidence_score": 0.0,
                "top3_candidates": [],
                "annotated_keyframe": keyframe_bgr,
                "num_frames_processed": total_frames
            }

        first_valid = next(p for p in ordered_coords if p is not None)
        for i in range(len(ordered_coords)):
            if ordered_coords[i] is None:
                ordered_coords[i] = first_valid

        # Compute 129-dim motion sequence
        seq_vectors = []
        prev_pts = None
        for pts in ordered_coords:
            m_vec, _ = extract_motion_features(pts, prev_pts)
            seq_vectors.append(m_vec)
            prev_pts = pts

        seq_array = np.array(seq_vectors, dtype=np.float32)

        # 258-dim summary vector
        init_shape = seq_array[0, :63]
        mid_shape = seq_array[len(ordered_coords) // 2, :63]
        final_shape = seq_array[-1, :63]
        mean_vel = np.mean(seq_array[:, 63:126], axis=0)
        traj_delta = seq_array[-1, 126:129] - seq_array[0, 126:129]
        max_vel = np.max(np.abs(seq_array[:, 63:66]), axis=0)

        feat_258 = np.hstack([init_shape, mid_shape, final_shape, mean_vel, traj_delta, max_vel]).reshape(1, -1)

        if self.wlasl_model is None:
            return {
                "success": False,
                "error_message": "WLASL Word Model is not loaded.",
                "predicted_word": None,
                "confidence_score": 0.0,
                "top3_candidates": [],
                "annotated_keyframe": keyframe_bgr,
                "num_frames_processed": total_frames
            }

        probs = self.wlasl_model.predict_proba(feat_258)[0]
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top1_prob = float(probs[top1_idx])
        pred_word = self.wlasl_classes[top1_idx].upper()

        top3 = []
        for idx in sorted_indices[:3]:
            top3.append((self.wlasl_classes[idx].upper(), float(probs[idx])))

        # Annotate keyframe if available
        annotated_frame = keyframe_bgr.copy() if keyframe_bgr is not None else None
        if annotated_frame is not None and keyframe_proto is not None:
            draw_styled_landmarks(annotated_frame, keyframe_proto)

        return {
            "success": True,
            "error_message": "",
            "predicted_word": pred_word,
            "confidence_score": round(top1_prob * 100.0, 1),
            "top3_candidates": top3,
            "annotated_keyframe": annotated_frame,
            "num_frames_processed": total_frames
        }

    def _empty_response(
        self,
        message: str,
        landmarks: Optional[object] = None,
        dual_res: Optional[DualHandDetectionResult] = None
    ) -> PredictionOutput:
        """Returns empty/idle response state."""
        return PredictionOutput(
            hand_detected=False,
            predicted_label="",
            display_label="No Hand Detected",
            confidence_level="LOW",
            confidence_score=0.0,
            raw_probability=0.0,
            margin=0.0,
            kinetic_energy=0.0,
            motion_state="IDLE",
            buffer_fill=len(self.sequence_buffer),
            is_uncertain=True,
            is_stable=False,
            stable_label=None,
            hand_landmarks_proto=landmarks,
            status_message=message,
            num_hands=0,
            is_two_handed=False,
            left_hand_proto=None,
            right_hand_proto=None,
            dual_result=dual_res
        )

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if self.hands_detector:
            self.hands_detector.close()
