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
import sys
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
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
        min_confidence: float = 0.35,
        min_margin: float = 0.04,
        engine_mode: str = "auto",
        enable_stabilization: bool = True,
        stabilizer_min_cutoff: float = 0.8,
        stabilizer_beta: float = 0.015,
        device: str = "cpu"
    ):
        self.lstm_model_path = Path(lstm_model_path)
        self.encoder_path = Path(encoder_path)
        self.rf_model_path = Path(rf_model_path)
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.engine_mode = engine_mode  # 'auto', 'rf', 'lstm'
        self.device = torch.device(device)
        self.enable_stabilization = enable_stabilization

        # Biomechanical Stabilizers (One-Euro Filter across Left and Right hands)
        self.landmark_stabilizer = LandmarkStabilizer(
            min_cutoff=stabilizer_min_cutoff,
            beta=stabilizer_beta
        )
        self.dual_stabilizer = DualLandmarkStabilizer(
            min_cutoff=stabilizer_min_cutoff,
            beta=stabilizer_beta
        )
        self.prob_stabilizer = ProbabilityStabilizer(alpha=0.65)

        # Whole-Sentence & Dual-Hand Recognition Engine
        self.phrase_recognizer = EnterprisePhraseRecognizer()

        # Temporal Sequence Window for LSTM (stores last N frames of 129 features)
        self.sequence_buffer: deque = deque(maxlen=sequence_length)
        self.prediction_history: deque = deque(maxlen=10)
        self.prev_raw_coords: Optional[List[Tuple[float, float, float]]] = None
        self.last_stable_label: Optional[str] = None
        self.tremor_deadband: bool = True

        # Load models
        self.lstm_model: Optional[MotionSignLSTM] = None
        self.rf_model = None
        self.label_encoder = None
        self.classes: List[str] = []
        self._load_models()

        # Initialize MediaPipe Hands supporting up to 2 hands (Dual Hand / Bimanual)
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )

    def _load_models(self) -> None:
        """Loads both LSTM checkpoint and Random Forest model if available."""
        if not self.encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at '{self.encoder_path}'. "
                "Please run training first: 'python training/train_lstm.py'"
            )

        self.label_encoder = joblib.load(self.encoder_path)
        self.classes = [str(c) for c in self.label_encoder.classes_]

        # Load LSTM model if exists
        if self.lstm_model_path.exists():
            try:
                self.lstm_model = load_lstm_checkpoint(
                    checkpoint_path=str(self.lstm_model_path),
                    input_size=TOTAL_MOTION_FEATURES,
                    num_classes=len(self.classes),
                    device=self.device
                )
                print(f"[+] SignPredictor: Loaded Deep Learning LSTM model from {self.lstm_model_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load LSTM model: {e}")

        # Load Random Forest model if exists
        if self.rf_model_path.exists():
            try:
                self.rf_model = joblib.load(self.rf_model_path)
                print(f"[+] SignPredictor: Loaded Random Forest baseline from {self.rf_model_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load RF model: {e}")

        if self.lstm_model is None and self.rf_model is None:
            raise FileNotFoundError(
                f"Neither LSTM checkpoint '{self.lstm_model_path}' nor RF model '{self.rf_model_path}' found."
            )

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
        kinetic_energy: float = 0.0
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
        t_tip = np.array(raw_coords[4])
        i_tip = np.array(raw_coords[8])
        i_mcp = np.array(raw_coords[5])
        m_tip = np.array(raw_coords[12])
        r_tip = np.array(raw_coords[16])
        p_tip = np.array(raw_coords[20])
        p_mcp = np.array(raw_coords[17])

        # Distances normalized by palm length
        d_im = np.linalg.norm(i_tip - m_tip) / palm
        d_mr = np.linalg.norm(m_tip - r_tip) / palm
        d_tp = np.linalg.norm(t_tip - p_tip) / palm
        d_ti = np.linalg.norm(t_tip - i_tip) / palm
        d_tm = np.linalg.norm(t_tip - m_tip) / palm

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
            if margin > 0.08:
                # Ensemble is already decisive (margin > 8%), respect the learned boundary
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

        # 1. 'W' vs '6' Disambiguation
        if top2_set == {"W", "6"}:
            if d_im >= 0.23 and d_mr >= 0.23:
                _disambiguate_pair("W", "6", favor_a=True, reason=f"Finger spread ({d_im:.2f}, {d_mr:.2f})")
            elif d_im < 0.20 and d_tp < 0.30:
                _disambiguate_pair("W", "6", favor_a=False, reason=f"Parallel stack & thumb-pinky ({d_tp:.2f})")

        # 2. 'U' vs 'V' Disambiguation
        if top2_set == {"U", "V"}:
            if d_im < 0.17:
                _disambiguate_pair("U", "V", favor_a=True, reason=f"Touching fingers ({d_im:.2f})")
            elif d_im >= 0.21:
                _disambiguate_pair("U", "V", favor_a=False, reason=f"Spread V-shape ({d_im:.2f})")

        # 3. 'U' vs '2' Disambiguation
        if top2_set == {"U", "2"}:
            if d_im < 0.17:
                _disambiguate_pair("U", "2", favor_a=True, reason=f"Touching fingers ({d_im:.2f})")
            elif d_im >= 0.21:
                _disambiguate_pair("U", "2", favor_a=False, reason=f"Spread fingers ({d_im:.2f})")

        # 4. 'C' vs 'O' Disambiguation
        if top2_set == {"C", "O"}:
            aperture = float(min(d_ti, d_tm))
            if aperture >= 0.42:
                _disambiguate_pair("C", "O", favor_a=True, reason=f"Open aperture ({aperture:.2f})")
            elif aperture <= 0.26:
                _disambiguate_pair("C", "O", favor_a=False, reason=f"Closed aperture ({aperture:.2f})")

        # 5. '0' vs 'O' Disambiguation
        if top2_set == {"0", "O"}:
            loop_width = np.linalg.norm(i_mcp - p_mcp) / palm
            loop_height = np.linalg.norm(i_tip - w) / palm
            aspect = loop_width / (loop_height + 1e-6)
            if aspect < 0.65:
                _disambiguate_pair("0", "O", favor_a=True, reason=f"Narrow oval zero ({aspect:.2f})")
            elif aspect >= 0.74:
                _disambiguate_pair("0", "O", favor_a=False, reason=f"Round circular O ({aspect:.2f})")

        # 6. 'Z' vs '1' Dynamic Motion Gating
        if top2_set == {"Z", "1"}:
            if kinetic_energy >= 0.035:
                _disambiguate_pair("Z", "1", favor_a=True, reason=f"Motion velocity ({kinetic_energy:.3f})")
            elif 0.0 < kinetic_energy < 0.022:
                _disambiguate_pair("Z", "1", favor_a=False, reason=f"Static point hold ({kinetic_energy:.3f})")

        # 7. 'J' vs 'I' Dynamic Motion Gating
        if top2_set == {"J", "I"}:
            if kinetic_energy >= 0.035:
                _disambiguate_pair("J", "I", favor_a=True, reason=f"Curve swoop motion ({kinetic_energy:.3f})")
            elif 0.0 < kinetic_energy < 0.022:
                _disambiguate_pair("J", "I", favor_a=False, reason=f"Static pinky hold ({kinetic_energy:.3f})")

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

        # Step 3: Evaluate Whole-Sentence & Two-Handed (Bimanual) Signs
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
        elif kinetic_energy > 0.15:
            motion_state = "FAST MOTION"
        elif kinetic_energy > 0.04:
            motion_state = "ACTIVE MOTION"
        else:
            motion_state = "STATIC HOLD"

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
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(self.device)
            probs_tensor = self.lstm_model.predict_probabilities(seq_tensor)[0]
            probs = probs_tensor.cpu().numpy()
            model_used_name = "Deep LSTM (Motion)"
        else:
            spatial_feat = np.array(motion_vec[:63]).reshape(1, -1)
            probs = self.rf_model.predict_proba(spatial_feat)[0]
            model_used_name = "Ensemble Stack (Spatial + Invariants)"

        # Anatomical Disambiguation
        probs, disambig_note = self._apply_anatomical_disambiguation(active_coords, probs, kinetic_energy=kinetic_energy)
        if disambig_note:
            model_used_name += f" [{disambig_note}]"

        # Probability Smoothing
        probs = self.prob_stabilizer.update(probs)

        # Analyze Probability Distribution
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx

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
            dual_result=dual_res
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

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        sorted_probs = np.sort(probs)[::-1]
        margin = top_prob - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

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
            status_message=f"Detected via '{stage_name}' stage, classified by {model_name}."
        )
        return output, annotated_image

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
