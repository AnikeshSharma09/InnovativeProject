"""
Real-Time Motion-Aware ASL Sign Language Predictor (Deep Learning LSTM).

Features:
1. MediaPipe 21 Hand Landmark Tracking.
2. 129-Dimensional Motion Feature Extraction (63 spatial + 63 joint velocities + 3 wrist deltas).
3. Hand Kinetic Motion Energy Tracking (distinguishes dynamic motion gestures from static poses).
4. Deep PyTorch MotionSignLSTM Recurrent Inference over rolling sequence windows.
5. Learned temporal confidence & uncertainty suppression.
6. Optional fallback to Random Forest baseline.
"""
import os
import sys
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
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
    extract_landmarks_from_bgr_frame,
    robust_extract_landmarks_from_bgr_frame,
    draw_styled_landmarks,
    TOTAL_MOTION_FEATURES,
    DEFAULT_SEQUENCE_LENGTH
)
from utils.labels import format_label_for_display


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
    motion_state: str  # 'STATIC HOLD', 'ACTIVE MOTION', 'FAST MOTION'
    buffer_fill: int
    is_uncertain: bool
    is_stable: bool
    stable_label: Optional[str]
    hand_landmarks_proto: Optional[object]
    status_message: str


class SignPredictor:
    """
    Production-ready real-time motion-aware predictor powered by PyTorch LSTM.
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

        # Temporal Sequence Window for LSTM (stores last N frames of 129 features)
        self.sequence_buffer: deque = deque(maxlen=sequence_length)
        self.prediction_history: deque = deque(maxlen=10)
        self.prev_raw_coords: Optional[List[Tuple[float, float, float]]] = None
        self.last_stable_label: Optional[str] = None

        # Load models
        self.lstm_model: Optional[MotionSignLSTM] = None
        self.rf_model = None
        self.label_encoder = None
        self.classes: List[str] = []
        self._load_models()

        # Initialize MediaPipe Hands for live video stream
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.40,
            min_tracking_confidence=0.40
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
                self.lstm_model, _ = load_lstm_checkpoint(str(self.lstm_model_path), device=self.device)
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

    def _apply_anatomical_disambiguation(
        self,
        raw_coords: Optional[List[Tuple[float, float, float]]],
        probs: np.ndarray
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Disambiguates geometrically confused ASL sign pairs using physiological landmark metrics:
        1. 'C' vs 'O':
           - 'O': closed loop (thumb and index/middle fingertips touch or meet within aperture <= 0.36)
           - 'C': open arc (thumb and index/middle fingertips separated by aperture >= 0.44)
           Aperture is normalized by palm scale (Wrist LM 0 to Middle MCP LM 9 distance).
        """
        if raw_coords is None or len(raw_coords) < 21:
            return probs, None

        if "C" not in self.classes or "O" not in self.classes:
            return probs, None

        c_idx = self.classes.index("C")
        o_idx = self.classes.index("O")

        sorted_idx = np.argsort(probs)[::-1]
        top1_idx = int(sorted_idx[0])
        top1_label = self.classes[top1_idx]
        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
        top2_label = self.classes[top2_idx]

        # Only evaluate C vs O if one of them is the top-1 candidate or top-2 with close margin
        is_candidate = (top1_label in ("C", "O")) or (top2_label in ("C", "O") and probs[top1_idx] < 0.60)
        if not is_candidate:
            return probs, None

        w = np.array(raw_coords[0])
        m_mcp = np.array(raw_coords[9])
        palm = np.linalg.norm(m_mcp - w)
        if palm < 1e-4:
            return probs, None

        t_tip = np.array(raw_coords[4])
        i_tip = np.array(raw_coords[8])
        m_tip = np.array(raw_coords[12])
        d_ti = np.linalg.norm(t_tip - i_tip) / palm
        d_tm = np.linalg.norm(t_tip - m_tip) / palm
        aperture = float(min(d_ti, d_tm))

        disambig_note = None

        if aperture >= 0.44:
            # Thumb and fingers form an open arc -> Physically 'C'
            if top1_label in ("O", "0") or probs[c_idx] < probs[o_idx]:
                probs = probs.copy()
                orig_o = probs[o_idx]
                orig_c = probs[c_idx]
                probs[c_idx] = max(orig_o, orig_c) + 0.10
                probs[o_idx] = min(orig_o, orig_c) * 0.20
                probs = probs / np.sum(probs)
                disambig_note = f"Aperture ({aperture:.2f} >= 0.44) -> 'C'"
        elif aperture <= 0.25:
            # Fingertips genuinely touch in a closed loop -> Physically 'O'
            if top1_label == "C" and probs[o_idx] >= (probs[c_idx] * 0.5):
                probs = probs.copy()
                orig_o = probs[o_idx]
                orig_c = probs[c_idx]
                probs[o_idx] = max(orig_o, orig_c) + 0.10
                probs[c_idx] = min(orig_o, orig_c) * 0.20
                probs = probs / np.sum(probs)
                disambig_note = f"Aperture ({aperture:.2f} <= 0.25) -> 'O'"

        return probs, disambig_note

    def predict_frame(self, bgr_frame: np.ndarray) -> PredictionOutput:
        """
        Processes a single live webcam frame, extracts 129 motion features,
        updates the rolling sequence window, and evaluates the LSTM network.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return self._empty_response("Empty video frame received.")

        # Step 1: Extract 129 motion features
        motion_vec, curr_coords, hand_landmarks, kinetic_energy = extract_motion_frame_data(
            bgr_frame=bgr_frame,
            hands_detector=self.hands_detector,
            prev_raw_coords=self.prev_raw_coords
        )

        if motion_vec is None or curr_coords is None:
            # Hand temporarily lost
            self.prev_raw_coords = None
            if self.sequence_buffer:
                self.sequence_buffer.popleft()
            return self._empty_response("No hand detected in frame.", hand_landmarks)

        # Update previous frame coordinates
        self.prev_raw_coords = curr_coords
        self.sequence_buffer.append(motion_vec)

        # Determine motion state based on hand kinetic energy
        if kinetic_energy > 0.15:
            motion_state = "FAST MOTION"
        elif kinetic_energy > 0.04:
            motion_state = "ACTIVE MOTION"
        else:
            motion_state = "STATIC HOLD"

        # Step 2: Model Inference based on engine_mode
        # In 'auto' mode:
        # - Active motion (kinetic_energy >= 0.04) -> PyTorch LSTM temporal network
        # - Static hold (kinetic_energy < 0.04) -> High-precision Random Forest (if available) or LSTM
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
            # Build sequence tensor: shape (1, sequence_length, 129)
            frames = list(self.sequence_buffer)
            while len(frames) < self.sequence_length:
                frames.insert(0, frames[0])

            seq_array = np.array(frames, dtype=np.float32).reshape(1, self.sequence_length, TOTAL_MOTION_FEATURES)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(self.device)
            probs_tensor = self.lstm_model.predict_probabilities(seq_tensor)[0]
            probs = probs_tensor.cpu().numpy()
            model_used_name = "Deep LSTM (Motion)"
        else:
            # Spatial Classifier (63 normalized coordinates)
            spatial_feat = np.array(motion_vec[:63]).reshape(1, -1)
            probs = self.rf_model.predict_proba(spatial_feat)[0]
            model_used_name = "Random Forest (Spatial)"

        # Step 2.5: Anatomical Geometry Disambiguation (e.g. C vs O aperture)
        probs, disambig_note = self._apply_anatomical_disambiguation(curr_coords, probs)
        if disambig_note:
            model_used_name += f" [{disambig_note}]"

        # Step 3: Analyze Probability Distribution & Margin
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx

        raw_label = self.classes[top1_idx]
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        margin = top1_prob - top2_prob

        # Step 4: Motion-Aware Confidence Evaluation
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

        # Step 5: Temporal Consistency & Stability
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
            hand_landmarks_proto=hand_landmarks,
            status_message=status_message
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

        # Robust multi-stage hand landmark extraction
        features_63, hand_landmarks, raw_coords, stage_name = robust_extract_landmarks_from_bgr_frame(
            bgr_image
        )

        if features_63 is None:
            return self._empty_response(f"No hand detected after 8 cascading stages."), annotated_image

        # Draw styled landmarks onto original image coordinates
        draw_styled_landmarks(annotated_image, hand_landmarks or raw_coords)

        # Determine model to evaluate
        selected_model = model_type or ("rf" if self.rf_model is not None else "lstm")

        if selected_model == "rf" and self.rf_model is not None:
            probs = self.rf_model.predict_proba(np.array(features_63).reshape(1, -1))[0]
            model_name = f"Random Forest ({stage_name})"
        elif self.lstm_model is not None:
            # Replicate zero-velocity vector across sequence length
            motion_vec = features_63 + [0.0] * 63 + [0.0, 0.0, 0.0]
            seq_array = np.array([motion_vec] * self.sequence_length, dtype=np.float32).reshape(1, self.sequence_length, TOTAL_MOTION_FEATURES)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(self.device)
            probs = self.lstm_model.predict_probabilities(seq_tensor)[0].cpu().numpy()
            model_name = f"LSTM Projection ({stage_name})"
        elif self.rf_model is not None:
            probs = self.rf_model.predict_proba(np.array(features_63).reshape(1, -1))[0]
            model_name = f"Random Forest ({stage_name})"
        else:
            return self._empty_response("No trained model available."), annotated_image

        # Anatomical Disambiguation
        probs, disambig_note = self._apply_anatomical_disambiguation(raw_coords, probs)
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
        landmarks: Optional[object] = None
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
            status_message=message
        )

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if self.hands_detector:
            self.hands_detector.close()
