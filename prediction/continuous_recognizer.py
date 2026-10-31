"""
continuous_recognizer.py
========================
Unified Continuous Multimodal Sign Language Recognition Pipeline.

Coordinates:
1. MediaPipe Holistic (Face Mesh + Pose + Left Hand + Right Hand).
2. MultimodalFeatureExtractor (210-D face, pose, and hand kinematics).
3. ContinuousSignSegmenter (Sign boundary detection, kinetic energy inflection, NMS).
4. ContinuousSignCTCNet (Variable-length CNN-BiLSTM-CTC sequence decoder).
5. AdvancedBimanualEngine (Handedness-normalized two-handed gesture engine).
6. ASLGrammarTranslator (Linguistic Topic-Comment & Non-Manual cue English translator).
7. Personalization & Confidence Manager (Top-3 ranking, user overrides, calibration).
8. Backward-Compatible Fallback to existing SignPredictor.
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import time
import numpy as np
import cv2
import mediapipe as mp

from utils.multimodal_features import (
    MultimodalFeatureExtractor,
    MultimodalFrameData,
    NonManualCues,
    UpperBodyPoseFeatures,
    TOTAL_MULTIMODAL_DIMS
)
from utils.continuous_segmenter import (
    ContinuousSignSegmenter,
    SignSegment,
    DecodedSignEvent
)
from models.ctc_model import (
    ContinuousSignCTCNet,
    create_or_load_ctc_model
)
from utils.bimanual_engine import (
    AdvancedBimanualEngine,
    BimanualSignResult,
    EXPANDED_ASL_VOCABULARY
)
from utils.asl_grammar import (
    ASLGrammarTranslator,
    ASLGlossToken,
    GrammarTranslationResult
)
from utils.personalization import (
    PersonalVocabularyManager,
    UserCorrectionTracker,
    evaluate_confidence,
    ConfidenceReport,
    CandidatePrediction
)


@dataclass
class ContinuousFrameResult:
    """Complete result emitted per video frame in continuous mode."""
    frame_idx: int
    has_hands: bool
    has_face: bool
    has_pose: bool
    num_hands: int
    dominant_hand: str
    active_gloss: str
    display_label: str
    confidence_score: float
    confidence_level: str
    top3_candidates: List[Tuple[str, float]]
    is_uncertain: bool
    kinetic_energy: float
    motion_state: str
    signing_space: str
    non_manual_summary: str
    asl_gloss_history: List[str]
    raw_asl_string: str
    translated_english: str
    sentence_type: str
    new_sign_emitted: bool = False
    emitted_sign_text: Optional[str] = None
    multimodal_data: Optional[MultimodalFrameData] = None


class ContinuousSignRecognizer:
    """
    High-level orchestrator for real-time continuous sign language understanding.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        enable_face_pose: bool = True,
        dominant_hand_preference: str = "Auto",
        device: str = "cpu"
    ):
        self.confidence_threshold = confidence_threshold
        self.enable_face_pose = enable_face_pose
        self.device = device

        # 1. Perception Engines
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # Ultra-fast CPU mode (30+ FPS)
            smooth_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45
        )

        # 2. Multimodal Feature & Segmentation Pipeline
        self.feature_extractor = MultimodalFeatureExtractor()
        self.segmenter = ContinuousSignSegmenter(feature_dim=TOTAL_MULTIMODAL_DIMS)

        # 3. Neural CTC Sequence Model
        self.ctc_model, self.ctc_classes = create_or_load_ctc_model(device=device)

        # 4. Advanced Bimanual Engine with Handedness Normalization
        self.bimanual_engine = AdvancedBimanualEngine(dominant_hand=dominant_hand_preference)

        # 5. ASL Grammar Translator
        self.grammar_translator = ASLGrammarTranslator()

        # 6. Personalization & User Corrections
        self.vocab_manager = PersonalVocabularyManager()
        self.correction_tracker = UserCorrectionTracker()

        # 7. Session Accumulator
        self.accumulated_gloss_tokens: List[ASLGlossToken] = []
        self.last_translation: GrammarTranslationResult = GrammarTranslationResult(
            raw_glosses=[], asl_gloss_string="", english_translation="", sentence_type="STATEMENT", detected_tense="PRESENT"
        )
        self.frame_counter = 0

    def reset_session(self):
        """Clears active sentence buffers, gloss history, and segmenter tracking."""
        self.accumulated_gloss_tokens.clear()
        self.segmenter.reset()
        self.feature_extractor.reset()
        self.last_translation = GrammarTranslationResult(
            raw_glosses=[], asl_gloss_string="", english_translation="", sentence_type="STATEMENT", detected_tense="PRESENT"
        )
        self.frame_counter = 0

    def undo_last_sign(self) -> Optional[str]:
        """Removes the most recently emitted sign gloss and updates English translation."""
        if self.accumulated_gloss_tokens:
            removed = self.accumulated_gloss_tokens.pop()
            self._update_translation()
            return removed.gloss
        return None

    def add_manual_sign(self, gloss: str):
        """Manually appends or corrects a sign gloss from UI selection."""
        clean = gloss.strip().upper()
        if clean:
            token = ASLGlossToken(gloss=clean, timestamp=time.time())
            self.accumulated_gloss_tokens.append(token)
            self._update_translation()

    def process_frame(self, frame_bgr: np.ndarray) -> ContinuousFrameResult:
        """
        Executes end-to-end continuous recognition on a single video frame.
        """
        self.frame_counter += 1
        h, w, _ = frame_bgr.shape

        # Step 1: MediaPipe Holistic Inference with fast perception scaling (60+ FPS)
        infer_w = 480
        if w > infer_w:
            scale = infer_w / float(w)
            small_bgr = cv2.resize(frame_bgr, (infer_w, int(h * scale)), interpolation=cv2.INTER_NEAREST)
            rgb_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        rgb_frame.flags.writeable = False
        holistic_res = self.holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Step 2: Multimodal Feature Extraction (mapped to original h, w dimensions)
        m_data = self.feature_extractor.extract_from_holistic(holistic_res, (h, w, 3))

        # Check for immediate two-handed sign
        bimanual_res: Optional[BimanualSignResult] = None
        if m_data.num_hands == 2:
            bimanual_res = self.bimanual_engine.evaluate_bimanual_sign(
                m_data.left_hand_landmarks,
                m_data.right_hand_landmarks,
                kinetic_energy=m_data.kinetic_energy
            )

        # Step 3: Stream Segmentation
        segment = self.segmenter.process_frame(
            m_data.multimodal_vector,
            m_data.kinetic_energy
        )

        # Step 4: Sign Evaluation & Token Emission
        emitted_gloss: Optional[str] = None
        emitted_conf: float = 0.0
        top3_candidates: List[Tuple[str, float]] = []

        if bimanual_res is not None:
            # Prioritize bimanual sign
            emitted_gloss = bimanual_res.gloss
            emitted_conf = bimanual_res.confidence
            top3_candidates = [(bimanual_res.gloss, bimanual_res.confidence)]
        elif segment is not None and segment.is_valid_stroke:
            # Decode segment trajectory via CTC model
            import torch
            seq_tensor = torch.from_numpy(segment.feature_sequence).unsqueeze(0).to(self.device)
            decoded_items = self.ctc_model.decode_greedy(seq_tensor, self.ctc_classes)

            if decoded_items:
                top_gloss, top_prob = decoded_items[0]
                # Apply personal calibration boost
                boost = self.correction_tracker.get_calibration_boost(top_gloss, top_gloss)
                calibrated_conf = min(0.99, top_prob * boost)

                # Apply temporal Non-Maximum Suppression (NMS)
                nms_gloss = self.segmenter.apply_temporal_nms(
                    top_gloss,
                    confidence=calibrated_conf,
                    min_confidence=self.confidence_threshold
                )

                if nms_gloss:
                    emitted_gloss = nms_gloss
                    emitted_conf = calibrated_conf
                    top3_candidates = decoded_items[:3]

        # Step 5: Update Sentence Buffer if New Sign Emitted
        new_sign_flag = False
        if emitted_gloss is not None and emitted_gloss not in ("del", "nothing", "space", "..."):
            # Check non-manual facial cues active at moment of emission
            token = ASLGlossToken(
                gloss=emitted_gloss,
                is_question=(m_data.face_cues.eyebrow_raise_left > 0.65 or m_data.face_cues.eyebrow_raise_right > 0.65),
                is_wh_question=(m_data.face_cues.eyebrow_furrow > 0.50),
                is_negated=(m_data.face_cues.is_head_shaking or emitted_gloss in ("NOT", "NO")),
                is_emphasis=(m_data.face_cues.mouth_aperture > 0.35),
                timestamp=time.time()
            )
            self.accumulated_gloss_tokens.append(token)
            self._update_translation()
            new_sign_flag = True

        # Build non-manual textual summary
        non_manual_notes = []
        if m_data.face_cues.eyebrow_raise_left > 0.65:
            non_manual_notes.append("Raised Eyebrows [Q]")
        if m_data.face_cues.eyebrow_furrow > 0.50:
            non_manual_notes.append("Furrowed Eyebrows [WH-Q]")
        if m_data.face_cues.is_head_shaking:
            non_manual_notes.append("Head Shake [NEG]")
        if m_data.face_cues.is_head_nodding:
            non_manual_notes.append("Head Nod [AFFIRM]")
        summary_str = " | ".join(non_manual_notes) if non_manual_notes else "Neutral Expression"

        # Determine motion state
        if m_data.kinetic_energy > 0.08:
            m_state = "FAST MOTION"
        elif m_data.kinetic_energy > 0.008:
            m_state = "ACTIVE STROKE"
        else:
            m_state = "REST / HOLD"

        # Current active display label
        current_active = emitted_gloss if emitted_gloss else (
            self.accumulated_gloss_tokens[-1].gloss if self.accumulated_gloss_tokens else "Awaiting Input"
        )

        return ContinuousFrameResult(
            frame_idx=self.frame_counter,
            has_hands=m_data.has_hands,
            has_face=m_data.has_face,
            has_pose=m_data.has_pose,
            num_hands=m_data.num_hands,
            dominant_hand=m_data.dominant_hand,
            active_gloss=current_active,
            display_label=current_active,
            confidence_score=round(emitted_conf * 100.0, 1) if emitted_conf > 0 else 0.0,
            confidence_level="HIGH" if emitted_conf >= 0.70 else ("MEDIUM" if emitted_conf >= 0.40 else "LOW"),
            top3_candidates=top3_candidates,
            is_uncertain=(emitted_conf < self.confidence_threshold and emitted_conf > 0),
            kinetic_energy=round(m_data.kinetic_energy, 4),
            motion_state=m_state,
            signing_space=m_data.pose_features.signing_space_region,
            non_manual_summary=summary_str,
            asl_gloss_history=[t.gloss for t in self.accumulated_gloss_tokens],
            raw_asl_string=self.last_translation.asl_gloss_string,
            translated_english=self.last_translation.english_translation,
            sentence_type=self.last_translation.sentence_type,
            new_sign_emitted=new_sign_flag,
            emitted_sign_text=emitted_gloss,
            multimodal_data=m_data
        )

    def _update_translation(self):
        """Re-evaluates grammar translation over the current accumulated gloss token stream."""
        self.last_translation = self.grammar_translator.translate_gloss_sequence(
            self.accumulated_gloss_tokens
        )

    def draw_multimodal_hud(
        self,
        frame: np.ndarray,
        result: ContinuousFrameResult
    ) -> np.ndarray:
        """
        Renders rich visual multimodal skeleton and HUD indicators on the video frame.
        """
        h, w, _ = frame.shape
        annotated = frame.copy()

        # 1. Render Upper-Body Pose Skeleton if available
        if result.multimodal_data and result.multimodal_data.has_pose:
            # Draw shoulder and arm guide lines
            p = result.multimodal_data.pose_features
            mid = p.shoulder_midpoint
            cx, cy = int(mid[0] * w), int(mid[1] * h)
            cv2.circle(annotated, (cx, cy), 5, (14, 134, 212), -1)

        # 2. Render On-Frame Closed Caption HUD bar at bottom
        cv2.rectangle(annotated, (0, h - 85), (w, h), (15, 23, 42), -1)

        # Raw ASL Glosses
        gloss_text = f"ASL: {result.raw_asl_string or '(Signing...)'}"
        cv2.putText(annotated, gloss_text[:55], (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 2, cv2.LINE_AA)

        # English Translation
        eng_text = f"ENG: {result.translated_english or '(Waiting for signs...)'}"
        cv2.putText(annotated, eng_text[:55], (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (50, 255, 150), 2, cv2.LINE_AA)

        # Top Badge: Signing Space & Non-Manual Expression
        badge_text = f"{result.signing_space} SPACE | {result.non_manual_summary}"
        cv2.putText(annotated, badge_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 200, 100), 2, cv2.LINE_AA)

        return annotated

    def close(self):
        """Releases MediaPipe Holistic resources."""
        if hasattr(self, 'holistic') and self.holistic is not None:
            self.holistic.close()
