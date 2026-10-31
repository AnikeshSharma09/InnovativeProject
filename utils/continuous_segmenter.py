"""
continuous_segmenter.py
=======================
Real-Time Temporal Sign Boundary Detection and Continuous Stream Segmentation.

Implements:
1. Multi-Cue Sign Boundary Detection:
   - Kinetic motion energy profiling (onset, stroke apex, offset).
   - Landmark velocity peak tracking and curvature zero-crossings.
   - Hand posture inflection analysis (cosine distance between consecutive poses).
   - Inter-sign resting pause detection.
2. Temporal Window Buffering:
   - Accumulates variable-length landmark trajectories (15 to 120 frames).
   - Adaptive segment extraction when an inflection or transition is confirmed.
3. Temporal Non-Maximum Suppression (NMS) & Label Smoothing:
   - Suppresses boundary transition noise and rapid duplicate flickering.
   - Minimum duration enforcement before committing continuous sign tokens.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class SignSegment:
    """Represents a continuous segment detected between two sign boundaries."""
    start_frame: int
    end_frame: int
    duration_frames: int
    feature_sequence: np.ndarray      # Shape: (T, Feature_Dim)
    peak_kinetic_energy: float
    mean_kinetic_energy: float
    is_valid_stroke: bool
    segment_type: str = "STROKE"      # "STROKE", "HOLD", "TRANSITION", "PAUSE"


@dataclass
class DecodedSignEvent:
    """A classified sign event resulting from continuous decoding."""
    gloss: str
    confidence: float
    start_timestamp: float
    end_timestamp: float
    is_stable: bool
    top3: List[Tuple[str, float]] = field(default_factory=list)


class ContinuousSignSegmenter:
    """
    Monitors streaming landmark feature vectors, identifies sign boundaries
    (co-articulation transitions vs. meaningful strokes), and yields segmented
    gesture trajectories for continuous sequence decoding.
    """

    def __init__(
        self,
        min_sign_frames: int = 8,
        max_sign_frames: int = 90,
        energy_onset_thresh: float = 0.012,
        energy_pause_thresh: float = 0.005,
        pause_consecutive_frames: int = 6,
        posture_change_thresh: float = 0.15,
        feature_dim: int = 210
    ):
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.energy_onset_thresh = energy_onset_thresh
        self.energy_pause_thresh = energy_pause_thresh
        self.pause_consecutive_frames = pause_consecutive_frames
        self.posture_change_thresh = posture_change_thresh
        self.feature_dim = feature_dim

        # Stream State
        self.frame_counter: int = 0
        self.stream_buffer: List[np.ndarray] = []
        self.energy_history: List[float] = []
        self.prev_spatial_posture: Optional[np.ndarray] = None
        self.in_sign_stroke: bool = False
        self.current_segment_start: int = 0
        self.consecutive_pause_frames: int = 0

        # NMS & Deduplication State
        self.last_emitted_gloss: Optional[str] = None
        self.last_emission_frame: int = -100
        self.nms_suppression_window: int = 10  # Minimum frames before repeating same sign

    def reset(self):
        """Clears all streaming buffers and temporal state."""
        self.frame_counter = 0
        self.stream_buffer.clear()
        self.energy_history.clear()
        self.prev_spatial_posture = None
        self.in_sign_stroke = False
        self.current_segment_start = 0
        self.consecutive_pause_frames = 0
        self.last_emitted_gloss = None
        self.last_emission_frame = -100

    def process_frame(
        self,
        feature_vector: np.ndarray,
        kinetic_energy: float
    ) -> Optional[SignSegment]:
        """
        Processes a single video frame's feature vector.
        Returns a SignSegment if a natural sign boundary is detected, else None.
        """
        self.frame_counter += 1
        self.stream_buffer.append(feature_vector)
        self.energy_history.append(kinetic_energy)

        # Extract spatial posture (first 63 dimensions = normalized 21 landmarks)
        current_posture = feature_vector[:63] if len(feature_vector) >= 63 else feature_vector

        # Calculate posture change rate (Euclidean drift from previous frame)
        posture_delta = 0.0
        if self.prev_spatial_posture is not None:
            posture_delta = float(np.linalg.norm(current_posture - self.prev_spatial_posture))
        self.prev_spatial_posture = current_posture.copy()

        # Track pauses (resting hand or stable hold)
        if kinetic_energy < self.energy_pause_thresh and posture_delta < 0.05:
            self.consecutive_pause_frames += 1
        else:
            self.consecutive_pause_frames = 0

        # ---------------------------------------------------------------------
        # 1. SIGN ONSET DETECTION
        # ---------------------------------------------------------------------
        if not self.in_sign_stroke:
            if kinetic_energy >= self.energy_onset_thresh or posture_delta >= self.posture_change_thresh:
                # Sign movement has initiated
                self.in_sign_stroke = True
                self.current_segment_start = max(0, len(self.stream_buffer) - 4)

        # ---------------------------------------------------------------------
        # 2. SIGN BOUNDARY / COMPLETION DETECTION
        # ---------------------------------------------------------------------
        segment_to_emit: Optional[SignSegment] = None

        if self.in_sign_stroke:
            current_duration = len(self.stream_buffer) - self.current_segment_start

            # Boundary Condition A: Signer paused / returned to neutral resting space
            is_pause_boundary = (self.consecutive_pause_frames >= self.pause_consecutive_frames and 
                                 current_duration >= self.min_sign_frames)

            # Boundary Condition B: Inflection point where kinetic energy drops after a peak
            is_inflection_boundary = False
            if current_duration >= self.min_sign_frames and len(self.energy_history) >= 7:
                recent_energies = self.energy_history[-7:]
                peak_idx = int(np.argmax(recent_energies))
                # Peak was in the middle of recent window and current energy dropped significantly
                if 1 <= peak_idx <= 4 and recent_energies[-1] < 0.40 * recent_energies[peak_idx]:
                    is_inflection_boundary = True

            # Boundary Condition C: Buffer exceeded maximum sign length
            is_max_length_boundary = current_duration >= self.max_sign_frames

            if is_pause_boundary or is_inflection_boundary or is_max_length_boundary:
                # Extract segment
                seg_features = np.array(self.stream_buffer[self.current_segment_start:], dtype=np.float32)
                seg_energies = self.energy_history[self.current_segment_start:]
                peak_e = float(np.max(seg_energies)) if seg_energies else 0.0
                mean_e = float(np.mean(seg_energies)) if seg_energies else 0.0

                segment_to_emit = SignSegment(
                    start_frame=self.current_segment_start,
                    end_frame=len(self.stream_buffer),
                    duration_frames=current_duration,
                    feature_sequence=seg_features,
                    peak_kinetic_energy=peak_e,
                    mean_kinetic_energy=mean_e,
                    is_valid_stroke=(current_duration >= self.min_sign_frames),
                    segment_type="STROKE" if not is_pause_boundary else "HOLD"
                )

                # Reset stroke tracker and prune buffer to maintain lean memory
                self.in_sign_stroke = False
                self.current_segment_start = len(self.stream_buffer)
                if len(self.stream_buffer) > 200:
                    keep_len = 30
                    self.stream_buffer = self.stream_buffer[-keep_len:]
                    self.energy_history = self.energy_history[-keep_len:]
                    self.current_segment_start = len(self.stream_buffer)

        return segment_to_emit

    def apply_temporal_nms(
        self,
        predicted_gloss: str,
        confidence: float,
        min_confidence: float = 0.35
    ) -> Optional[str]:
        """
        Applies Non-Maximum Suppression (NMS) across consecutive sign emissions
        to eliminate double-triggers and spurious co-articulation noise.
        """
        if confidence < min_confidence or not predicted_gloss or predicted_gloss == "...":
            return None

        # Check suppression window
        frames_since_last = self.frame_counter - self.last_emission_frame
        if predicted_gloss == self.last_emitted_gloss and frames_since_last < self.nms_suppression_window:
            # Suppress duplicate repetition within short temporal window
            return None

        self.last_emitted_gloss = predicted_gloss
        self.last_emission_frame = self.frame_counter
        return predicted_gloss
