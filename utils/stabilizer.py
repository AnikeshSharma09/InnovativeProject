"""
Biomechanical Anti-Tremor & Shiver Stabilizer for Real-Time Hand Landmark Tracking.
Implements the 1€ (One-Euro) Filter (Casiez, Roussel, Vogel - CHI 2012)
and adaptive exponential moving average for probability smoothing.

Specifically engineered to solve:
1. Involuntary hand shivering / tremor (medical, fatigue, nervousness).
2. Webcam sensor pixel noise and flickering.
3. Micro-jitter that falsely triggers dynamic motion gating.
"""
import math
import time
from typing import List, Tuple, Optional
import numpy as np


class LowPassFilter:
    """First-order low-pass exponential filter."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.s: Optional[float] = None

    def reset(self):
        self.s = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter:
    """
    1€ (One-Euro) Filter for adaptive, low-latency jitter suppression.
    
    Dynamically adjusts cutoff frequency:
    - At low speeds (shivering, still hold): cutoff drops to min_cutoff -> heavy smoothing.
    - At high speeds (intentional gestures): cutoff scales with speed -> zero lag.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 0.8,
        beta: float = 0.03,
        d_cutoff: float = 1.0
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None

    def reset(self):
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None

    @staticmethod
    def _compute_alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.time()

        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filt.filter(x, alpha=1.0)

        dt = timestamp - self.last_time
        if dt <= 1e-5:
            dt = 1.0 / max(1.0, self.freq)
        self.last_time = timestamp
        rate = 1.0 / dt

        # Estimate derivative of the signal (speed)
        prev_x = self.x_filt.s if self.x_filt.s is not None else x
        dx = (x - prev_x) * rate
        alpha_d = self._compute_alpha(rate, self.d_cutoff)
        edx = self.dx_filt.filter(dx, alpha=alpha_d)

        # Dynamic cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._compute_alpha(rate, cutoff)
        return self.x_filt.filter(x, alpha=alpha)


class LandmarkStabilizer:
    """
    Stabilizes all 21 3D hand landmarks (63 values) in real-time.
    Eliminates camera pixel noise and human hand trembling.
    """

    def __init__(
        self,
        min_cutoff: float = 0.7,
        beta: float = 0.04,
        shiver_threshold: float = 0.003
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.shiver_threshold = shiver_threshold

        # 21 landmarks * 3 dimensions (x, y, z) = 63 filters
        self.filters: List[List[OneEuroFilter]] = [
            [
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta),
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta),
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            ]
            for _ in range(21)
        ]
        self.jitter_history: List[float] = []
        self.last_jitter_score: float = 0.0

    def reset(self):
        for lm_filters in self.filters:
            for f in lm_filters:
                f.reset()
        self.jitter_history.clear()
        self.last_jitter_score = 0.0

    def update_parameters(self, min_cutoff: float, beta: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        for lm_filters in self.filters:
            for f in lm_filters:
                f.min_cutoff = min_cutoff
                f.beta = beta

    def stabilize(
        self,
        raw_coords: List[Tuple[float, float, float]],
        timestamp: Optional[float] = None
    ) -> Tuple[List[Tuple[float, float, float]], float, bool]:
        """
        Filters the 21 3D landmarks.
        
        Returns:
            Tuple of:
            - stabilized_coords: List of 21 smoothed (x, y, z) tuples
            - jitter_score: Variance/magnitude of suppressed micro-jitter
            - is_shivering: True if significant high-frequency tremor was actively dampened
        """
        if raw_coords is None or len(raw_coords) < 21:
            return raw_coords, 0.0, False

        stabilized: List[Tuple[float, float, float]] = []
        diffs = []

        for i in range(21):
            rx, ry, rz = raw_coords[i]
            sx = self.filters[i][0].filter(rx, timestamp)
            sy = self.filters[i][1].filter(ry, timestamp)
            sz = self.filters[i][2].filter(rz, timestamp)
            stabilized.append((sx, sy, sz))

            # Quantify micro-jitter (difference between raw jittery coordinate and smooth point)
            diffs.append((rx - sx) ** 2 + (ry - sy) ** 2 + (rz - sz) ** 2)

        # Compute current frame jitter score
        current_jitter = float(np.mean(diffs))
        self.last_jitter_score = 0.85 * self.last_jitter_score + 0.15 * current_jitter
        is_shivering = self.last_jitter_score > (self.shiver_threshold ** 2)

        return stabilized, self.last_jitter_score, is_shivering


class ProbabilityStabilizer:
    """
    Smooths class probability distributions over consecutive frames
    to prevent single-frame flickering or classification flapping.
    """

    def __init__(self, alpha: float = 0.65):
        """
        alpha: Weight given to new frame (0.65 = 65% new, 35% history).
        """
        self.alpha = alpha
        self.smoothed_probs: Optional[np.ndarray] = None

    def reset(self):
        self.smoothed_probs = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        if self.smoothed_probs is None or len(self.smoothed_probs) != len(probs):
            self.smoothed_probs = np.array(probs, dtype=np.float32)
        else:
            self.smoothed_probs = (
                self.alpha * probs + (1.0 - self.alpha) * self.smoothed_probs
            )
            # Re-normalize to maintain valid probability distribution
            total = np.sum(self.smoothed_probs)
            if total > 1e-6:
                self.smoothed_probs /= total
        return self.smoothed_probs


class DualLandmarkStabilizer:
    """
    Coordinates simultaneous 1€ anti-tremor stabilization for both Left and Right hands.
    Ensures independent tracking history for each hand so presence or absence of one
    hand does not disrupt the other.
    """

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.015, shiver_threshold: float = 0.003):
        self.left_stabilizer = LandmarkStabilizer(min_cutoff=min_cutoff, beta=beta, shiver_threshold=shiver_threshold)
        self.right_stabilizer = LandmarkStabilizer(min_cutoff=min_cutoff, beta=beta, shiver_threshold=shiver_threshold)

    def reset(self):
        self.left_stabilizer.reset()
        self.right_stabilizer.reset()

    def update_parameters(self, min_cutoff: float, beta: float):
        self.left_stabilizer.update_parameters(min_cutoff, beta)
        self.right_stabilizer.update_parameters(min_cutoff, beta)

    def stabilize(
        self,
        left_coords: Optional[List[Tuple[float, float, float]]],
        right_coords: Optional[List[Tuple[float, float, float]]],
        timestamp: Optional[float] = None
    ) -> Tuple[Optional[List[Tuple[float, float, float]]], Optional[List[Tuple[float, float, float]]], float, bool]:
        """
        Stabilizes both Left and Right hands independently.
        Returns (stab_left, stab_right, max_jitter, is_shivering).
        """
        stab_left, l_jitter, l_shiv = (None, 0.0, False)
        stab_right, r_jitter, r_shiv = (None, 0.0, False)

        if left_coords:
            stab_left, l_jitter, l_shiv = self.left_stabilizer.stabilize(left_coords, timestamp)
        else:
            self.left_stabilizer.reset()

        if right_coords:
            stab_right, r_jitter, r_shiv = self.right_stabilizer.stabilize(right_coords, timestamp)
        else:
            self.right_stabilizer.reset()

        combined_jitter = max(l_jitter, r_jitter)
        is_shivering = (l_shiv or r_shiv)
        return stab_left, stab_right, combined_jitter, is_shivering

