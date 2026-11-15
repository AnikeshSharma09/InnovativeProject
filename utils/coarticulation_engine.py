"""
Co-Articulation Kinematic Velocity Decoder
===========================================
World-first continuous fingerspelling decoder for sign language AI.
Enables rapid-fire fingerspelling (4-6 signs/sec) without requiring the signer
to unnaturally freeze or hold hands still for each letter.

Algorithm:
1. Tracks landmark-wise velocity vectors and kinetic energy curves in real time.
2. Identifies local deceleration valleys (inflection points / apex of co-articulated handshapes).
3. Evaluates classifier at the exact mechanical apex of tendon extension.
4. Suppresses inter-sign transitional noise and duplicate emissions via adaptive refractory window.
"""

import time
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class ApexEvent:
    is_apex: bool = False
    kinetic_energy: float = 0.0
    emitted_sign: Optional[str] = None


class CoArticulationDecoder:
    """
    Decodes rapid, continuous fingerspelling gestures by detecting
    kinematic deceleration apexes along the continuous motion trajectory.
    """

    def __init__(
        self,
        energy_window_size: int = 5,
        apex_velocity_threshold: float = 0.045,
        min_refractory_sec: float = 0.16,  # ~160ms between letters (allows up to 6 signs/sec)
        confidence_threshold: float = 0.55
    ):
        self.energy_window_size = energy_window_size
        self.apex_velocity_threshold = apex_velocity_threshold
        self.min_refractory_sec = min_refractory_sec
        self.confidence_threshold = confidence_threshold

        # Historical buffers
        self.prev_coords: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None
        self.energy_history: deque = deque(maxlen=energy_window_size)
        self.timestamp_history: deque = deque(maxlen=energy_window_size)

        # Emission tracking
        self.last_emission_time: float = 0.0
        self.last_emitted_sign: Optional[str] = None
        self.total_apexes_detected: int = 0

    def reset(self):
        """Clears kinematic tracking history."""
        self.prev_coords = None
        self.prev_time = None
        self.energy_history.clear()
        self.timestamp_history.clear()
        self.last_emission_time = 0.0
        self.last_emitted_sign = None

    def decode_step(
        self,
        raw_coords: Any,
        predicted_label: Optional[str] = None,
        confidence: float = 0.0,
        timestamp: Optional[float] = None
    ) -> ApexEvent:
        """
        Processes a single frame for rapid co-articulation fingerspelling.
        Returns an ApexEvent with apex status and emitted sign if validated.
        """
        ke, is_apex = self.update_kinematics(raw_coords, current_time=timestamp)
        emitted = None
        if is_apex and predicted_label:
            emitted = self.evaluate_apex_candidate(predicted_label, confidence, current_time=timestamp)
        return ApexEvent(is_apex=is_apex, kinetic_energy=ke, emitted_sign=emitted)

    def update_kinematics(
        self,
        raw_coords: Any = None,
        current_time: Optional[float] = None,
        coords: Any = None,
        predicted_label: Optional[str] = None,
        confidence: float = 0.0,
        timestamp: Optional[float] = None
    ) -> Union[Tuple[float, bool], ApexEvent]:
        """
        Calculates instantaneous joint kinetic energy and checks if current
        frame corresponds to a kinematic deceleration apex (sign inflection point).

        Returns:
            Tuple of (smoothed_kinetic_energy, is_apex_detected) or ApexEvent if called with predicted_label
        """
        active_coords = coords if coords is not None else raw_coords
        now = timestamp or current_time or time.time()
        
        if active_coords is None:
            if predicted_label is not None:
                return ApexEvent(is_apex=False, kinetic_energy=0.0, emitted_sign=None)
            return 0.0, False

        coords_arr = np.array(active_coords, dtype=np.float32)

        if self.prev_coords is None or self.prev_time is None:
            self.prev_coords = coords_arr
            self.prev_time = now
            if predicted_label is not None:
                return ApexEvent(is_apex=False, kinetic_energy=0.0, emitted_sign=None)
            return 0.0, False

        dt = max(1e-4, now - self.prev_time)
        self.prev_time = now

        # Compute velocity per landmark
        vel = (coords_arr - self.prev_coords) / dt
        self.prev_coords = coords_arr

        # Instantaneous mean kinetic energy across all 21 hand joints
        # Emphasize fingertips (indices 4, 8, 12, 16, 20) as they articulate most during sign transitions
        weights = np.ones(21, dtype=np.float32)
        weights[[4, 8, 12, 16, 20]] = 2.5
        weights = weights / np.sum(weights)

        joint_speeds = np.sum(vel ** 2, axis=1)  # shape (21,)
        instant_ke = float(np.sum(joint_speeds * weights))

        self.energy_history.append(instant_ke)
        self.timestamp_history.append(now)

        if len(self.energy_history) < self.energy_window_size:
            if predicted_label is not None:
                return ApexEvent(is_apex=False, kinetic_energy=instant_ke, emitted_sign=None)
            return instant_ke, False

        # Moving average smoothed kinetic energy
        smoothed_ke = float(np.mean(self.energy_history))

        # Check for apex inflection point:
        # A valley in energy occurs when middle element is lower than neighbors
        # and below the co-articulation apex threshold
        mid_idx = len(self.energy_history) // 2
        mid_val = self.energy_history[mid_idx]

        is_valley = True
        for i in range(len(self.energy_history)):
            if i != mid_idx and self.energy_history[i] < mid_val:
                is_valley = False
                break

        is_apex = False
        if is_valley and mid_val < self.apex_velocity_threshold:
            time_since_last = now - self.last_emission_time
            if time_since_last >= self.min_refractory_sec:
                is_apex = True
                self.total_apexes_detected += 1

        if predicted_label is not None:
            emitted = None
            if is_apex:
                emitted = self.evaluate_apex_candidate(predicted_label, confidence, current_time=now)
            return ApexEvent(is_apex=is_apex, kinetic_energy=smoothed_ke, emitted_sign=emitted)

        return smoothed_ke, is_apex

    def evaluate_apex_candidate(
        self,
        predicted_label: str,
        confidence: float,
        current_time: Optional[float] = None
    ) -> Optional[str]:
        """
        Validates whether a detected inflection apex should emit a new fingerspelled sign.
        Handles debouncing and rapid sequential letter repetition.
        """
        now = current_time or time.time()
        if confidence < self.confidence_threshold:
            return None

        clean_label = predicted_label.strip().upper()
        if clean_label in ("NOTHING", "UNCERTAIN GESTURE", "UNCERTAIN"):
            return None

        # Allow same letter repeated only if sufficient refractory time elapsed
        # (e.g. double 'L' in 'HELLO')
        if clean_label == self.last_emitted_sign:
            if (now - self.last_emission_time) < (self.min_refractory_sec * 1.5):
                return None

        self.last_emission_time = now
        self.last_emitted_sign = clean_label
        return clean_label


if __name__ == "__main__":
    decoder = CoArticulationDecoder()
    print("[+] CoArticulationDecoder initialized successfully.")
    # Test synthetic trajectory
    coords_dummy = [[float(i), float(i), 0.0] for i in range(21)]
    for t_step in range(10):
        # Simulate acceleration then deceleration
        jitter = 0.001 if (t_step % 4 == 0) else 0.05
        c_mod = [[p[0] + jitter * t_step, p[1], p[2]] for p in coords_dummy]
        ke, is_apex = decoder.update_kinematics(c_mod, current_time=t_step * 0.05)
        print(f"Step {t_step}: KE={ke:.5f}, Apex={is_apex}")
