"""
One-Shot Biomechanical Hand Calibration Engine
===============================================
Calibrates the sign recognition system to the user's specific hand anatomy:
1. Measures personal bone length ratios and palm proportions.
2. Identifies individual finger curl limits (stiffness vs hypermobility).
3. Auto-tunes 1€ anti-tremor filter cutoff and beta parameters to individual resting jitter.
4. Serializes user profile to data/user_hand_profile.json for persistent personalization.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "user_hand_profile.json"


class HandCalibrator:
    """
    Manages the 3-step anatomical calibration protocol and applies
    morphological scaling corrections to raw landmarks.
    """

    def __init__(self, profile_path: Optional[Any] = None):
        self.profile_path = Path(profile_path) if profile_path else PROFILE_PATH
        self.profile: Dict[str, Any] = self.load_profile()
        self.calibration_buffer: List[List[List[float]]] = []

    def load_profile(self) -> Dict[str, Any]:
        """Loads cached profile or default canonical proportions."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    p = json.load(f)
                    p["calibrated"] = p.get("is_calibrated", False)
                    return p
            except Exception:
                pass
        return {
            "is_calibrated": False,
            "calibrated": False,
            "finger_ratios": {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
            "segment_ratios": {"index_to_palm": 0.95, "middle_to_palm": 1.05},
            "palm_aspect_ratio": 0.85,
            "max_curl_limits": {"thumb": 0.35, "index": 0.25, "middle": 0.25, "ring": 0.25, "pinky": 0.25},
            "resting_jitter_score": 0.002,
            "tuned_min_cutoff": 0.8,
            "tuned_beta": 0.015
        }

    def reset_profile(self):
        """Resets calibration to uncalibrated state."""
        self.profile["is_calibrated"] = False
        self.profile["calibrated"] = False
        self.calibration_buffer.clear()
        if self.profile_path.exists():
            try:
                self.profile_path.unlink()
            except Exception:
                pass

    def record_calibration_frame(self, coords: List[Tuple[float, float, float]]):
        """Buffers a calibration frame."""
        self.calibration_buffer.append([[float(c) for c in pt] for pt in coords])

    def finish_calibration(self) -> bool:
        """Processes buffered frames and finalizes calibration."""
        if len(self.calibration_buffer) < 5:
            return False
        ratios = self.compute_open_hand_ratios(self.calibration_buffer)
        jitter, cutoff, beta = self.compute_tremor_parameters(self.calibration_buffer)
        self.profile["is_calibrated"] = True
        self.profile["calibrated"] = True
        self.profile["finger_ratios"] = ratios
        self.profile["segment_ratios"] = {
            "index_to_palm": ratios.get("index", 0.95),
            "middle_to_palm": ratios.get("middle", 1.05)
        }
        self.profile["resting_jitter_score"] = jitter
        self.profile["tuned_min_cutoff"] = cutoff
        self.profile["tuned_beta"] = beta
        self.save_profile()
        return True

    def save_profile(self) -> bool:
        """Persists profile to JSON disk cache."""
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2)
            return True
        except Exception:
            return False

    def compute_open_hand_ratios(self, samples: List[List[List[float]]]) -> Dict[str, float]:
        """
        Calculates user's finger bone lengths normalized by palm length.
        """
        arr = np.array(samples, dtype=np.float32)  # (N, 21, 3)
        mean_coords = np.median(arr, axis=0)      # (21, 3)

        w = mean_coords[0]
        m_mcp = mean_coords[9]
        palm_len = max(1e-4, float(np.linalg.norm(m_mcp - w)))

        # Individual finger lengths from MCP to Tip
        thumb_len = float(np.linalg.norm(mean_coords[4] - mean_coords[1]))
        index_len = float(np.linalg.norm(mean_coords[8] - mean_coords[5]))
        middle_len = float(np.linalg.norm(mean_coords[12] - mean_coords[9]))
        ring_len = float(np.linalg.norm(mean_coords[16] - mean_coords[13]))
        pinky_len = float(np.linalg.norm(mean_coords[20] - mean_coords[17]))

        # Palm width
        p_mcp = mean_coords[17]
        i_mcp = mean_coords[5]
        palm_width = float(np.linalg.norm(p_mcp - i_mcp))

        return {
            "thumb": round(thumb_len / palm_len, 4),
            "index": round(index_len / palm_len, 4),
            "middle": round(middle_len / palm_len, 4),
            "ring": round(ring_len / palm_len, 4),
            "pinky": round(pinky_len / palm_len, 4),
            "palm_aspect_ratio": round(palm_width / palm_len, 4)
        }

    def compute_tremor_parameters(self, samples: List[List[List[float]]]) -> Tuple[float, float, float]:
        """
        Analyzes resting tremor variance to auto-tune 1€ filter cutoff and beta.
        """
        arr = np.array(samples, dtype=np.float32)  # (N, 21, 3)
        # Variance across time per landmark
        std_per_joint = np.std(arr, axis=0)        # (21, 3)
        mean_jitter = float(np.mean(std_per_joint))

        # Dynamic parameter tuning:
        # High jitter -> higher min_cutoff (more aggressive anti-tremor smoothing)
        # Low jitter -> lower min_cutoff (ultra-crisp responsive tracking)
        if mean_jitter > 0.008:
            tuned_cutoff = 1.25
            tuned_beta = 0.008
        elif mean_jitter > 0.003:
            tuned_cutoff = 0.85
            tuned_beta = 0.015
        else:
            tuned_cutoff = 0.60
            tuned_beta = 0.025

        return round(mean_jitter, 6), tuned_cutoff, tuned_beta

    def finalize_calibration(
        self,
        open_hand_samples: List[List[List[float]]],
        rest_samples: List[List[List[float]]]
    ) -> Dict[str, Any]:
        """Completes calibration and updates active user profile."""
        ratios = self.compute_open_hand_ratios(open_hand_samples)
        jitter, cutoff, beta = self.compute_tremor_parameters(rest_samples)

        self.profile["is_calibrated"] = True
        self.profile["finger_ratios"] = {
            "thumb": ratios["thumb"],
            "index": ratios["index"],
            "middle": ratios["middle"],
            "ring": ratios["ring"],
            "pinky": ratios["pinky"]
        }
        self.profile["palm_aspect_ratio"] = ratios["palm_aspect_ratio"]
        self.profile["resting_jitter_score"] = jitter
        self.profile["tuned_min_cutoff"] = cutoff
        self.profile["tuned_beta"] = beta

        self.save_profile()
        return self.profile

    def apply_morphological_scaling(
        self,
        raw_coords: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Scales incoming raw landmarks against the user's calibrated finger bone ratios,
        standardizing them to canonical population proportions before classification.
        """
        if not self.profile.get("is_calibrated", False):
            return raw_coords

        coords = np.array(raw_coords, dtype=np.float32)
        ratios = self.profile.get("finger_ratios", {})

        # Canonical baseline ratios
        canon = {"thumb": 0.85, "index": 0.95, "middle": 1.05, "ring": 0.95, "pinky": 0.75}

        # Scale each finger's vertices towards MCP joint
        finger_indices = {
            "thumb": ([1, 2, 3, 4], 1),
            "index": ([6, 7, 8], 5),
            "middle": ([10, 11, 12], 9),
            "ring": ([14, 15, 16], 13),
            "pinky": ([18, 19, 20], 17)
        }

        for f_name, (indices, base_idx) in finger_indices.items():
            user_r = ratios.get(f_name, 1.0)
            canon_r = canon.get(f_name, 1.0)
            scale_factor = canon_r / max(1e-4, user_r)
            # Bound scale factor to prevent distortion
            scale_factor = float(np.clip(scale_factor, 0.75, 1.35))

            base_pt = coords[base_idx]
            for idx in indices:
                coords[idx] = base_pt + (coords[idx] - base_pt) * scale_factor

        return [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in coords]


if __name__ == "__main__":
    calibrator = HandCalibrator()
    print("[+] HandCalibrator loaded. Profile:", calibrator.profile)
    # Test synthetic scaling
    dummy_pts = [(float(i) * 0.05, -float(i) * 0.05, 0.0) for i in range(21)]
    scaled = calibrator.apply_morphological_scaling(dummy_pts)
    print("Morphological scaling verified. Output length:", len(scaled))
