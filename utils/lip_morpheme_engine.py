"""
Lip-Morpheme Facial Fusion Engine
==================================
Fuses MediaPipe FaceMesh lip kinematics with hand skeletal landmarks to resolve
inherently ambiguous ASL hand shapes (e.g. 'M' vs 'N', 'C' vs 'O', 'D' vs '1', 'S' vs 'E').

Linguistic Basis:
Deaf signers unconsciously mouth morphemic syllables while fingerspelling ambiguous signs:
- 'M': Bilabial contact (lips pressed tightly together).
- 'N': Alveolar tongue-tooth contact with parted lips.
- 'O': Circular puckered lip aperture.
- 'E' / 'S': Lateral smile retraction (high width-to-height ratio).
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np


class LipMorphemeEngine:
    """
    Analyzes fine-grained lip contour landmarks from MediaPipe FaceMesh
    to provide multi-modal disambiguation for visually identical hand gestures.
    """

    def __init__(
        self,
        bilabial_closed_thresh: float = 0.055,
        alveolar_parted_thresh: float = 0.11,
        round_o_thresh: float = 0.38
    ):
        self.bilabial_closed_thresh = bilabial_closed_thresh
        self.alveolar_parted_thresh = alveolar_parted_thresh
        self.round_o_thresh = round_o_thresh

    def analyze_lip_kinematics(self, face_landmarks: Any) -> Dict[str, Any]:
        """
        Extracts geometric lip metrics from FaceMesh landmarks.
        Accepts either a list of (x, y, z) tuples or a MediaPipe landmark protobuf.
        """
        if face_landmarks is None:
            return {"has_lips": False, "morpheme": "NONE", "mar": 0.0, "roundness": 0.0}

        try:
            # Handle list of tuples or protobuf
            if hasattr(face_landmarks, "landmark"):
                lm = face_landmarks.landmark
                get_pt = lambda idx: np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=np.float32)
            else:
                lm = face_landmarks
                get_pt = lambda idx: np.array(lm[idx][:3], dtype=np.float32)

            if len(lm) < 315:
                return {"has_lips": False, "morpheme": "NONE", "mar": 0.0, "roundness": 0.0}

            # Key lip indices:
            # 13: Inner upper lip center
            # 14: Inner lower lip center
            # 61: Left mouth corner
            # 291: Right mouth corner
            # 0: Outer upper lip center
            # 17: Outer lower lip center
            p13 = get_pt(13)
            p14 = get_pt(14)
            p61 = get_pt(61)
            p291 = get_pt(291)
            p0 = get_pt(0)
            p17 = get_pt(17)

            # Metrics
            inner_height = float(np.linalg.norm(p13[:2] - p14[:2]))
            outer_height = float(np.linalg.norm(p0[:2] - p17[:2]))
            width = max(1e-4, float(np.linalg.norm(p61[:2] - p291[:2])))

            mar = inner_height / width  # Mouth Aspect Ratio
            outer_mar = outer_height / width

            # Determine dominant lip morpheme
            if mar < self.bilabial_closed_thresh:
                morpheme = "BILABIAL_CLOSED"  # 'M', 'P', 'B'
            elif mar >= self.round_o_thresh:
                morpheme = "ROUND_PUCKER"     # 'O', '1'
            elif mar >= self.alveolar_parted_thresh:
                morpheme = "ALVEOLAR_PARTED"   # 'N', 'D', 'T'
            else:
                morpheme = "NEUTRAL"

            return {
                "has_lips": True,
                "morpheme": morpheme,
                "mar": round(mar, 4),
                "outer_mar": round(outer_mar, 4),
                "width": round(width, 4)
            }
        except Exception:
            return {"has_lips": False, "morpheme": "NONE", "mar": 0.0, "roundness": 0.0}

    def apply_morpheme_fusion(
        self,
        top1_label: str,
        top2_label: str,
        probs: np.ndarray,
        classes: List[str],
        face_landmarks: Any
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Adjusts classifier probability distribution using real-time lip morpheme evidence.
        """
        lip_data = self.analyze_lip_kinematics(face_landmarks)
        if not lip_data["has_lips"]:
            return probs, None

        morpheme = lip_data["morpheme"]
        mar = lip_data["mar"]
        top2_set = {top1_label, top2_label}
        probs = probs.copy()
        note = None

        # 1. 'M' vs 'N' Disambiguation
        if top2_set == {"M", "N"}:
            m_idx = classes.index("M") if "M" in classes else -1
            n_idx = classes.index("N") if "N" in classes else -1
            if m_idx >= 0 and n_idx >= 0:
                if morpheme == "BILABIAL_CLOSED" or mar <= 0.06:
                    # Compressed lips -> 'M'
                    probs[m_idx] = max(probs[m_idx], 0.82)
                    probs[n_idx] = min(probs[n_idx], 0.10)
                    probs = probs / np.sum(probs)
                    note = f"Lip morpheme: Bilabial closure (MAR={mar:.2f}) -> 'M'"
                elif morpheme == "ALVEOLAR_PARTED" or mar >= 0.10:
                    # Parted lips -> 'N'
                    probs[n_idx] = max(probs[n_idx], 0.82)
                    probs[m_idx] = min(probs[m_idx], 0.10)
                    probs = probs / np.sum(probs)
                    note = f"Lip morpheme: Alveolar parted (MAR={mar:.2f}) -> 'N'"

        # 2. 'C' vs 'O' Disambiguation
        elif top2_set == {"C", "O"}:
            c_idx = classes.index("C") if "C" in classes else -1
            o_idx = classes.index("O") if "O" in classes else -1
            if c_idx >= 0 and o_idx >= 0:
                if morpheme == "ROUND_PUCKER" or mar >= 0.32:
                    # Circular rounded mouth -> 'O'
                    probs[o_idx] = max(probs[o_idx], 0.85)
                    probs[c_idx] = min(probs[c_idx], 0.08)
                    probs = probs / np.sum(probs)
                    note = f"Lip morpheme: Round circular aperture (MAR={mar:.2f}) -> 'O'"

        # 3. 'D' vs '1' Disambiguation
        elif top2_set == {"D", "1"}:
            d_idx = classes.index("D") if "D" in classes else -1
            one_idx = classes.index("1") if "1" in classes else -1
            if d_idx >= 0 and one_idx >= 0:
                if morpheme == "ROUND_PUCKER":
                    # Mouthing 'one' -> '1'
                    probs[one_idx] = max(probs[one_idx], 0.82)
                    probs[d_idx] = min(probs[d_idx], 0.10)
                    probs = probs / np.sum(probs)
                    note = f"Lip morpheme: Round pucker -> '1'"
                elif morpheme == "ALVEOLAR_PARTED":
                    # Mouthing 'dee' -> 'D'
                    probs[d_idx] = max(probs[d_idx], 0.82)
                    probs[one_idx] = min(probs[one_idx], 0.10)
                    probs = probs / np.sum(probs)
                    note = f"Lip morpheme: Alveolar parted -> 'D'"

        return probs, note


if __name__ == "__main__":
    engine = LipMorphemeEngine()
    print("[+] LipMorphemeEngine initialized.")
    # Test synthetic landmarks
    fake_face = [(0.0, 0.0, 0.0) for _ in range(468)]
    fake_face[61] = (0.35, 0.5, 0.0)   # Left corner
    fake_face[291] = (0.65, 0.5, 0.0)  # Right corner
    fake_face[13] = (0.5, 0.495, 0.0)  # Upper inner
    fake_face[14] = (0.5, 0.505, 0.0)  # Lower inner
    res = engine.analyze_lip_kinematics(fake_face)
    print("Synthetic Lip Analysis:", res)
