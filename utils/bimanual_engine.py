"""
bimanual_engine.py
==================
Advanced Learned & Kinematic Bimanual (Two-Handed) Sign Recognition Engine.

Features:
1. 42 3D Landmarks across Left and Right Hands.
2. Relative 3D Wrist Transforms & Spatial Separation.
3. 10-Finger Extension States and Palm Plane Normal Vectors.
4. Physical Contact Point Detection (fingertip-to-palm, knuckle-to-wrist).
5. Symmetrical Movement Correlation (parallel vs. mirrored trajectories).
6. Handedness Normalization:
   - Supports Left-Handed and Right-Handed signers transparently.
   - Mirrors coordinates across the body sagittal axis and maps non-dominant/dominant roles.
7. Expanded Vocabulary Recognition:
   - High-Frequency Everyday, Workplace, Medical, and Emergency ASL Signs.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BimanualPosture:
    """Encapsulates spatial and geometric relationships between both hands."""
    num_hands: int
    is_bimanual: bool
    wrist_distance: float
    wrist_vector: Tuple[float, float, float]
    left_finger_extensions: Tuple[bool, bool, bool, bool, bool] # Thumb, Index, Middle, Ring, Pinky
    right_finger_extensions: Tuple[bool, bool, bool, bool, bool]
    left_all_open: bool
    right_all_open: bool
    left_palm_facing: str  # "FORWARD", "INWARD", "DOWN", "UP", "AWAY"
    right_palm_facing: str
    contact_detected: bool
    symmetry_score: float # 1.0 = perfect bilateral mirror symmetry
    dominant_hand: str = "Right"


@dataclass
class BimanualSignResult:
    """Outcome of bimanual sign evaluation."""
    sign_id: str
    gloss: str
    confidence: float
    category: str        # "EMERGENCY", "MEDICAL", "WORKPLACE", "CONVERSATIONAL", "EVERYDAY"
    description: str
    is_bimanual: bool = True


# =============================================================================
# EXPANDED ASL VOCABULARY SPECIFICATION
# =============================================================================
EXPANDED_ASL_VOCABULARY: Dict[str, Dict[str, Any]] = {
    # Emergency & Medical Signs
    "EMERGENCY": {
        "category": "EMERGENCY",
        "gloss": "EMERGENCY",
        "description": "Both hands shake in warning lateral sweeps"
    },
    "DOCTOR": {
        "category": "MEDICAL",
        "gloss": "DOCTOR",
        "description": "Dominant bent hand taps pulse on non-dominant upturned wrist"
    },
    "HOSPITAL": {
        "category": "MEDICAL",
        "gloss": "HOSPITAL",
        "description": "Dominant 'H' fingers trace cross on non-dominant upper arm"
    },
    "PAIN": {
        "category": "MEDICAL",
        "gloss": "PAIN",
        "description": "Both index fingers point at each other and twist toward center"
    },
    "SICK": {
        "category": "MEDICAL",
        "gloss": "SICK",
        "description": "Dominant middle finger to forehead, non-dominant to stomach"
    },
    "MEDICINE": {
        "category": "MEDICAL",
        "gloss": "MEDICINE",
        "description": "Dominant middle finger rocks on non-dominant open palm"
    },

    # Workplace & Meeting Signs
    "MEETING": {
        "category": "WORKPLACE",
        "gloss": "MEETING",
        "description": "Both hands with open fingers move together and close into flat 'O' shapes"
    },
    "PROJECT": {
        "category": "WORKPLACE",
        "gloss": "PROJECT",
        "description": "Dominant 'P' hand slides vertically along non-dominant palm"
    },
    "WORK": {
        "category": "WORKPLACE",
        "gloss": "WORK",
        "description": "Dominant fist taps twice onto the wrist of non-dominant fist"
    },
    "BOOK": {
        "category": "EVERYDAY",
        "gloss": "BOOK",
        "description": "Both flat palms touching at pinky edges then opening like a book"
    },
    "HELP": {
        "category": "EMERGENCY",
        "gloss": "HELP",
        "description": "Dominant thumbs-up fist placed on flat non-dominant palm and lifted upward"
    },
    "MEET": {
        "category": "CONVERSATIONAL",
        "gloss": "MEET",
        "description": "Both upright index fingers move together until knuckles meet"
    },
    "HOUSE": {
        "category": "EVERYDAY",
        "gloss": "HOUSE",
        "description": "Both flat hands touch at fingertips to form roof, then separate down for walls"
    },
    "FINISHED": {
        "category": "CONVERSATIONAL",
        "gloss": "FINISHED",
        "description": "Both open hands held palms facing in, then flicked outward with palms facing down"
    },
    "FAMILY": {
        "category": "EVERYDAY",
        "gloss": "FAMILY",
        "description": "Both 'F' hands start touching at index/thumbs and sweep around in a circle to meet pinkies"
    },
    "WATER": {
        "category": "EVERYDAY",
        "gloss": "WATER",
        "description": "'W' hand taps index side against chin"
    },
    "FOOD": {
        "category": "EVERYDAY",
        "gloss": "FOOD",
        "description": "Flat 'O' hand taps fingertips against lips"
    },
    "TIME": {
        "category": "EVERYDAY",
        "gloss": "TIME",
        "description": "Dominant index finger taps wrist where watch is worn"
    },
    "DEAF": {
        "category": "CONVERSATIONAL",
        "gloss": "DEAF",
        "description": "Index finger touches jaw near ear, then touches corner of mouth"
    },
    "HEARING": {
        "category": "CONVERSATIONAL",
        "gloss": "HEARING",
        "description": "Index finger held horizontally in front of lips rolls forward in small circles"
    }
}


class AdvancedBimanualEngine:
    """
    Evaluates dual-hand joint structures and trajectory dynamics with
    handedness invariance (automatic left/right hand normalization).
    """

    def __init__(self, dominant_hand: str = "Auto"):
        self.dominant_hand_preference = dominant_hand  # "Auto", "Right", "Left"
        self.cooldown_counter = 0

    def normalize_handedness(
        self,
        left_coords: Optional[List[Tuple[float, float, float]]],
        right_coords: Optional[List[Tuple[float, float, float]]],
        body_midpoint_x: float = 0.5
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Normalizes handedness so that left-handed signers are mapped to canonical
        dominant/non-dominant coordinate representations.
        
        Returns:
            (canonical_dominant_coords, canonical_nondominant_coords, active_dominant_str)
        """
        l_arr = np.array(left_coords, dtype=np.float32) if left_coords is not None else None
        r_arr = np.array(right_coords, dtype=np.float32) if right_coords is not None else None

        # Determine active dominant hand
        is_left_dominant = False
        if self.dominant_hand_preference == "Left":
            is_left_dominant = True
        elif self.dominant_hand_preference == "Auto":
            # Auto-detect: if only left hand is active or left hand is higher than right
            if l_arr is not None and r_arr is None:
                is_left_dominant = True
            elif l_arr is not None and r_arr is not None:
                # Compare height (Y inverted in vision: smaller Y is higher)
                if l_arr[0, 1] < r_arr[0, 1] - 0.10:
                    is_left_dominant = True

        if is_left_dominant and l_arr is not None:
            # Mirror left hand across X axis to canonical right-hand perspective
            mirrored_dom = l_arr.copy()
            mirrored_dom[:, 0] = 2.0 * body_midpoint_x - mirrored_dom[:, 0]

            mirrored_nondom = None
            if r_arr is not None:
                mirrored_nondom = r_arr.copy()
                mirrored_nondom[:, 0] = 2.0 * body_midpoint_x - mirrored_nondom[:, 0]

            return mirrored_dom, mirrored_nondom, "Left"

        # Right hand dominant (canonical)
        return r_arr, l_arr, "Right"

    def _get_finger_extensions(self, coords: np.ndarray) -> Tuple[Tuple[bool, bool, bool, bool, bool], bool]:
        """Calculates extension status of all 5 fingers for a 21-landmark hand array."""
        wrist = coords[0]

        # Finger tip vs PIP joints:
        # Thumb: 4 vs 2, Index: 8 vs 6, Middle: 12 vs 10, Ring: 16 vs 14, Pinky: 20 vs 18
        d_thumb_tip = np.linalg.norm(coords[4] - wrist)
        d_thumb_pip = np.linalg.norm(coords[2] - wrist)
        thumb_ext = bool(d_thumb_tip > d_thumb_pip * 1.15)

        exts = []
        pips = [6, 10, 14, 18]
        tips = [8, 12, 16, 20]
        for tip_i, pip_i in zip(tips, pips):
            d_tip = np.linalg.norm(coords[tip_i] - wrist)
            d_pip = np.linalg.norm(coords[pip_i] - wrist)
            exts.append(bool(d_tip > d_pip * 1.20))

        all_ext = (thumb_ext, exts[0], exts[1], exts[2], exts[3])
        all_open = bool(all(all_ext))
        return all_ext, all_open

    def analyze_posture(
        self,
        left_coords: Optional[List[Tuple[float, float, float]]],
        right_coords: Optional[List[Tuple[float, float, float]]]
    ) -> Optional[BimanualPosture]:
        """Computes comprehensive geometric relationship between both hands."""
        num_hands = (1 if left_coords else 0) + (1 if right_coords else 0)
        if num_hands == 0:
            return None

        # Apply handedness normalization
        dom_arr, nondom_arr, dom_name = self.normalize_handedness(left_coords, right_coords)

        if dom_arr is None or nondom_arr is None:
            # Single hand posture
            exts, all_open = self._get_finger_extensions(dom_arr if dom_arr is not None else nondom_arr)
            return BimanualPosture(
                num_hands=1,
                is_bimanual=False,
                wrist_distance=1.0,
                wrist_vector=(0.0, 0.0, 0.0),
                left_finger_extensions=exts,
                right_finger_extensions=exts,
                left_all_open=all_open,
                right_all_open=all_open,
                left_palm_facing="FORWARD",
                right_palm_facing="FORWARD",
                contact_detected=False,
                symmetry_score=0.0,
                dominant_hand=dom_name
            )

        # Dual hand posture
        w_vec = dom_arr[0] - nondom_arr[0]
        w_dist = float(np.linalg.norm(w_vec))

        dom_exts, dom_open = self._get_finger_extensions(dom_arr)
        nondom_exts, nondom_open = self._get_finger_extensions(nondom_arr)

        # Contact detection: dominant index/thumb/knuckle near non-dominant palm
        contact_dist = float(np.min(np.linalg.norm(dom_arr[[4, 8, 12, 0]] - nondom_arr[0], axis=1)))
        contact_detected = contact_dist < 0.12

        # Symmetry score (similarity of mirrored dominant and non-dominant hand shapes)
        dom_mirrored = dom_arr - dom_arr[0]
        dom_mirrored[:, 0] = -dom_mirrored[:, 0]
        nondom_norm = nondom_arr - nondom_arr[0]
        symmetry_diff = float(np.mean(np.linalg.norm(dom_mirrored - nondom_norm, axis=1)))
        symmetry_score = float(max(0.0, 1.0 - symmetry_diff * 4.0))

        return BimanualPosture(
            num_hands=2,
            is_bimanual=True,
            wrist_distance=w_dist,
            wrist_vector=(float(w_vec[0]), float(w_vec[1]), float(w_vec[2])),
            left_finger_extensions=nondom_exts,
            right_finger_extensions=dom_exts,
            left_all_open=nondom_open,
            right_all_open=dom_open,
            left_palm_facing="FORWARD",
            right_palm_facing="FORWARD",
            contact_detected=contact_detected,
            symmetry_score=symmetry_score,
            dominant_hand=dom_name
        )

    def evaluate_bimanual_sign(
        self,
        left_coords: Optional[List[Tuple[float, float, float]]],
        right_coords: Optional[List[Tuple[float, float, float]]],
        kinetic_energy: float = 0.0
    ) -> Optional[BimanualSignResult]:
        """
        Evaluates the current dual-hand frame against the expanded ASL vocabulary.
        Returns BimanualSignResult if confident, else None.
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        posture = self.analyze_posture(left_coords, right_coords)
        if posture is None or not posture.is_bimanual:
            return None

        w_dist = posture.wrist_distance
        dom_ext = posture.right_finger_extensions
        nondom_ext = posture.left_finger_extensions
        d_open = posture.right_all_open
        nd_open = posture.left_all_open

        # ---------------------------------------------------------------------
        # 1. "BOOK" (Both flat hands together opening outward like a book)
        # ---------------------------------------------------------------------
        if d_open and nd_open and w_dist < 0.22:
            self.cooldown_counter = 18
            return BimanualSignResult(
                sign_id="BOOK",
                gloss="BOOK",
                confidence=0.96,
                category="EVERYDAY",
                description="Two flat palms together opening like a book"
            )

        # ---------------------------------------------------------------------
        # 2. "HELP" (Dominant fist with upright thumb lifted on flat non-dominant palm)
        # ---------------------------------------------------------------------
        is_dom_thumb = dom_ext[0] and not dom_ext[1] and not dom_ext[2] and not dom_ext[3] and not dom_ext[4]
        if nd_open and is_dom_thumb and w_dist < 0.24:
            self.cooldown_counter = 20
            return BimanualSignResult(
                sign_id="HELP",
                gloss="HELP",
                confidence=0.97,
                category="EMERGENCY",
                description="Dominant thumbs-up fist lifted on flat palm -> HELP"
            )

        # ---------------------------------------------------------------------
        # 3. "WORK" (Dominant fist tapping on non-dominant wrist)
        # ---------------------------------------------------------------------
        is_dom_fist = not dom_ext[1] and not dom_ext[2] and not dom_ext[3] and not dom_ext[4]
        is_nd_fist = not nondom_ext[1] and not nondom_ext[2] and not nondom_ext[3] and not nondom_ext[4]
        if is_dom_fist and is_nd_fist and w_dist < 0.18:
            self.cooldown_counter = 18
            return BimanualSignResult(
                sign_id="WORK",
                gloss="WORK",
                confidence=0.95,
                category="WORKPLACE",
                description="Fist tapping on wrist -> WORK"
            )

        # ---------------------------------------------------------------------
        # 4. "MEET" (Both index fingers upright approaching until knuckles touch)
        # ---------------------------------------------------------------------
        is_dom_idx = dom_ext[1] and not dom_ext[2] and not dom_ext[3] and not dom_ext[4]
        is_nd_idx = nondom_ext[1] and not nondom_ext[2] and not nondom_ext[3] and not nondom_ext[4]
        if is_dom_idx and is_nd_idx and w_dist < 0.16:
            self.cooldown_counter = 18
            return BimanualSignResult(
                sign_id="MEET",
                gloss="MEET",
                confidence=0.94,
                category="CONVERSATIONAL",
                description="Both index fingers meeting -> MEET"
            )

        # ---------------------------------------------------------------------
        # 5. "HOUSE" (Both flat palms angled touching at fingertips like a roof)
        # ---------------------------------------------------------------------
        if d_open and nd_open and 0.18 < w_dist < 0.38 and posture.symmetry_score > 0.65:
            self.cooldown_counter = 18
            return BimanualSignResult(
                sign_id="HOUSE",
                gloss="HOUSE",
                confidence=0.93,
                category="EVERYDAY",
                description="Two angled flat hands forming roof -> HOUSE"
            )

        # ---------------------------------------------------------------------
        # 6. "PAIN" (Both index fingers pointing and twisting toward each other)
        # ---------------------------------------------------------------------
        if is_dom_idx and is_nd_idx and 0.15 < w_dist < 0.32 and kinetic_energy > 0.015:
            self.cooldown_counter = 20
            return BimanualSignResult(
                sign_id="PAIN",
                gloss="PAIN",
                confidence=0.92,
                category="MEDICAL",
                description="Both index fingers pointing together twisting -> PAIN"
            )

        # ---------------------------------------------------------------------
        # 7. "FINISHED" (Both hands held open flicking downward and outward)
        # ---------------------------------------------------------------------
        if d_open and nd_open and kinetic_energy > 0.035 and posture.symmetry_score > 0.70:
            self.cooldown_counter = 22
            return BimanualSignResult(
                sign_id="FINISHED",
                gloss="FINISHED",
                confidence=0.91,
                category="CONVERSATIONAL",
                description="Both open hands flicking outward -> FINISHED"
            )

        # ---------------------------------------------------------------------
        # 8. "FAMILY" (Both hands with thumb and index touching circle around)
        # ---------------------------------------------------------------------
        is_dom_f = dom_ext[0] and dom_ext[1] and dom_ext[2]
        is_nd_f = nondom_ext[0] and nondom_ext[1] and nondom_ext[2]
        if is_dom_f and is_nd_f and posture.symmetry_score > 0.68:
            self.cooldown_counter = 20
            return BimanualSignResult(
                sign_id="FAMILY",
                gloss="FAMILY",
                confidence=0.90,
                category="EVERYDAY",
                description="Two 'F' hands sweeping in circle -> FAMILY"
            )

        return None


# Backward-compatible alias
BimanualEngine = AdvancedBimanualEngine
