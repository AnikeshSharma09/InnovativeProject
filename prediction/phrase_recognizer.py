"""
Enterprise Phrase & Whole-Sentence Sign Language Recognition Engine.
Enables real-time detection of:
1. Single-Hand Macro Workplace ASL Signs ("APPROVED", "PLEASE WAIT", "I AGREE", "THANK YOU", "HELLO", "STANDUP UPDATE")
2. Two-Handed (Bimanual) ASL Words & Sentences ("BOOK", "HELP", "MEET", "WORK", "HOUSE", "FINISHED", "PLAY")
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import numpy as np


@dataclass
class PhraseDetectionResult:
    """Encapsulates phrase recognition outcome for a video frame sequence."""
    is_phrase_detected: bool
    phrase_id: str
    phrase_text: str
    confidence: float
    description: str
    is_bimanual: bool = False


class EnterprisePhraseRecognizer:
    """
    Detects single-hand and dual-hand ASL signs/words in real-time
    using 3D spatial geometry, inter-hand distances, and motion buffers.
    """

    def __init__(self, history_len: int = 20, enable_single_hand_macros: bool = False):
        self.history_len = history_len
        self.enable_single_hand_macros = enable_single_hand_macros
        self.trajectory_buffer: deque = deque(maxlen=history_len)
        self.raw_coords_buffer: deque = deque(maxlen=history_len)
        self.last_emitted_phrase: Optional[str] = None
        self.cooldown_frames: int = 0

    def reset(self):
        self.trajectory_buffer.clear()
        self.raw_coords_buffer.clear()
        self.last_emitted_phrase = None
        self.cooldown_frames = 0

    @staticmethod
    def _get_finger_extensions(curr: np.ndarray) -> Tuple[bool, bool, bool, bool, bool, bool]:
        """
        Calculates extension status of all 5 fingers for a 21-landmark hand.
        Returns: (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, all_open)
        """
        wrist = curr[0]
        thumb_ext = np.linalg.norm(curr[4] - wrist) > np.linalg.norm(curr[2] - wrist) * 1.15
        index_ext = np.linalg.norm(curr[8] - wrist) > np.linalg.norm(curr[6] - wrist)
        middle_ext = np.linalg.norm(curr[12] - wrist) > np.linalg.norm(curr[10] - wrist)
        ring_ext = np.linalg.norm(curr[16] - wrist) > np.linalg.norm(curr[14] - wrist)
        pinky_ext = np.linalg.norm(curr[20] - wrist) > np.linalg.norm(curr[18] - wrist)
        all_open = (index_ext and middle_ext and ring_ext and pinky_ext)
        return thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, all_open

    def update(
        self,
        coords: Optional[List[Tuple[float, float, float]]] = None,
        kinetic_energy: float = 0.0,
        left_coords: Optional[List[Tuple[float, float, float]]] = None,
        right_coords: Optional[List[Tuple[float, float, float]]] = None,
        wrist_dist: float = 0.0,
        index_dist: float = 0.0
    ) -> Optional[PhraseDetectionResult]:
        """
        Updates recognition buffer and detects both 1-hand and 2-hand signs.
        """
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return None

        # ---------------------------------------------------------------------
        # SECTION A: TWO-HANDED (BIMANUAL) SIGN RECOGNITION
        # ---------------------------------------------------------------------
        if left_coords is not None and right_coords is not None:
            l_arr = np.array(left_coords)
            r_arr = np.array(right_coords)
            tl, il, ml, rl, pl, all_l = self._get_finger_extensions(l_arr)
            tr, ir, mr, rr, pr, all_r = self._get_finger_extensions(r_arr)

            # 1. "BOOK" (Both hands flat open, wrists close, index tips together then opening)
            if all_l and all_r and tl and tr:
                if wrist_dist < 0.22 and index_dist < 0.16:
                    return self._emit_phrase(
                        phrase_id="BOOK",
                        phrase_text="Book",
                        confidence=0.95,
                        description="Two flat hands opened together -> Book",
                        is_bimanual=True
                    )

            # 2. "HELP" (Flat palm base supporting a thumbs-up fist lifting upward)
            is_l_thumb = tl and not il and not ml and not rl and not pl
            is_r_thumb = tr and not ir and not mr and not rr and not pr
            if (all_l and is_r_thumb) or (all_r and is_l_thumb):
                if wrist_dist < 0.24:
                    return self._emit_phrase(
                        phrase_id="HELP",
                        phrase_text="Help",
                        confidence=0.96,
                        description="Flat palm supporting thumbs-up fist -> Help",
                        is_bimanual=True
                    )

            # 3. "MEET" (Both index fingers upright, coming together until tips touch)
            only_il = il and not ml and not rl and not pl
            only_ir = ir and not mr and not rr and not pr
            if only_il and only_ir:
                if index_dist < 0.10:
                    return self._emit_phrase(
                        phrase_id="MEET",
                        phrase_text="Meet",
                        confidence=0.94,
                        description="Two upright index fingers meeting -> Meet",
                        is_bimanual=True
                    )

            # 4. "WORK" (Both hands in fists, one tapping on other wrist)
            is_l_fist = not il and not ml and not rl and not pl
            is_r_fist = not ir and not mr and not rr and not pr
            if is_l_fist and is_r_fist:
                if wrist_dist < 0.18:
                    return self._emit_phrase(
                        phrase_id="WORK",
                        phrase_text="Work",
                        confidence=0.93,
                        description="Fist tapping on wrist -> Work",
                        is_bimanual=True
                    )

            # 5. "HOUSE" (Both flat hands angled touching at apex roof, wrists apart)
            if all_l and all_r:
                if index_dist < 0.09 and wrist_dist > 0.18:
                    return self._emit_phrase(
                        phrase_id="HOUSE",
                        phrase_text="House",
                        confidence=0.92,
                        description="Flat hands touching at apex roof -> House",
                        is_bimanual=True
                    )

            # 6. "FINISHED / ALL DONE" (Both hands open, sweeping outwards apart)
            if all_l and all_r:
                if wrist_dist > 0.30 and kinetic_energy > 0.04:
                    return self._emit_phrase(
                        phrase_id="FINISHED",
                        phrase_text="Finished",
                        confidence=0.92,
                        description="Two open hands sweeping outwards -> Finished",
                        is_bimanual=True
                    )

            # 7. "PLAY" (Both hands in Y-shape: thumb + pinky open)
            is_l_y = tl and pl and not il and not ml and not rl
            is_r_y = tr and pr and not ir and not mr and not rr
            if is_l_y and is_r_y:
                return self._emit_phrase(
                    phrase_id="PLAY",
                    phrase_text="Play",
                    confidence=0.93,
                    description="Two Y-shaped hands oscillating -> Play",
                    is_bimanual=True
                )

        # ---------------------------------------------------------------------
        # SECTION B: SINGLE-HAND SIGN RECOGNITION (OPT-IN WORKPLACE MACROS)
        # ---------------------------------------------------------------------
        if not self.enable_single_hand_macros:
            return None

        primary_coords = coords or right_coords or left_coords
        if primary_coords is None or len(primary_coords) < 21:
            return None

        self.raw_coords_buffer.append(primary_coords)
        self.trajectory_buffer.append(primary_coords[0])  # Wrist trajectory

        if len(self.raw_coords_buffer) < 8:
            return None

        curr = np.array(primary_coords)
        wrist = curr[0]
        thumb_tip = curr[4]
        index_tip = curr[8]
        index_pip = curr[6]
        middle_tip = curr[12]
        middle_pip = curr[10]
        ring_tip = curr[16]
        pinky_tip = curr[20]

        palm_height = np.linalg.norm(curr[9] - wrist)
        if palm_height < 1e-4:
            return None

        thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, all_fingers_open = self._get_finger_extensions(curr)

        # 1. "APPROVED / LOOKS GOOD" (Firm Thumbs-Up Hold)
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if (wrist[1] - thumb_tip[1]) > (palm_height * 0.7):
                if kinetic_energy < 0.05:
                    return self._emit_phrase(
                        phrase_id="APPROVED",
                        phrase_text="Approved / Looks Good!",
                        confidence=0.92,
                        description="Firm Thumbs-Up hold -> Approved"
                    )

        # 2. "PLEASE WAIT" (Flat Open Hand Facing Camera Static Hold)
        if all_fingers_open and thumb_ext:
            fingers_up = (
                index_tip[1] < index_pip[1] and
                middle_tip[1] < middle_pip[1] and
                ring_tip[1] < curr[14][1] and
                pinky_tip[1] < curr[18][1]
            )
            if fingers_up and kinetic_energy < 0.04:
                return self._emit_phrase(
                    phrase_id="PLEASE_WAIT",
                    phrase_text="Please wait a moment",
                    confidence=0.91,
                    description="Open palm facing camera hold -> Please Wait"
                )

        # 3. "I AGREE / YES" (Fist with vertical nodding trajectory)
        is_fist = not index_ext and not middle_ext and not ring_ext and not pinky_ext
        if is_fist and len(self.trajectory_buffer) >= 12:
            y_vals = [pt[1] for pt in self.trajectory_buffer]
            y_range = max(y_vals) - min(y_vals)
            x_vals = [pt[0] for pt in self.trajectory_buffer]
            x_range = max(x_vals) - min(x_vals)
            if y_range > 0.06 and x_range < 0.04 and kinetic_energy > 0.03:
                return self._emit_phrase(
                    phrase_id="AGREE",
                    phrase_text="I agree / Yes",
                    confidence=0.90,
                    description="Fist nodding vertically -> I Agree"
                )

        # 4. "DISAGREE / NO" (Horizontal wave / head-shake style)
        if len(self.trajectory_buffer) >= 12:
            x_vals = [pt[0] for pt in self.trajectory_buffer]
            x_range = max(x_vals) - min(x_vals)
            y_vals = [pt[1] for pt in self.trajectory_buffer]
            y_range = max(y_vals) - min(y_vals)
            if x_range > 0.08 and y_range < 0.04 and kinetic_energy > 0.04:
                return self._emit_phrase(
                    phrase_id="DISAGREE",
                    phrase_text="I disagree / No",
                    confidence=0.88,
                    description="Horizontal lateral snap -> Disagree"
                )

        # 5. "THANK YOU" (Flat open hand moving outward from chin)
        if all_fingers_open and len(self.trajectory_buffer) >= 10:
            dy_out = self.trajectory_buffer[-1][1] - self.trajectory_buffer[0][1]
            dz_out = self.trajectory_buffer[-1][2] - self.trajectory_buffer[0][2]
            if (dy_out > 0.05 or dz_out < -0.04) and kinetic_energy > 0.04:
                return self._emit_phrase(
                    phrase_id="THANK_YOU",
                    phrase_text="Thank you",
                    confidence=0.91,
                    description="Open hand moving outward from face -> Thank You"
                )

        # 6. "STANDUP UPDATE" (Two vertical fingers standing hold)
        two_fingers = index_ext and middle_ext and not ring_ext and not pinky_ext
        if two_fingers:
            fingers_parallel = abs(index_tip[0] - middle_tip[0]) < (palm_height * 0.3)
            if fingers_parallel and kinetic_energy < 0.04:
                return self._emit_phrase(
                    phrase_id="STANDUP",
                    phrase_text="Standup update",
                    confidence=0.90,
                    description="Two fingers standing together upright -> Standup Update"
                )

        # 7. "HELLO / GREETINGS" (High open hand lateral wave)
        if all_fingers_open and wrist[1] < 0.40:
            if len(self.trajectory_buffer) >= 10:
                x_travel = abs(self.trajectory_buffer[-1][0] - self.trajectory_buffer[0][0])
                if x_travel > 0.06 and kinetic_energy > 0.05:
                    return self._emit_phrase(
                        phrase_id="HELLO",
                        phrase_text="Hello",
                        confidence=0.93,
                        description="High open hand wave -> Hello"
                    )

        return None

    def _emit_phrase(
        self,
        phrase_id: str,
        phrase_text: str,
        confidence: float,
        description: str,
        is_bimanual: bool = False
    ) -> PhraseDetectionResult:
        """Constructs phrase result and engages cooldown."""
        self.last_emitted_phrase = phrase_id
        self.cooldown_frames = 25  # ~0.8s cooldown
        return PhraseDetectionResult(
            is_phrase_detected=True,
            phrase_id=phrase_id,
            phrase_text=phrase_text,
            confidence=confidence,
            description=description,
            is_bimanual=is_bimanual
        )
