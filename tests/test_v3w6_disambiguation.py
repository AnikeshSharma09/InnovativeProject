"""
Unit tests for ASL 'V' vs '3' vs 'W' vs '6' vs 'U' mutually exclusive anatomical disambiguation.
Ensures zero confusion between 2-finger upright (V, 3, U, 2) and 3-finger upright (W, 6).
"""
import unittest
import numpy as np
from prediction.predictor import SignPredictor


def build_synthetic_hand(
    thumb_pos,
    idx_ext: bool,
    mid_ext: bool,
    ring_ext: bool,
    pnk_ext: bool,
    thumb_touches_pinky: bool = False,
    is_k: bool = False,
    is_r: bool = False,
    spread_v: bool = True
):
    """
    Synthesizes standard MediaPipe 21-landmark normalized hand coordinates.
    Wrist is at (0.50, 0.80, 0.0), Middle MCP at (0.50, 0.50, 0.0) -> Palm = 0.30.
    """
    coords = [None] * 21
    # Wrist & Thumb base
    coords[0] = [0.50, 0.80, 0.0]  # Wrist
    coords[1] = [0.45, 0.72, 0.0]  # Thumb CMC
    coords[2] = [0.40, 0.65, 0.0]  # Thumb MCP
    coords[3] = [0.36, 0.60, 0.0]  # Thumb IP
    coords[4] = list(thumb_pos)     # Thumb Tip

    # MCPs
    coords[5] = [0.42, 0.52, 0.0]   # Index MCP
    coords[9] = [0.50, 0.50, 0.0]   # Middle MCP
    coords[13] = [0.57, 0.52, 0.0]  # Ring MCP
    coords[17] = [0.63, 0.56, 0.0]  # Pinky MCP

    # Index
    coords[6] = [0.40, 0.42, 0.0]   # Index PIP
    coords[7] = [0.39, 0.34, 0.0]   # Index DIP
    if idx_ext:
        if is_r:
            # Crossed over middle
            coords[8] = [0.52, 0.22, 0.0]
        elif spread_v:
            coords[8] = [0.36, 0.22, 0.0]
        else:
            # Touching middle (parallel U)
            coords[8] = [0.48, 0.22, 0.0]
    else:
        coords[8] = [0.42, 0.58, 0.0]

    # Middle
    coords[10] = [0.50, 0.40, 0.0]  # Middle PIP
    coords[11] = [0.50, 0.32, 0.0]  # Middle DIP
    if mid_ext:
        if is_r:
            # Crossed under index
            coords[8] = [0.52, 0.22, 0.0]
            coords[12] = [0.40, 0.21, 0.0]
        else:
            coords[12] = [0.50, 0.20, 0.0]
    else:
        coords[12] = [0.50, 0.58, 0.0]

    # Ring
    coords[14] = [0.58, 0.42, 0.0]  # Ring PIP
    coords[15] = [0.59, 0.34, 0.0]  # Ring DIP
    if ring_ext:
        coords[16] = [0.62, 0.22, 0.0]
    else:
        coords[16] = [0.58, 0.58, 0.0]

    # Pinky
    coords[18] = [0.65, 0.48, 0.0]  # Pinky PIP
    coords[19] = [0.66, 0.42, 0.0]  # Pinky DIP
    if thumb_touches_pinky:
        coords[20] = [thumb_pos[0] + 0.01, thumb_pos[1] + 0.01, thumb_pos[2]]
    else:
        coords[20] = [0.67, 0.28, 0.0] if pnk_ext else [0.64, 0.60, 0.0]

    if is_k:
        # In ASL 'K', thumb rests between index and middle near middle PIP (coords[10])
        coords[4] = [0.46, 0.43, 0.0]

    return coords


class TestV3W6Disambiguation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SignPredictor()
        cls.classes = cls.predictor.classes

    def _make_probs(self, top_class: str, top_prob: float = 0.55, second_class: str = None, second_prob: float = 0.30):
        probs = np.full(len(self.classes), 0.005)
        if top_class in self.classes:
            probs[self.classes.index(top_class)] = top_prob
        if second_class and second_class in self.classes:
            probs[self.classes.index(second_class)] = second_prob
        return probs / np.sum(probs)

    def test_v_handshape_overcomes_false_w_init(self):
        """Peace sign (V): Ring curled, Pinky curled, Thumb folded, Index & Mid spread."""
        hand = build_synthetic_hand(
            thumb_pos=[0.50, 0.62, 0.0],  # Folded thumb
            idx_ext=True,
            mid_ext=True,
            ring_ext=False,
            pnk_ext=False,
            spread_v=True
        )
        # Even if initial RF predicted 'W' with 55% and 'V' with 30%
        probs_in = self._make_probs("W", 0.55, "V", 0.30)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(hand, probs_in)

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "V", f"Expected 'V' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("V")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("W")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("6")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("3")], 0.02)

    def test_3_handshape_overcomes_false_v_init(self):
        """ASL 3: Thumb extended outward, Index and Middle extended, Ring and Pinky curled."""
        hand = build_synthetic_hand(
            thumb_pos=[0.24, 0.55, 0.0],  # Extended thumb
            idx_ext=True,
            mid_ext=True,
            ring_ext=False,
            pnk_ext=False
        )
        # Even if initial RF predicted 'V' with 60% and '3' with 25%
        probs_in = self._make_probs("V", 0.60, "3", 0.25)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(hand, probs_in)

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "3", f"Expected '3' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("3")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("W")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("6")], 0.02)

    def test_w_handshape_overcomes_false_v_init(self):
        """ASL W: 3 fingers upright & spread, Pinky curled, Thumb folded."""
        hand = build_synthetic_hand(
            thumb_pos=[0.52, 0.62, 0.0],  # Thumb folded over curled pinky
            idx_ext=True,
            mid_ext=True,
            ring_ext=True,
            pnk_ext=False
        )
        # Even if initial RF predicted 'V' with 50% and 'W' with 35%
        probs_in = self._make_probs("V", 0.50, "W", 0.35)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(hand, probs_in)

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "W", f"Expected 'W' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("W")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("3")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("6")], 0.02)

    def test_6_handshape_overcomes_false_w_init(self):
        """ASL 6: Index, Middle, Ring upright, Thumb touches Pinky tip."""
        hand = build_synthetic_hand(
            thumb_pos=[0.58, 0.62, 0.0],  # Thumb touching pinky tip
            idx_ext=True,
            mid_ext=True,
            ring_ext=True,
            pnk_ext=False,
            thumb_touches_pinky=True
        )
        # Even if initial RF predicted 'W' with 65% and '6' with 20%
        probs_in = self._make_probs("W", 0.65, "6", 0.20)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(hand, probs_in)

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "6", f"Expected '6' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("6")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("W")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("3")], 0.02)

    def test_u_handshape_disambiguation(self):
        """ASL U: Index and Middle touching, Ring and Pinky curled, Thumb folded."""
        hand = build_synthetic_hand(
            thumb_pos=[0.50, 0.62, 0.0],
            idx_ext=True,
            mid_ext=True,
            ring_ext=False,
            pnk_ext=False,
            spread_v=False
        )
        probs_in = self._make_probs("V", 0.50, "U", 0.35)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(hand, probs_in)

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "U", f"Expected 'U' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("U")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)

    def test_k_handshape_guard(self):
        """ASL K: Thumb tucked between index and middle near middle PIP."""
        hand = build_synthetic_hand(
            thumb_pos=[0.46, 0.43, 0.0],
            idx_ext=True,
            mid_ext=True,
            ring_ext=False,
            pnk_ext=False,
            is_k=True
        )
        probs_in = self._make_probs("K", 0.60, "V", 0.25)
        probs_out, _ = self.predictor._apply_anatomical_disambiguation(hand, probs_in)
        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "K")

    def test_unrelated_fist_untouched(self):
        """Fist gesture (e.g. 'A'): All fingers curled."""
        hand = build_synthetic_hand(
            thumb_pos=[0.40, 0.55, 0.0],
            idx_ext=False,
            mid_ext=False,
            ring_ext=False,
            pnk_ext=False
        )
        probs_in = self._make_probs("A", 0.70, "S", 0.20)
        probs_out, _ = self.predictor._apply_anatomical_disambiguation(hand, probs_in)
        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "A")


    def test_extreme_rf_bias_handled(self):
        """Even with 85% false probability from RF, physical landmarks must govern."""
        # 1. 85% false W on true V hand
        h_v = build_synthetic_hand([0.50, 0.62, 0.0], True, True, False, False, spread_v=True)
        p_v = self._make_probs("W", 0.85, "V", 0.10)
        out_v, _ = self.predictor._apply_anatomical_disambiguation(h_v, p_v)
        self.assertEqual(self.classes[int(np.argmax(out_v))], "V")

        # 2. 85% false V on true 3 hand
        h_3 = build_synthetic_hand([0.24, 0.55, 0.0], True, True, False, False)
        p_3 = self._make_probs("V", 0.85, "3", 0.10)
        out_3, _ = self.predictor._apply_anatomical_disambiguation(h_3, p_3)
        self.assertEqual(self.classes[int(np.argmax(out_3))], "3")

        # 3. 85% false W on true 6 hand
        h_6 = build_synthetic_hand([0.58, 0.62, 0.0], True, True, True, False, thumb_touches_pinky=True)
        p_6 = self._make_probs("W", 0.85, "6", 0.10)
        out_6, _ = self.predictor._apply_anatomical_disambiguation(h_6, p_6)
        self.assertEqual(self.classes[int(np.argmax(out_6))], "6")

        # 4. 85% false 6 on true W hand
        h_w = build_synthetic_hand([0.52, 0.62, 0.0], True, True, True, False)
        p_w = self._make_probs("6", 0.85, "W", 0.10)
        out_w, _ = self.predictor._apply_anatomical_disambiguation(h_w, p_w)
        self.assertEqual(self.classes[int(np.argmax(out_w))], "W")


if __name__ == "__main__":
    unittest.main()
