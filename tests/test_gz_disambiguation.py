"""
Unit tests for ASL 'G' vs 'Z' vs '1' vs 'D' mutually exclusive anatomical and kinematic disambiguation.
Ensures zero confusion between static index pointing (G, 1, D) and dynamic air-drawing stroke (Z).
"""
import unittest
import numpy as np
from prediction.predictor import SignPredictor


def build_1finger_synthetic_hand(
    thumb_pos,
    v_idx_dir,
    thumb_touches_middle: bool = False,
    is_l: bool = False
):
    """
    Synthesizes standard MediaPipe 21-landmark normalized hand coordinates for 1-finger index gestures.
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

    # Index (Extended)
    coords[6] = [0.40, 0.42, 0.0]   # Index PIP
    coords[7] = [0.39, 0.34, 0.0]   # Index DIP
    coords[8] = [coords[5][0] + v_idx_dir[0], coords[5][1] + v_idx_dir[1], 0.0]  # Index Tip

    # Middle (Curled into palm)
    coords[10] = [0.50, 0.40, 0.0]  # Middle PIP
    coords[11] = [0.50, 0.32, 0.0]
    coords[12] = [0.50, 0.58, 0.0]  # Middle Tip (curled)

    # Ring (Curled into palm)
    coords[14] = [0.58, 0.42, 0.0]
    coords[15] = [0.59, 0.34, 0.0]
    coords[16] = [0.58, 0.58, 0.0]  # Ring Tip (curled)

    # Pinky (Curled into palm)
    coords[18] = [0.65, 0.48, 0.0]
    coords[19] = [0.66, 0.42, 0.0]
    coords[20] = [0.64, 0.60, 0.0]  # Pinky Tip (curled)

    if thumb_touches_middle:
        # In ASL 'D', thumb touches middle tip forming a circular loop
        coords[4] = [coords[12][0] - 0.02, coords[12][1] - 0.02, 0.0]

    if is_l:
        # In ASL 'L', thumb points horizontally at a 90-degree right angle
        coords[4] = [0.22, 0.65, 0.0]

    return coords


class TestGZDisambiguation(unittest.TestCase):
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

    def test_static_g_overcomes_false_z_init(self):
        """Static horizontal index point (G): KE < 0.025, index pointing sideways, thumb parallel."""
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.24, 0.50, 0.0],   # Thumb parallel to index
            v_idx_dir=[-0.25, -0.05]       # Horizontal index pointing to the side
        )
        # Initial false RF prediction: Z with 55% and G with 30%
        probs_in = self._make_probs("Z", 0.55, "G", 0.30)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.008
        )

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "G", f"Expected 'G' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("G")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("Z")], 0.02)

    def test_dynamic_z_overcomes_false_g_init(self):
        """Dynamic drawing stroke (Z): KE >= 0.030, index moving in air."""
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.48, 0.52, 0.0],
            v_idx_dir=[0.15, -0.25]
        )
        # Even if initial prediction favored 'G' with 60% and 'Z' with 25%
        probs_in = self._make_probs("G", 0.60, "Z", 0.25)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.048
        )

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "Z", f"Expected 'Z' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("Z")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("G")], 0.02)

    def test_static_1_overcomes_false_z_init(self):
        """Static upright point (1): KE < 0.025, index straight UP, thumb folded on middle."""
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.48, 0.50, 0.0],   # Thumb folded across middle
            v_idx_dir=[0.00, -0.32]        # Straight UP
        )
        probs_in = self._make_probs("Z", 0.50, "1", 0.35)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.005
        )

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "1", f"Expected '1' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("1")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("Z")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("G")], 0.02)

    def test_static_d_pinch_loop(self):
        """Static D sign: Index straight UP, thumb touching middle tip in circle."""
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.48, 0.56, 0.0],
            v_idx_dir=[0.00, -0.32],
            thumb_touches_middle=True
        )
        probs_in = self._make_probs("1", 0.50, "D", 0.35)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.005
        )

        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "D", f"Expected 'D' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("D")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("1")], 0.02)

    def test_extreme_rf_bias_static_g(self):
        """Even with 85% false Z probability on static G pose, G must govern."""
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.24, 0.50, 0.0],
            v_idx_dir=[-0.25, -0.05]
        )
        probs_in = self._make_probs("Z", 0.85, "G", 0.10)
        probs_out, _ = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.005
        )
        self.assertEqual(self.classes[int(np.argmax(probs_out))], "G")

    def test_extreme_rf_bias_moving_z(self):
        """Even with 85% false G probability during drawing motion, Z must govern."""
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.48, 0.52, 0.0],
            v_idx_dir=[0.15, -0.25]
        )
        probs_in = self._make_probs("G", 0.85, "Z", 0.10)
        probs_out, _ = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.050
        )
        self.assertEqual(self.classes[int(np.argmax(probs_out))], "Z")

    def test_l_pose_preserved(self):
        """L sign: Index straight UP, thumb pointing horizontally at 90 deg corner."""
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.22, 0.65, 0.0],
            v_idx_dir=[0.00, -0.32],
            is_l=True
        )
        probs_in = self._make_probs("L", 0.65, "1", 0.20)
        probs_out, _ = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.005
        )
        self.assertEqual(self.classes[int(np.argmax(probs_out))], "L")

    def test_user_screenshot_diagonal_z_pose(self):
        """Replicates user's screenshot: static diagonal index finger, thumb folded on fist.
        Even with 90% hostile false G prediction, diagonal pose with folded thumb MUST predict Z.
        """
        hand = build_1finger_synthetic_hand(
            thumb_pos=[0.48, 0.52, 0.0],     # Thumb folded on middle MCP / proximal fist
            v_idx_dir=[-0.18, -0.25]         # Diagonal index (~36 deg tilt from vertical, ratio=0.72)
        )
        probs_in = self._make_probs("G", 0.90, "Z", 0.05)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            hand, probs_in, kinetic_energy=0.005
        )
        pred_label = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred_label, "Z", f"Expected 'Z' but got '{pred_label}' (note: {note})")
        self.assertGreaterEqual(probs_out[self.classes.index("Z")], 0.80)
        self.assertLessEqual(probs_out[self.classes.index("G")], 0.02)

    def test_canonical_z_and_g_with_hostile_priors(self):
        """Ensures canonical Z and G 3D landmarks are 100% mutually exclusive."""
        import json
        with open("data/canonical_signs_3d.json", "r") as f:
            canon = json.load(f)

        # Canonical Z with 90% false G
        probs_in_z = self._make_probs("G", 0.90, "Z", 0.05)
        probs_out_z, note_z = self.predictor._apply_anatomical_disambiguation(
            canon["Z"], probs_in_z, kinetic_energy=0.005
        )
        self.assertEqual(self.classes[int(np.argmax(probs_out_z))], "Z", f"Canonical Z failed: {note_z}")

        # Canonical G with 90% false Z
        probs_in_g = self._make_probs("Z", 0.90, "G", 0.05)
        probs_out_g, note_g = self.predictor._apply_anatomical_disambiguation(
            canon["G"], probs_in_g, kinetic_energy=0.005
        )
        self.assertEqual(self.classes[int(np.argmax(probs_out_g))], "G", f"Canonical G failed: {note_g}")


if __name__ == "__main__":
    unittest.main()
