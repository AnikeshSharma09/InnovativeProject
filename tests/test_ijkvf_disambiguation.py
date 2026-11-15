"""
Unit test suite specifically verifying zero mixing and mutual exclusivity for:
- I vs J (Static pinky vs Dynamic swoop motion)
- V vs 2 vs K (Spread V vs Number 2 vs Thumb-at-PIP K pose)
- F vs 9 vs B vs D (Thumb-index contact circle vs Number 9 vs Flat hand B vs Thumb-middle loop D)
"""
import unittest
import numpy as np
from prediction.predictor import SignPredictor
import json


class TestIJKVFDisambiguation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SignPredictor()
        cls.classes = cls.predictor.classes
        with open("data/canonical_signs_3d.json", "r") as f:
            cls.canon = json.load(f)

    def _make_probs(self, top_class: str, top_prob: float = 0.65, hostile_class: str = None, hostile_prob: float = 0.25):
        probs = np.full(len(self.classes), 0.005)
        if top_class in self.classes:
            probs[self.classes.index(top_class)] = top_prob
        if hostile_class and hostile_class in self.classes:
            probs[self.classes.index(hostile_class)] = hostile_prob
        return probs / np.sum(probs)

    # -------------------------------------------------------------
    # 1. I vs J Disambiguation Tests
    # -------------------------------------------------------------
    def test_static_i_overcomes_false_j_init(self):
        """Static hold of pinky must predict 'I', even if RF had false 'J' with 70%."""
        coords = self.canon["I"]
        # Hostile: RF predicts J with 70%, I with 20%
        probs_in = self._make_probs("J", 0.70, "I", 0.20)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.008  # Static hold
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "I", f"Expected 'I' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("J")], 0.02)

    def test_dynamic_j_overcomes_false_i_init(self):
        """Dynamic swoop motion of pinky must predict 'J', even if RF had 'I' with 70%."""
        coords = self.canon["J"]
        probs_in = self._make_probs("I", 0.70, "J", 0.20)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.045  # Active swoop motion
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "J", f"Expected 'J' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("I")], 0.02)

    def test_static_j_pose_retains_j(self):
        """Static hold of curved J pose (KE=0.008) must stay 'J' and NOT flip to 'I'."""
        coords = self.canon["J"]
        probs_in = self._make_probs("J", 0.85, "I", 0.05)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.008  # Stationary hold
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "J", f"Expected 'J' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("I")], 0.02)

    def test_moving_i_pose_retains_i(self):
        """Moving hand holding upright I (KE=0.038) must stay 'I' and NOT flip to 'J'."""
        coords = self.canon["I"]
        probs_in = self._make_probs("I", 0.85, "J", 0.05)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.038  # Tremor or movement
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "I", f"Expected 'I' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("J")], 0.02)

    # -------------------------------------------------------------
    # 2. V vs 2 vs K Disambiguation Tests
    # -------------------------------------------------------------
    def test_k_overcomes_false_v_init(self):
        """K sign (thumb at middle PIP) must predict 'K', even if RF predicted 'V' with 80%."""
        coords = self.canon["K"]
        # Hostile prior: 80% V, 10% K
        probs_in = self._make_probs("V", 0.80, "K", 0.10)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "K", f"Expected 'K' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("2")], 0.02)

    def test_k_overcomes_false_2_init(self):
        """K sign must predict 'K', even if RF predicted '2' with 80%."""
        coords = self.canon["K"]
        probs_in = self._make_probs("2", 0.80, "K", 0.10)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "K", f"Expected 'K' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("2")], 0.02)

    def test_v_stable_not_flickering_to_2(self):
        """When showing 'V', '2' must be suppressed and 'V' must remain dominant."""
        coords = self.canon["V"]
        # Both V and 2 high
        probs_in = self._make_probs("V", 0.55, "2", 0.35)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "V", f"Expected 'V' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("2")], 0.02)

    def test_2_recognized_when_decisive_number_predicted(self):
        """When showing number 2, it should output 2 and suppress V."""
        coords = self.canon["2"]
        probs_in = self._make_probs("2", 0.70, "V", 0.20)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "2", f"Expected '2' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)

    def test_v_with_folded_thumb_does_not_flip_to_k(self):
        """V sign with thumb folded across palm must predict 'V' and suppress 'K', even if K has hostile prior."""
        coords = self.canon["V"]
        probs_in = self._make_probs("K", 0.65, "V", 0.25)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "V", f"Expected 'V' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("K")], 0.02)

    def test_k_with_elevated_thumb_does_not_flip_to_v(self):
        """K sign with elevated thumb must predict 'K' and suppress 'V'."""
        coords = self.canon["K"]
        probs_in = self._make_probs("V", 0.65, "K", 0.25)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "K", f"Expected 'K' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)

    # -------------------------------------------------------------
    # 3. F vs 9 vs B vs D Disambiguation Tests
    # -------------------------------------------------------------
    def test_f_overcomes_false_b_init(self):
        """F sign (OK handshape) must predict 'F', even if RF had 'B' with 75%."""
        coords = self.canon["F"]
        probs_in = self._make_probs("B", 0.75, "F", 0.15)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "F", f"Expected 'F' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("B")], 0.02)

    def test_f_not_hijacked_by_9(self):
        """F sign must predict 'F' when both F and 9 are present, without flipping."""
        coords = self.canon["F"]
        probs_in = self._make_probs("F", 0.50, "9", 0.40)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "F", f"Expected 'F' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("9")], 0.02)

    def test_b_flat_hand_not_hijacked_by_f(self):
        """B sign (4 flat fingers, thumb folded across palm) must predict 'B', not 'F'."""
        coords = self.canon["B"]
        probs_in = self._make_probs("F", 0.65, "B", 0.25)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "B", f"Expected 'B' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("F")], 0.02)

    # -------------------------------------------------------------
    # 4. Y vs I and K vs V vs H Tests
    # -------------------------------------------------------------
    def test_y_overcomes_false_i_init(self):
        """Y sign (shaka: thumb + pinky) must predict 'Y', even if RF had false 'I' at 85%."""
        coords = self.canon["Y"]
        probs_in = self._make_probs("I", 0.85, "Y", 0.10)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "Y", f"Expected 'Y' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("I")], 0.02)

    def test_k_overcomes_hostile_h(self):
        """K sign must predict 'K', even if model had false 'H' at 80%."""
        coords = self.canon["K"]
        probs_in = self._make_probs("H", 0.80, "K", 0.10)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "K", f"Expected 'K' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("H")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)

    def test_h_horizontal_two_fingers(self):
        """H sign (horizontal 2 parallel fingers) must predict 'H', not 'V' or 'K'."""
        coords = self.canon["H"]
        probs_in = self._make_probs("V", 0.70, "H", 0.20)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "H", f"Expected 'H' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("V")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("K")], 0.02)
        self.assertLessEqual(probs_out[self.classes.index("1")], 0.02)

    def test_h_horizontal_one_finger_occluded(self):
        """H sign with occluded middle finger pointing horizontal must predict 'H', NEVER '1'."""
        # Take H and curl middle finger slightly so only index is detected as extended
        coords = [list(pt) for pt in self.canon["H"]]
        # Retract middle fingertip towards palm so mid_ext becomes False
        coords[12] = [coords[9][0], coords[9][1] - 0.05, coords[9][2]]
        probs_in = self._make_probs("1", 0.75, "H", 0.15)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "H", f"Expected 'H' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("1")], 0.02)

    def test_1_strictly_upright(self):
        """1 sign pointing upright must predict '1', not 'H' or 'G'."""
        coords = self.canon["1"]
        probs_in = self._make_probs("H", 0.60, "1", 0.25)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "1", f"Expected '1' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("H")], 0.02)

    def test_upright_hand_not_hijacked_by_downward_p(self):
        """Upright K hand with RF noise predicting P must NOT trigger downward P."""
        coords = self.canon["K"]
        # Model predicts P due to noise, but hand is upright pointing at ceiling
        probs_in = self._make_probs("P", 0.60, "K", 0.25)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "K", f"Expected 'K' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("P")], 0.02)

    def test_diagonal_d_not_hijacked_by_g(self):
        """D sign tilted at 45 degrees with hostile RF predicting G must predict 'D', NEVER 'G'."""
        coords = [list(pt) for pt in self.canon["D"]]
        # Tilt index finger at ~45 degrees (v_idx[0] ~ -0.25, v_idx[1] ~ -0.25, ratio_xy ~ 1.0)
        coords[8] = [coords[5][0] - 0.25, coords[5][1] - 0.25, coords[8][2]]
        probs_in = self._make_probs("G", 0.80, "D", 0.15)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "D", f"Expected 'D' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("G")], 0.02)

    def test_relaxed_y_with_abducted_thumb_predicts_y(self):
        """Y sign with realistic thumb abduction (d_tp ~ 0.95, d_tm_pip ~ 0.55) must predict 'Y', NEVER 'I'."""
        coords = [list(pt) for pt in self.canon["I"]]
        # Abduct thumb laterally to form relaxed 'Y' pose
        w = np.array(coords[0])
        palm = np.linalg.norm(np.array(coords[9]) - w)
        coords[4] = [coords[2][0] + 0.65 * palm, coords[2][1] - 0.15 * palm, coords[4][2]]
        # Hostile: RF predicts I with 85%, Y with 0.10
        probs_in = self._make_probs("I", 0.85, "Y", 0.10)
        probs_out, note = self.predictor._apply_anatomical_disambiguation(
            coords, probs_in, kinetic_energy=0.005
        )
        pred = self.classes[int(np.argmax(probs_out))]
        self.assertEqual(pred, "Y", f"Expected 'Y' but got '{pred}' (note: {note})")
        self.assertLessEqual(probs_out[self.classes.index("I")], 0.02)


if __name__ == "__main__":
    unittest.main()
