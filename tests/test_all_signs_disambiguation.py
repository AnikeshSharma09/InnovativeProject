"""
Comprehensive Unit Test Suite for All ASL Alphabets (A-Z) and Numbers (0-9).
Verifies 100% anatomical and kinematic disambiguation across all 36 core signs.
"""
import json
import unittest
import numpy as np
from prediction.predictor import SignPredictor


class TestAllSignsDisambiguation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SignPredictor()
        cls.classes = cls.predictor.classes
        with open("data/canonical_signs_3d.json", "r") as f:
            cls.canon = json.load(f)

    def _make_hostile_probs(self, target_class: str, hostile_class: str, target_prob: float = 0.25, hostile_prob: float = 0.65):
        probs = np.full(len(self.classes), 0.005)
        if hostile_class in self.classes:
            probs[self.classes.index(hostile_class)] = hostile_prob
        if target_class in self.classes:
            probs[self.classes.index(target_class)] = target_prob
        return probs / np.sum(probs)

    def test_all_10_numbers_resolve_accurately(self):
        """All 10 numbers (0-9) must resolve correctly against canonical landmarks."""
        confusing_pairs = {
            "0": "O",
            "1": "Z",
            "2": "3",  # 2 fingers vs 3 digits
            "3": "V",  # 3 digits vs 2 fingers
            "4": "B",  # 4 spread fingers vs B closed
            "5": "4",  # 5 open digits vs 4 fingers
            "6": "W",  # 6 thumb-pinky vs W open
            "7": "6",  # 7 thumb-ring vs 6 thumb-pinky
            "8": "7",  # 8 thumb-middle vs 7 thumb-ring
            "9": "8",  # 9 thumb-index vs 8 thumb-middle
        }
        for num_str in [str(i) for i in range(10)]:
            self.assertIn(num_str, self.canon, f"Canonical sign missing: {num_str}")
            coords = self.canon[num_str]
            hostile_cls = confusing_pairs[num_str]
            probs_in = self._make_hostile_probs(num_str, hostile_cls)
            probs_out, note = self.predictor._apply_anatomical_disambiguation(
                coords, probs_in, kinetic_energy=0.005
            )
            pred = self.classes[int(np.argmax(probs_out))]
            self.assertEqual(pred, num_str, f"Number '{num_str}' misclassified as '{pred}' (hostile: {hostile_cls}, note: {note})")

    def test_all_26_alphabet_letters_resolve_accurately(self):
        """All 26 letters (A-Z) must resolve correctly against canonical landmarks."""
        confusing_pairs = {
            "A": "S", "B": "4", "C": "O", "D": "1", "E": "S",
            "F": "B", "G": "Z", "H": "U", "I": "J", "J": "I",
            "K": "V", "L": "1", "M": "N", "N": "M", "O": "C",
            "P": "Q", "Q": "P", "R": "U", "S": "A", "T": "A",
            "U": "V", "V": "U", "W": "6", "X": "1", "Y": "I", "Z": "G"
        }
        for char_code in range(ord("A"), ord("Z") + 1):
            letter = chr(char_code)
            self.assertIn(letter, self.canon, f"Canonical sign missing: {letter}")
            coords = self.canon[letter]
            hostile_cls = confusing_pairs[letter]
            probs_in = self._make_hostile_probs(letter, hostile_cls)
            ke = 0.040 if letter in ("J", "Z") else 0.005
            probs_out, note = self.predictor._apply_anatomical_disambiguation(
                coords, probs_in, kinetic_energy=ke
            )
            pred = self.classes[int(np.argmax(probs_out))]
            self.assertEqual(pred, letter, f"Letter '{letter}' misclassified as '{pred}' (hostile: {hostile_cls}, note: {note})")

    def test_fist_signs_mutual_exclusivity(self):
        """A, S, T, E must be strictly mutually exclusive."""
        for sign in ["A", "S", "T", "E"]:
            coords = self.canon[sign]
            probs_in = self._make_hostile_probs(sign, "A" if sign != "A" else "S")
            probs_out, note = self.predictor._apply_anatomical_disambiguation(
                coords, probs_in, kinetic_energy=0.005
            )
            pred = self.classes[int(np.argmax(probs_out))]
            self.assertEqual(pred, sign, f"Fist sign '{sign}' predicted as '{pred}'")

    def test_contact_pinches_6_7_8_9(self):
        """Numbers 6, 7, 8, 9 contact loops must be strictly distinguishable."""
        coords_6 = self.canon["6"]
        coords_7 = self.canon["7"]
        coords_8 = self.canon["8"]
        coords_9 = self.canon["9"]

        # 6 (Thumb touches pinky)
        p6, _ = self.predictor._apply_anatomical_disambiguation(coords_6, self._make_hostile_probs("6", "W"))
        self.assertEqual(self.classes[int(np.argmax(p6))], "6")

        # 7 (Thumb touches ring)
        p7, _ = self.predictor._apply_anatomical_disambiguation(coords_7, self._make_hostile_probs("7", "6"))
        self.assertEqual(self.classes[int(np.argmax(p7))], "7")

        # 8 (Thumb touches middle)
        p8, _ = self.predictor._apply_anatomical_disambiguation(coords_8, self._make_hostile_probs("8", "7"))
        self.assertEqual(self.classes[int(np.argmax(p8))], "8")

        # 9 (Thumb touches index)
        p9, _ = self.predictor._apply_anatomical_disambiguation(coords_9, self._make_hostile_probs("9", "F"))
        self.assertEqual(self.classes[int(np.argmax(p9))], "9")

    def test_downward_signs_p_and_q(self):
        """P vs Q downward hand disambiguation."""
        coords_p = self.canon["P"]
        coords_q = self.canon["Q"]

        pp, _ = self.predictor._apply_anatomical_disambiguation(coords_p, self._make_hostile_probs("P", "Q"))
        self.assertEqual(self.classes[int(np.argmax(pp))], "P")

        pq, _ = self.predictor._apply_anatomical_disambiguation(coords_q, self._make_hostile_probs("Q", "P"))
        self.assertEqual(self.classes[int(np.argmax(pq))], "Q")

    def test_pinky_cluster_i_j_y(self):
        """I (static), J (dynamic motion), Y (shaka with thumb)."""
        coords_i = self.canon["I"]
        coords_j = self.canon["J"]
        coords_y = self.canon["Y"]

        # Static hold -> I
        pi, _ = self.predictor._apply_anatomical_disambiguation(coords_i, self._make_hostile_probs("I", "J"), kinetic_energy=0.005)
        self.assertEqual(self.classes[int(np.argmax(pi))], "I")

        # Dynamic motion -> J
        pj, _ = self.predictor._apply_anatomical_disambiguation(coords_j, self._make_hostile_probs("J", "I"), kinetic_energy=0.045)
        self.assertEqual(self.classes[int(np.argmax(pj))], "J")

        # Thumb + Pinky -> Y
        py, _ = self.predictor._apply_anatomical_disambiguation(coords_y, self._make_hostile_probs("Y", "I"), kinetic_energy=0.005)
        self.assertEqual(self.classes[int(np.argmax(py))], "Y")


if __name__ == "__main__":
    unittest.main()
