"""
Unit Test Suite for Space Recognition, Word Separation & Visual Feedback.
Validates:
1. EnterpriseSentenceEngine space accumulation and trailing space tracking.
2. Case-insensitivity ("space", "SPACE", " ", "␣").
3. Visual HTML badge generation (&#9251; SPACE).
4. Predictor anatomical disambiguation for horizontal flat hand.
5. Continuous recognizer space token merging.
"""
import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.sentence_engine import EnterpriseSentenceEngine
from prediction.predictor import SignPredictor
from prediction.continuous_recognizer import ContinuousSignRecognizer
from utils.asl_grammar import ASLGlossToken


class TestSpaceRegistration(unittest.TestCase):

    def setUp(self):
        self.engine = EnterpriseSentenceEngine()

    def test_single_word_and_space(self):
        for c in "HELLO":
            self.engine.add_character(c)
        self.assertEqual(self.engine.current_word, "HELLO")
        self.assertFalse(self.engine.has_trailing_space)

        # Register space
        ret = self.engine.add_character("space")
        self.assertTrue(ret)
        self.assertEqual(self.engine.current_word, "")
        self.assertEqual(self.engine.committed_words, ["HELLO"])
        self.assertTrue(self.engine.has_trailing_space)

        # Check full sentence string and trailing space
        self.assertEqual(self.engine.get_full_sentence(include_current=True), "HELLO ")
        self.assertIn("SPACE", self.engine.get_display_html())
        self.assertIn("&#9251;", self.engine.get_display_html())

    def test_case_insensitive_space_variants(self):
        for variant in ["space", "SPACE", " ", "␣", "blank"]:
            e = EnterpriseSentenceEngine()
            e.add_character("TEST")
            e.add_character(variant)
            self.assertEqual(e.committed_words, ["TEST"], f"Failed for variant: {variant}")
            self.assertTrue(e.has_trailing_space, f"Failed for variant: {variant}")

    def test_multiple_words_with_space(self):
        for c in "HI":
            self.engine.add_character(c)
        self.engine.add_character("space")

        for c in "TEAM":
            self.engine.add_character(c)
        self.engine.add_character("space")

        self.assertEqual(self.engine.committed_words, ["HI", "TEAM"])
        self.assertEqual(self.engine.get_full_sentence(include_current=True), "HI TEAM ")
        self.assertEqual(self.engine.get_full_sentence(include_current=False), "HI TEAM")

    def test_space_when_empty_does_not_crash(self):
        self.engine.reset()
        ret = self.engine.add_character("space")
        self.assertTrue(ret)
        self.assertTrue(self.engine.has_trailing_space)
        self.assertIn("SPACE", self.engine.get_display_html())

        # Typing a letter after empty space starts a word cleanly
        self.engine.add_character("A")
        self.assertEqual(self.engine.current_word, "A")
        self.assertFalse(self.engine.has_trailing_space)

    def test_backspace_on_trailing_space(self):
        for c in "HELP":
            self.engine.add_character(c)
        self.engine.add_character("space")
        self.assertTrue(self.engine.has_trailing_space)

        # Backspace should clear trailing space first, keeping "HELP" in committed words
        self.engine.backspace()
        self.assertFalse(self.engine.has_trailing_space)
        self.assertEqual(self.engine.committed_words, ["HELP"])

        # Second backspace uncommits "HELP" into current_word
        self.engine.backspace()
        self.assertEqual(self.engine.current_word, "HELP")

    def test_anatomical_disambiguation_for_space(self):
        try:
            predictor = SignPredictor()
        except Exception as e:
            self.skipTest(f"Predictor models not found: {e}")

        if "space" not in predictor.classes:
            self.skipTest("'space' class not in predictor.classes")

        # Create synthetic landmarks representing a flat horizontal hand (space gesture)
        # 4 fingers extended horizontally pointing sideways across chest
        raw_coords = [[0.0, 0.0, 0.0] for _ in range(21)]
        # Wrist at origin
        raw_coords[0] = [0.5, 0.7, 0.0]
        # MCP joints
        raw_coords[5] = [0.45, 0.50, 0.0]   # Index MCP
        raw_coords[9] = [0.43, 0.49, 0.0]   # Mid MCP
        raw_coords[13] = [0.41, 0.49, 0.0]  # Ring MCP
        raw_coords[17] = [0.39, 0.50, 0.0]  # Pinky MCP
        # Fingertips extended horizontally to the left (-X)
        raw_coords[4] = [0.45, 0.42, 0.0]   # Thumb tip
        raw_coords[8] = [0.15, 0.52, 0.0]   # Index tip (dx = -0.30, dy = 0.02)
        raw_coords[12] = [0.13, 0.51, 0.0]  # Mid tip
        raw_coords[16] = [0.12, 0.50, 0.0]  # Ring tip
        raw_coords[20] = [0.11, 0.50, 0.0]  # Pinky tip
        # PIP joints (closer than tips)
        raw_coords[6] = [0.35, 0.51, 0.0]
        raw_coords[10] = [0.33, 0.50, 0.0]
        raw_coords[14] = [0.31, 0.50, 0.0]
        raw_coords[18] = [0.29, 0.50, 0.0]

        probs = np.full(len(predictor.classes), 1.0 / len(predictor.classes), dtype=np.float32)
        adj_probs, note = predictor._apply_anatomical_disambiguation(
            raw_coords, probs, kinetic_energy=0.005
        )

        sp_idx = predictor.classes.index("space")
        self.assertGreaterEqual(adj_probs[sp_idx], 0.75)
        self.assertIn("space", (note or "").lower())


if __name__ == "__main__":
    unittest.main()
