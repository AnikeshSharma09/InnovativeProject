"""
Unit Test Suite for Reverse Speech-to-Sign 3D Avatar Subsystem.
Validates:
1. Canonical 3D landmark database integrity (21 joints per class).
2. English text normalization and contraction expansion.
3. Multi-word idiom mapping to ASL grammatical order.
4. Timed 3D keyframe trajectory generation.
5. Self-contained 3D Canvas visualizer HTML generation.
"""

import sys
import os
import unittest
from pathlib import Path

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.canonical_sign_library import get_canonical_library, HAND_CONNECTIONS, FINGER_GROUPS
from utils.text_to_asl_synthesizer import TextToASLSynthesizer
from utils.reverse_sign_renderer import generate_avatar_html


class TestReverseSignAvatar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.library = get_canonical_library()
        cls.synthesizer = TextToASLSynthesizer()

    def test_canonical_library_coverage(self):
        # Must contain all 26 alphabet letters, 10 digits, and essential control poses
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIn(letter, self.library, f"Missing alphabet letter: {letter}")
            pts = self.library[letter]
            self.assertEqual(len(pts), 21, f"Letter {letter} must have exactly 21 landmarks")
            for pt in pts:
                self.assertEqual(len(pt), 3, f"Each landmark must be 3D [x, y, z]")

        for digit in "0123456789":
            self.assertIn(digit, self.library, f"Missing digit: {digit}")

        self.assertIn("REST", self.library)
        self.assertIn("SPACE", self.library)
        self.assertGreaterEqual(len(self.library), 39)

    def test_hand_connections_topology(self):
        self.assertEqual(len(HAND_CONNECTIONS), 21)
        for c in HAND_CONNECTIONS:
            self.assertEqual(len(c), 2)
            self.assertTrue(0 <= c[0] <= 20)
            self.assertTrue(0 <= c[1] <= 20)

    def test_text_normalization_and_contractions(self):
        raw = "What's up? I'm ready to learn, let's go!"
        norm = self.synthesizer.normalize_english(raw)
        self.assertIn("WHAT IS", norm)
        self.assertIn("I AM", norm)
        self.assertIn("LET US", norm)
        self.assertNotIn("?", norm)
        self.assertNotIn("!", norm)

    def test_idiom_mapping_and_gloss_translation(self):
        # Test idiom "THANK YOU"
        glosses = self.synthesizer.translate_to_asl_glosses("Thank you very much")
        tokens = [g["token"] for g in glosses]
        self.assertIn("THANK", tokens)
        self.assertIn("YOU", tokens)

    def test_animation_payload_generation(self):
        text = "Hello world"
        payload = self.synthesizer.generate_animation_payload(text, hold_time_ms=400, transition_time_ms=200)

        self.assertEqual(payload["input_text"], text)
        self.assertGreater(len(payload["keyframes"]), 5)
        self.assertGreater(payload["total_duration_ms"], 1000)

        # First and last keyframes should be resting posture
        self.assertEqual(payload["keyframes"][0]["sign"], "REST")
        self.assertEqual(payload["keyframes"][-1]["sign"], "REST")

        # Verify keyframe structure
        for kf in payload["keyframes"]:
            self.assertIn("landmarks", kf)
            self.assertEqual(len(kf["landmarks"]), 21)
            self.assertIn("start_time_ms", kf)
            self.assertIn("hold_time_ms", kf)

    def test_html_visualizer_generation(self):
        html_code = generate_avatar_html(default_text="TEST ASL AVATAR")
        self.assertIn("<canvas id=\"avatarCanvas\"", html_code)
        self.assertIn("SpeechRecognition", html_code)
        self.assertIn("project3D", html_code)
        self.assertIn("smoothstep", html_code)
        self.assertGreater(len(html_code), 5000)


if __name__ == "__main__":
    unittest.main()
