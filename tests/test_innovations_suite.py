"""
Unit & Integration Test Suite for the 5 Breakthrough ASL Innovations:
1. Co-Articulation Kinematic Velocity Decoder (utils/coarticulation_engine.py)
2. Lip-Morpheme Facial Fusion Engine (utils/lip_morpheme_engine.py)
3. Ambient Peripheral Vision HUD (utils/ambient_hud.py)
4. One-Shot Biomechanical Hand Calibration (utils/hand_calibrator.py)
5. Air-Workspace In-Meeting Gesture Macro Controller (utils/air_workspace.py)
"""

import sys
import os
import time
import unittest
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.coarticulation_engine import CoArticulationDecoder, ApexEvent
from utils.lip_morpheme_engine import LipMorphemeEngine
from utils.ambient_hud import AmbientPeripheralHUD
from utils.hand_calibrator import HandCalibrator
from utils.air_workspace import AirWorkspaceController, AirMacroEvent
from prediction.predictor import SignPredictor


class TestInnovationsSuite(unittest.TestCase):

    # =========================================================================
    # 1. CO-ARTICULATION KINEMATIC VELOCITY DECODER
    # =========================================================================
    def test_coarticulation_decoder_apex_detection(self):
        decoder = CoArticulationDecoder(energy_window_size=3, apex_velocity_threshold=50.0, min_refractory_sec=0.01)
        base_coords = [(0.5, 0.5, 0.0)] * 21
        
        # Fast movement -> decelerating apex valley -> fast movement
        v_seq = [0.0, 0.05, 0.001, 0.0005, 0.04, 0.05]
        detected_apex = False
        for i, v in enumerate(v_seq):
            c = list(base_coords)
            c[8] = (0.5 + v * i, 0.5, 0.0)
            res = decoder.update_kinematics(c, predicted_label="A", confidence=0.88, timestamp=float(i) * 0.05)
            if res.is_apex and res.emitted_sign == "A":
                detected_apex = True
                break

        self.assertTrue(detected_apex, "Co-articulation engine should detect kinematic stroke apex and emit sign")

    # =========================================================================
    # 2. LIP-MORPHEME FACIAL FUSION ENGINE
    # =========================================================================
    def test_lip_morpheme_kinematic_classification(self):
        engine = LipMorphemeEngine()

        # Construct synthetic FaceMesh landmark points (468 points)
        synth_face = [(0.5, 0.5, 0.0)] * 468
        
        # Bilabial closed lips (M): inner upper (13) and lower (14) very close together
        synth_face[13] = (0.50, 0.50, 0.0)
        synth_face[14] = (0.50, 0.502, 0.0)  # height = 0.002
        synth_face[61] = (0.44, 0.50, 0.0)   # left corner
        synth_face[291] = (0.56, 0.50, 0.0)  # right corner (width = 0.12, MAR = 0.0167)
        res_m = engine.analyze_lip_kinematics(synth_face)
        self.assertEqual(res_m["morpheme"], "BILABIAL_CLOSED")

        # Round pucker lips (O): height is large relative to width
        synth_face[13] = (0.50, 0.45, 0.0)
        synth_face[14] = (0.50, 0.55, 0.0)   # height = 0.10
        synth_face[61] = (0.45, 0.50, 0.0)
        synth_face[291] = (0.55, 0.50, 0.0)  # width = 0.10 (MAR = 1.0)
        res_o = engine.analyze_lip_kinematics(synth_face)
        self.assertEqual(res_o["morpheme"], "ROUND_PUCKER")

        # Alveolar parted lips (N): moderate parting
        synth_face[13] = (0.50, 0.485, 0.0)
        synth_face[14] = (0.50, 0.515, 0.0)  # height = 0.030
        synth_face[61] = (0.42, 0.50, 0.0)
        synth_face[291] = (0.58, 0.50, 0.0)  # width = 0.16 (MAR = 0.187)
        res_n = engine.analyze_lip_kinematics(synth_face)
        self.assertEqual(res_n["morpheme"], "ALVEOLAR_PARTED")

    def test_lip_morpheme_fusion_disambiguation(self):
        engine = LipMorphemeEngine()
        classes = ["M", "N", "C", "O", "D", "1"]
        
        # Test 1: Ambiguous M vs N with closed lips -> resolves to M
        probs = np.array([0.45, 0.44, 0.02, 0.03, 0.03, 0.03], dtype=np.float32)
        synth_face = [(0.5, 0.5, 0.0)] * 468
        synth_face[13] = (0.50, 0.50, 0.0)
        synth_face[14] = (0.50, 0.501, 0.0)
        synth_face[61] = (0.44, 0.50, 0.0)
        synth_face[291] = (0.56, 0.50, 0.0)
        
        fused_probs, note = engine.apply_morpheme_fusion("M", "N", probs, classes, synth_face)
        self.assertGreater(fused_probs[0], 0.75, "M should be favored when lips are closed")
        self.assertIn("Bilabial", note)

        # Test 2: Ambiguous C vs O with round lips -> resolves to O
        probs_co = np.array([0.02, 0.03, 0.44, 0.45, 0.03, 0.03], dtype=np.float32)
        synth_face[13] = (0.50, 0.45, 0.0)
        synth_face[14] = (0.50, 0.55, 0.0)
        synth_face[61] = (0.45, 0.50, 0.0)
        synth_face[291] = (0.55, 0.50, 0.0)
        fused_co, note_co = engine.apply_morpheme_fusion("C", "O", probs_co, classes, synth_face)
        self.assertGreater(fused_co[3], 0.75, "O should be favored when lips are round")
        self.assertIn("Round circular", note_co)

    # =========================================================================
    # 3. AMBIENT PERIPHERAL VISION HUD
    # =========================================================================
    def test_ambient_peripheral_hud(self):
        hud = AmbientPeripheralHUD(border_thickness=12)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Trigger Commit event (cyan glow)
        hud.trigger_event("COMMIT", "A")
        rendered = hud.render_perimeter_halo(frame)
        self.assertEqual(rendered.shape, frame.shape)
        # Verify that top border has cyan illumination
        border_sample = rendered[5, 320]
        self.assertTrue(border_sample[0] > 0 or border_sample[1] > 0, "Top border must be illuminated")

        # Trigger Typo event (crimson pulse)
        hud.trigger_event("TYPO")
        rendered_typo = hud.render_perimeter_halo(frame)
        self.assertEqual(rendered_typo.shape, frame.shape)

        # Trigger Tracking event (emerald shimmer)
        hud.trigger_event("TRACKING")
        rendered_tracking = hud.render_perimeter_halo(frame)
        self.assertEqual(rendered_tracking.shape, frame.shape)

    # =========================================================================
    # 4. ONE-SHOT BIOMECHANICAL HAND CALIBRATION
    # =========================================================================
    def test_hand_calibrator(self):
        calibrator = HandCalibrator(profile_path="data/test_user_profile.json")
        calibrator.reset_profile()
        self.assertFalse(calibrator.profile["calibrated"])

        # Feed 15 synthetic hand frames
        for _ in range(15):
            synth_coords = [(float(i) * 0.03, float(i) * 0.02, 0.0) for i in range(21)]
            calibrator.record_calibration_frame(synth_coords)

        success = calibrator.finish_calibration()
        self.assertTrue(success)
        self.assertTrue(calibrator.profile["calibrated"])

        # Morphological scaling
        scaled = calibrator.apply_morphological_scaling(synth_coords)
        self.assertEqual(len(scaled), 21)

        # Cleanup test profile
        test_p = Path("data/test_user_profile.json")
        if test_p.exists():
            test_p.unlink()

    # =========================================================================
    # 5. AIR-WORKSPACE IN-MEETING GESTURE MACRO CONTROLLER
    # =========================================================================
    def test_air_workspace_macros(self):
        controller = AirWorkspaceController(cooldown_seconds=0.01)

        # Flat open palm held steady:
        # All 5 fingers extended and spread, low kinetic energy
        flat_palm = [
            (0.5, 0.7, 0.0),   # 0: wrist
            (0.45, 0.65, 0.0), (0.42, 0.60, 0.0), (0.40, 0.55, 0.0), (0.38, 0.50, 0.0),  # thumb
            (0.46, 0.55, 0.0), (0.46, 0.48, 0.0), (0.46, 0.42, 0.0), (0.46, 0.35, 0.0),  # index
            (0.50, 0.54, 0.0), (0.50, 0.46, 0.0), (0.50, 0.39, 0.0), (0.50, 0.32, 0.0),  # middle
            (0.54, 0.55, 0.0), (0.54, 0.48, 0.0), (0.54, 0.42, 0.0), (0.54, 0.36, 0.0),  # ring
            (0.58, 0.57, 0.0), (0.58, 0.51, 0.0), (0.58, 0.46, 0.0), (0.58, 0.40, 0.0)   # pinky
        ]

        # Accumulate frames for 1.3 seconds to trigger TOGGLE_MUTE
        triggered_ev = None
        for _ in range(45):
            ev = controller.evaluate_macros(
                active_coords=flat_palm,
                kinetic_energy=0.005,
                is_two_handed=False
            )
            if ev.triggered:
                triggered_ev = ev
                break
            time.sleep(0.035)

        self.assertIsNotNone(triggered_ev, "Flat palm hold should trigger meeting mute macro")
        self.assertEqual(triggered_ev.macro_type, "TOGGLE_MUTE")


if __name__ == "__main__":
    unittest.main()
