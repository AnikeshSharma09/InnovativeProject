"""
Verification test suite for Enterprise & MNC Features:
1. 1€ Anti-Tremor Filter Landmark Variance Reduction
2. Probability Stabilizer Exponential Smoothing
3. Levenshtein Autocorrect & Contextual Autocomplete
4. Whole-Sentence Phrase Detection Biomechanics
"""
import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.stabilizer import LandmarkStabilizer, ProbabilityStabilizer
from utils.sentence_engine import EnterpriseSentenceEngine
from prediction.phrase_recognizer import EnterprisePhraseRecognizer


def test_anti_tremor_variance_reduction():
    print("--- [1] Testing Anti-Tremor 1€ Filter Variance Reduction ---")
    stabilizer = LandmarkStabilizer(min_cutoff=0.8, beta=0.015)
    clean_coords = [(0.5, 0.5, 0.0)] * 21
    noisy_history = []
    stabilized_history = []

    for t in range(60):
        # High-frequency hand tremor oscillation
        shiver_noise = 0.015 * np.sin(t * 1.5)
        noisy_frame = [(x + shiver_noise, y + shiver_noise, z) for (x, y, z) in clean_coords]
        stab_frame, jitter, is_shiv = stabilizer.stabilize(noisy_frame, timestamp=t * 0.033)
        noisy_history.append(noisy_frame[0][0])
        stabilized_history.append(stab_frame[0][0])

    # Discard warm-up period (first 10 frames)
    noisy_var = np.var(noisy_history[10:])
    stab_var = np.var(stabilized_history[10:])
    reduction = (1.0 - stab_var / noisy_var) * 100.0

    print(f"    Raw Tremor Variance       : {noisy_var:.6f}")
    print(f"    Stabilized Tremor Variance : {stab_var:.6f}")
    print(f"    Tremor Dampening Ratio    : {reduction:.1f}%")
    assert reduction >= 65.0, f"Expected >= 65% reduction, got {reduction:.1f}%"
    print("    [PASS] Anti-Tremor 1€ filter successfully dampens hand shivering!")


def test_autocorrect_and_autocomplete():
    print("\n--- [2] Testing NLP Autocorrect & Bigram Autocomplete ---")
    engine = EnterpriseSentenceEngine()
    test_cases = [
        ("HEKLO", "HELLO"),
        ("THABK", "THANK"),
        ("AGREEE", "AGREE"),
        ("MEETIBG", "MEETING")
    ]
    for typo, expected in test_cases:
        engine.current_word = typo
        corrected = engine.commit_current_word(autocorrect=True)
        print(f"    Typo: '{typo}' -> Autocorrected: '{corrected}' (Expected: '{expected}')")
        assert corrected == expected, f"Failed: {typo} -> {corrected} != {expected}"

    # Test predictive autocomplete for last word
    preds = engine.get_next_word_predictions(max_predictions=3)
    print(f"    Contextual predictions after '{expected}': {preds}")
    assert len(preds) > 0, "Expected at least one contextual prediction"
    print("    [PASS] Autocorrect and Autocomplete operating accurately!")


def test_phrase_recognition():
    print("\n--- [3] Testing Whole-Sentence Biomechanical Recognition ---")
    recognizer = EnterprisePhraseRecognizer(history_len=20)
    
    # Construct synthetic "APPROVED / GOOD JOB" (thumbs-up: thumb extended, other fingers curled)
    # 21 points: wrist=(0.5, 0.8, 0.0), thumb_tip=(0.5, 0.4, 0.0), others near wrist
    coords = [(0.5, 0.8, 0.0)] * 21
    # Thumb tip (id 4) far from wrist
    coords[4] = (0.5, 0.3, 0.0)
    # Thumb IP (id 2)
    coords[2] = (0.5, 0.6, 0.0)
    # Middle MCP (id 9)
    coords[9] = (0.5, 0.65, 0.0)
    # Other tips (8, 12, 16, 20) curled close to wrist
    for tip in [8, 12, 16, 20]:
        coords[tip] = (0.5, 0.78, 0.0)
    # PIPs (6, 10, 14, 18)
    for pip in [6, 10, 14, 18]:
        coords[pip] = (0.5, 0.75, 0.0)

    # Feed 15 stationary frames
    detected_result = None
    for _ in range(15):
        res = recognizer.update(coords, kinetic_energy=0.008)
        if res and res.is_phrase_detected:
            detected_result = res
            break

    assert detected_result is not None, "Thumbs-up phrase was not detected!"
    print(f"    Detected Phrase: '{detected_result.phrase_text}' (ID: {detected_result.phrase_id})")
    assert detected_result.phrase_id == "APPROVED"
    print("    [PASS] Whole-Sentence Biomechanical Phrase Recognition functioning!")


def test_dual_hand_recognition():
    print("\n--- [4] Testing Two-Hand (Bimanual) Sign Recognition ---")
    recognizer = EnterprisePhraseRecognizer(history_len=20)

    # 1. Test "BOOK" (Both hands flat open, wrists close, index fingertips close)
    # 21 points for left hand with all 5 fingers extended
    left_hand = [(0.4, 0.6, 0.0)] * 21
    # Extended fingertips for left hand
    left_hand[4] = (0.35, 0.45, 0.0)   # Thumb
    left_hand[8] = (0.38, 0.35, 0.0)   # Index
    left_hand[12] = (0.40, 0.33, 0.0)  # Middle
    left_hand[16] = (0.42, 0.35, 0.0)  # Ring
    left_hand[20] = (0.44, 0.38, 0.0)  # Pinky
    # PIP joints
    for pip, tip in [(6, 8), (10, 12), (14, 16), (18, 20)]:
        left_hand[pip] = (left_hand[tip][0], 0.50, 0.0)

    # 21 points for right hand with all 5 fingers extended (mirrored)
    right_hand = [(0.6, 0.6, 0.0)] * 21
    right_hand[4] = (0.65, 0.45, 0.0)   # Thumb
    right_hand[8] = (0.62, 0.35, 0.0)   # Index
    right_hand[12] = (0.60, 0.33, 0.0)  # Middle
    right_hand[16] = (0.58, 0.35, 0.0)  # Ring
    right_hand[20] = (0.56, 0.38, 0.0)  # Pinky
    for pip, tip in [(6, 8), (10, 12), (14, 16), (18, 20)]:
        right_hand[pip] = (right_hand[tip][0], 0.50, 0.0)

    res_book = recognizer.update(
        left_coords=left_hand,
        right_coords=right_hand,
        wrist_dist=0.18,
        index_dist=0.12,
        kinetic_energy=0.01
    )
    assert res_book is not None and res_book.phrase_id == "BOOK", f"Expected BOOK, got {res_book}"
    print(f"    Detected Two-Hand Word: '{res_book.phrase_text}' (ID: {res_book.phrase_id}, Bimanual={res_book.is_bimanual})")
    print("    [PASS] Two-Hand 'BOOK' sign successfully recognized!")

    # 2. Test "HELP" (Left flat palm + Right thumbs up fist)
    recognizer.reset()
    right_fist = [(0.5, 0.5, 0.0)] * 21
    right_fist[4] = (0.5, 0.25, 0.0)  # Thumb extended high
    right_fist[2] = (0.5, 0.38, 0.0)  # Thumb IP
    for tip in [8, 12, 16, 20]:
        right_fist[tip] = (0.5, 0.48, 0.0) # Tips curled
    for pip in [6, 10, 14, 18]:
        right_fist[pip] = (0.5, 0.45, 0.0)

    res_help = recognizer.update(
        left_coords=left_hand,
        right_coords=right_fist,
        wrist_dist=0.15,
        index_dist=0.18,
        kinetic_energy=0.02
    )
    assert res_help is not None and res_help.phrase_id == "HELP", f"Expected HELP, got {res_help}"
    print(f"    Detected Two-Hand Word: '{res_help.phrase_text}' (ID: {res_help.phrase_id}, Bimanual={res_help.is_bimanual})")
    print("    [PASS] Two-Hand 'HELP' sign successfully recognized!")


if __name__ == "__main__":
    test_anti_tremor_variance_reduction()
    test_autocorrect_and_autocomplete()
    test_phrase_recognition()
    test_dual_hand_recognition()
    print("\n=======================================================")
    print("   ALL SINGLE & DUAL-HAND TESTS PASSED (100%)          ")
    print("=======================================================")
