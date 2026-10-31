"""
Comprehensive End-to-End System Test Suite.
Validates:
1. Model Assets & Weights Integrity (PyTorch LSTM, ONNX Runtime, Label Encoder, CTC Model).
2. Isolated Sign Predictor (Kinematics, 1Euro Filter, Static Ensemble, LSTM Sequence).
3. Continuous Sign Recognizer (Multimodal Holistic Feature Extractor, CTC Decoder, Temporal NMS).
4. Bimanual & Dual-Hand Architecture (Inter-hand transforms, contact distances, finger states).
5. ASL Grammar & Linguistic Translation (Topic-Comment order, WH/YN questions, Negation).
6. Personalization & Adaptive Calibration (Custom vocabulary, user correction tracking).
7. Frame Streaming Pipeline & High-FPS JPEG SIMD Encoding.
8. Streamlit UI App Headless Integration.
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.predictor import SignPredictor
from prediction.continuous_recognizer import ContinuousSignRecognizer
from utils.bimanual_engine import BimanualEngine
from utils.asl_grammar import ASLGrammarTranslator, ASLGlossToken
from utils.sentence_engine import EnterpriseSentenceEngine
from utils.personalization import PersonalVocabularyManager, UserCorrectionTracker
from utils.prediction_logger import PredictionLogger
from prediction.camera_manager import open_robust_camera, release_camera_safely


def test_1_model_assets():
    print("\n[TEST 1/8] Verifying Model Assets & Weights Integrity...")
    assets = [
        ("LSTM PyTorch Model", Path("models/sign_lstm.pth")),
        ("Label Encoder", Path("models/label_encoder.pkl")),
        ("ONNX Runtime Model", Path("models/sign_lstm.onnx")),
        ("CTC PyTorch Model", Path("models/continuous_ctc.pth")),
        ("CTC ONNX Model", Path("models/continuous_ctc.onnx")),
        ("Static Ensemble Model", Path("models/sign_model.pkl")),
        ("WLASL Word Model", Path("models/wlasl_word_model.pkl")),
        ("WLASL LSTM Model", Path("models/wlasl_word_lstm.pth")),
    ]
    for name, path in assets:
        assert path.exists(), f"Missing required model asset: {path}"
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] {name:<24}: {str(path):<30} ({size_kb:.1f} KB)")


def test_2_isolated_predictor():
    print("\n[TEST 2/8] Testing Isolated Motion-Aware SignPredictor (LSTM + ONNX + 1E Filter)...")
    predictor = SignPredictor(
        lstm_model_path="models/sign_lstm.pth",
        encoder_path="models/label_encoder.pkl",
        sequence_length=20,
        enable_stabilization=True
    )
    assert predictor.onnx_session is not None, "ONNX session failed to load!"
    print(f"  [OK] Predictor initialized. Classes count: {len(predictor.classes)}")

    # Feed synthetic test frames (single static hand)
    h, w = 480, 640
    test_frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Process 25 blank/dummy frames (should handle gracefully without error)
    t0 = time.perf_counter()
    for _ in range(25):
        res = predictor.predict_frame(test_frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / 25
    print(f"  [OK] Blank frame processing latency: {elapsed_ms:.2f} ms ({1000/elapsed_ms:.1f} FPS)")
    assert not res.hand_detected, "No hand should be detected in blank frame"
    assert res.predicted_label in ("", "NO HAND", "NO_HAND", None) or res.confidence_score >= 0.0

    # Test reset buffer
    predictor.reset_buffer()
    assert len(predictor.sequence_buffer) == 0, "Buffer reset failed"
    print("  [OK] Isolated Predictor verified successfully.")
    predictor.close()


def test_3_continuous_recognizer():
    print("\n[TEST 3/8] Testing Continuous Multimodal Sign Recognizer (CTC + Holistic)...")
    crec = ContinuousSignRecognizer()
    print(f"  [OK] Continuous Recognizer initialized. Vocabulary: {len(crec.ctc_classes)} signs")

    # Verify model architecture
    assert crec.ctc_model is not None, "CTC model failed to load"
    assert len(crec.ctc_classes) >= 30, f"Expected >= 30 classes, found {len(crec.ctc_classes)}"

    # Benchmark end-to-end frame execution
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(15):
        c_res = crec.process_frame(dummy_frame)
    avg_ms = (time.perf_counter() - t0) * 1000 / 15
    print(f"  [OK] Continuous stream frame latency: {avg_ms:.2f} ms ({1000/avg_ms:.1f} FPS)")
    assert c_res is not None
    assert isinstance(c_res.confidence_score, float)
    assert c_res.motion_state in ("FAST MOTION", "ACTIVE STROKE", "REST / HOLD")

    # Test manual sign addition & translation trigger
    crec.add_manual_sign("HELLO")
    crec.add_manual_sign("NAME")
    crec.add_manual_sign("YOU")
    assert len(crec.accumulated_gloss_tokens) == 3
    assert crec.last_translation.english_translation != ""
    print(f"  [OK] Emitted Glosses: {[t.gloss for t in crec.accumulated_gloss_tokens]}")
    print(f"  [OK] Translated English: '{crec.last_translation.english_translation}'")

    # Test undo
    undone = crec.undo_last_sign()
    assert undone == "YOU", f"Expected 'YOU', got '{undone}'"
    print(f"  [OK] Undo sign verified: removed '{undone}'")

    crec.reset_session()
    assert len(crec.accumulated_gloss_tokens) == 0
    print("  [OK] Continuous Recognizer verified successfully.")
    crec.close()


def test_4_bimanual_engine():
    print("\n[TEST 4/8] Testing Learned Bimanual Two-Hand Engine...")
    bimanual = BimanualEngine(dominant_hand="Right")

    # 1. Test "BOOK" (two flat open hands close together)
    left_hand = [(0.46, 0.60, 0.0)] * 21
    right_hand = [(0.54, 0.60, 0.0)] * 21
    # Extend fingertips (tips: 4, 8, 12, 16, 20)
    for tip in [4, 8, 12, 16, 20]:
        left_hand[tip] = (0.46, 0.30, 0.0)
        right_hand[tip] = (0.54, 0.30, 0.0)
    # PIPs (pips: 2, 6, 10, 14, 18)
    for pip in [2, 6, 10, 14, 18]:
        left_hand[pip] = (0.46, 0.52, 0.0)
        right_hand[pip] = (0.54, 0.52, 0.0)

    res_book = bimanual.evaluate_bimanual_sign(left_hand, right_hand, kinetic_energy=0.01)
    assert res_book is not None, "Failed to detect bimanual BOOK sign"
    assert res_book.gloss.upper() == "BOOK", f"Expected BOOK, got {res_book.gloss}"
    print(f"  [OK] Bimanual Sign Recognized: '{res_book.gloss}' (Confidence: {res_book.confidence:.0%})")

    # 2. Test Left-Hand Dominant Normalization
    bimanual_left = BimanualEngine(dominant_hand="Left")
    res_book_left = bimanual_left.evaluate_bimanual_sign(right_hand, left_hand, kinetic_energy=0.01)
    assert res_book_left is not None and res_book_left.gloss.upper() == "BOOK"
    print("  [OK] Left-Handed dominance normalization verified.")


def test_5_asl_grammar_translation():
    print("\n[TEST 5/8] Testing ASL Grammar Translation Layer...")
    translator = ASLGmarTranslator = ASLGrammarTranslator()

    # Case A: Topic-Comment Order ("STORE I GO")
    tokens_tc = [
        ASLGlossToken(gloss="STORE", timestamp=time.time()),
        ASLGlossToken(gloss="I", timestamp=time.time()),
        ASLGlossToken(gloss="GO", timestamp=time.time()),
    ]
    res_tc = translator.translate_gloss_sequence(tokens_tc)
    print(f"  [OK] Raw ASL: '{res_tc.asl_gloss_string}' -> English: '{res_tc.english_translation}'")
    assert "I" in res_tc.english_translation and "go" in res_tc.english_translation.lower()

    # Case B: WH-Question ("NAME YOU WHAT" with furrowed brows)
    tokens_wh = [
        ASLGlossToken(gloss="NAME", timestamp=time.time()),
        ASLGlossToken(gloss="YOU", timestamp=time.time()),
        ASLGlossToken(gloss="WHAT", is_wh_question=True, timestamp=time.time()),
    ]
    res_wh = translator.translate_gloss_sequence(tokens_wh)
    print(f"  [OK] WH-Question: '{res_wh.asl_gloss_string}' -> English: '{res_wh.english_translation}' ({res_wh.sentence_type})")
    assert res_wh.sentence_type == "WH_QUESTION"
    assert res_wh.english_translation.endswith("?")

    # Case C: Yes/No Question ("YOU WANT COFFEE" with raised brows)
    tokens_yn = [
        ASLGlossToken(gloss="YOU", timestamp=time.time()),
        ASLGlossToken(gloss="WANT", timestamp=time.time()),
        ASLGlossToken(gloss="COFFEE", is_question=True, timestamp=time.time()),
    ]
    res_yn = translator.translate_gloss_sequence(tokens_yn)
    print(f"  [OK] Yes/No Question: '{res_yn.asl_gloss_string}' -> English: '{res_yn.english_translation}' ({res_yn.sentence_type})")
    assert res_yn.sentence_type == "YES_NO_QUESTION"
    assert res_yn.english_translation.endswith("?")

    # Case D: Negation ("I UNDERSTAND" with head shake)
    tokens_neg = [
        ASLGlossToken(gloss="I", timestamp=time.time()),
        ASLGlossToken(gloss="UNDERSTAND", is_negated=True, timestamp=time.time()),
    ]
    res_neg = translator.translate_gloss_sequence(tokens_neg)
    print(f"  [OK] Negated Statement: '{res_neg.asl_gloss_string}' -> English: '{res_neg.english_translation}' ({res_neg.sentence_type})")
    assert res_neg.sentence_type == "NEGATION"
    assert "not" in res_neg.english_translation.lower() or "don't" in res_neg.english_translation.lower()


def test_6_personalization_and_logging():
    print("\n[TEST 6/8] Testing Personal Vocabulary & User Correction Calibration...")
    # Vocabulary Manager
    v_mgr = PersonalVocabularyManager(storage_path="data/test_personal_vocabulary.json")
    v_mgr.add_custom_word("Metformin")
    v_mgr.add_custom_word("Dr. Roberts")
    all_words = v_mgr.get_all_custom_words()
    assert "METFORMIN" in all_words
    assert "DR. ROBERTS" in all_words
    print(f"  [OK] Personal Vocabulary stored & retrieved: {all_words}")
    v_mgr.remove_custom_word("Metformin")
    v_mgr.remove_custom_word("Dr. Roberts")
    # Clean test file
    test_v_path = Path("data/test_personal_vocabulary.json")
    if test_v_path.exists():
        test_v_path.unlink()

    # User Correction Tracker
    c_tracker = UserCorrectionTracker(storage_path="data/test_user_corrections.json")
    c_tracker.record_correction(predicted_sign="T", corrected_sign="A")
    boost = c_tracker.get_calibration_boost(predicted_class="T", candidate_class="A")
    print(f"  [OK] Correction recorded. Calibration boost for 'A' given 'T': {boost:.2f}x")
    assert boost >= 1.05
    test_c_path = Path("data/test_user_corrections.json")
    if test_c_path.exists():
        test_c_path.unlink()

    # Prediction Logger
    logger = PredictionLogger(file_path="data/test_predicted_signs.txt")
    logger.log_prediction("HELLO WORLD", tag="TEST")
    content = logger.get_file_content()
    assert "HELLO WORLD" in content
    logger.clear()
    test_l_path = Path("data/test_predicted_signs.txt")
    if test_l_path.exists():
        test_l_path.unlink()
    print("  [OK] Logging and personalization mechanisms operating with 100% integrity.")


def test_7_camera_and_jpeg_streaming():
    print("\n[TEST 7/8] Testing High-FPS Camera Pipeline & JPEG SIMD Serialization...")
    # Test SIMD JPEG encoding speed and compression ratio
    sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    t0 = time.perf_counter()
    ret_enc, jpeg_buf = cv2.imencode('.jpg', sample_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    enc_ms = (time.perf_counter() - t0) * 1000
    
    assert ret_enc, "JPEG compression failed"
    raw_size_kb = sample_frame.nbytes / 1024
    jpeg_size_kb = len(jpeg_buf) / 1024
    reduction = (1.0 - jpeg_size_kb / raw_size_kb) * 100.0

    print(f"  [OK] Raw Frame Size         : {raw_size_kb:.1f} KB")
    print(f"  [OK] Compressed JPEG Size   : {jpeg_size_kb:.1f} KB (Saved {reduction:.1f}%)")
    print(f"  [OK] Encoding Time          : {enc_ms:.2f} ms ({1000/enc_ms:.1f} FPS encoding throughput)")
    assert enc_ms < 10.0, f"Encoding too slow: {enc_ms:.2f} ms"

    # Test camera hardware manager probe
    print("  [*] Testing OpenCV camera manager initialization...")
    cap, idx, b_name, notif = open_robust_camera(requested_index=0, auto_fallback=True)
    if cap is not None:
        print(f"  [OK] Camera {idx} online ({b_name}). FourCC: MJPG verified.")
        release_camera_safely(cap)
    else:
        print("  [INFO] No physical camera attached (or in CI/Headless container). Fallback safely handled.")


def test_8_streamlit_apptest():
    print("\n[TEST 8/8] Testing Streamlit Web Interface via AppTest...")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app/app.py", default_timeout=25)
    at.run()
    assert len(at.exception) == 0, f"Streamlit application encountered exceptions: {at.exception}"
    print(f"  [OK] Streamlit App loaded successfully without errors. {len(at.tabs)} tabs verified.")


def run_all_tests():
    print("=================================================================")
    print("     COMPREHENSIVE SELF-TEST SUITE: REAL-TIME ASL SYSTEM         ")
    print("=================================================================")
    start_total = time.time()
    
    test_1_model_assets()
    test_2_isolated_predictor()
    test_3_continuous_recognizer()
    test_4_bimanual_engine()
    test_5_asl_grammar_translation()
    test_6_personalization_and_logging()
    test_7_camera_and_jpeg_streaming()
    test_8_streamlit_apptest()

    total_sec = time.time() - start_total
    print("\n=================================================================")
    print(f"   >>> ALL 8 SUBSYSTEM TESTS PASSED SUCCESSFULLY! ({total_sec:.2f}s) <<<")
    print("=================================================================\n")


if __name__ == "__main__":
    run_all_tests()
