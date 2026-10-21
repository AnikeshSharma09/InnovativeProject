"""
Real-Time Motion-Aware ASL Recognition Desktop Application (Native OpenCV).

High-performance, 30+ FPS live webcam system featuring:
- MediaPipe 21 hand landmarks tracking
- 1€ Adaptive Anti-Shiver & Tremor Landmark Stabilization
- Direct Sign-to-Word & Sentence Prediction
- Automatic Persistent File Logging (`data/predicted_signs.txt`)
- Deep Learning PyTorch MotionSignLSTM recurrent sequence inference
- Instantaneous hand kinetic energy and motion state tracking (Static / Active Motion)
- Enterprise Whole-Sentence / Phrase ASL Signs Detection
- On-demand offline Text-to-Speech (pyttsx3)
- Keyboard shortcuts:
    - [SPACE]     : Insert Space / Commit word (logs to file)
    - [BACKSPACE] : Delete last character
    - [A]         : Manually add current stable sign
    - [C]         : Clear accumulated text
    - [S]         : Speak text and log complete sentence
    - [Q] / [ESC] : Exit application
"""
import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.predictor import SignPredictor
from utils.landmarks_util import draw_styled_landmarks, draw_dual_styled_landmarks
from utils.labels import format_label_for_display
from utils.speech import speak_text
from utils.sentence_engine import EnterpriseSentenceEngine
from utils.prediction_logger import PredictionLogger


def run_realtime_application(
    camera_index: int = 0,
    lstm_model_path: str = "models/sign_lstm.pth",
    encoder_path: str = "models/label_encoder.pkl",
    auto_commit_frames: int = 15
) -> None:
    """
    Main OpenCV real-time video processing loop with LSTM motion tracking,
    anti-shiver landmark stabilization, and direct prediction logging.
    """
    print("\n=======================================================")
    print("      REAL-TIME MOTION-AWARE ASL SIGN PREDICTOR        ")
    print("   1€ Anti-Tremor Filter • Persistent Prediction Log   ")
    print("=======================================================")
    print("[*] Keyboard Controls:")
    print("    - [SPACE]     : Commit word & log to file")
    print("    - [BACKSPACE] : Delete last character")
    print("    - [A]         : Manually add current stable character")
    print("    - [C]         : Clear sentence buffer")
    print("    - [S]         : Speak text & save sentence to log file")
    print("    - [Q] / [ESC] : Exit application")
    print("=======================================================\n")

    logger = PredictionLogger()
    print(f"[*] Logging predictions to: {logger.file_path.resolve()}\n")

    try:
        predictor = SignPredictor(
            lstm_model_path=lstm_model_path,
            encoder_path=encoder_path,
            sequence_length=20,
            enable_stabilization=True
        )
    except FileNotFoundError as err:
        print(f"\n[!] ERROR: {err}")
        print("Please train the LSTM model first: python training/train_lstm.py")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"\n[!] ERROR: Could not access webcam at index {camera_index}.")
        print("Check your camera connection or try '--camera 1'.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sentence_engine = EnterpriseSentenceEngine()
    current_stable_streak = 0
    last_committed_char = ""
    last_detected_phrase = ""
    fps_time = time.time()
    fps = 0.0

    window_name = "Real-Time Sign Language Prediction & File Logger"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            now = time.time()
            dt = now - fps_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            fps_time = now

            # Step 1: Run prediction pipeline (with 1€ stabilization & phrase recognition)
            pred = predictor.predict_frame(frame)

            # Step 2: Draw hand landmarks (Dual-Hand or Single-Hand)
            if pred.hand_detected:
                if pred.dual_result is not None:
                    draw_dual_styled_landmarks(frame, pred.dual_result)
                elif pred.hand_landmarks_proto:
                    draw_styled_landmarks(frame, pred.hand_landmarks_proto)

            # Step 3: Handle Whole-Sentence & Two-Handed Phrases
            if pred.is_phrase and pred.phrase_text:
                if pred.phrase_text != last_detected_phrase:
                    sentence_engine.add_phrase(pred.phrase_text)
                    logger.log_prediction(pred.phrase_text, tag="2-HAND" if pred.is_two_handed else "PHRASE")
                    last_detected_phrase = pred.phrase_text
                    print(f"\n[★] {'2-HAND ' if pred.is_two_handed else ''}PHRASE SAVED TO FILE: {pred.phrase_text}")

            # Step 4: Handle Automatic Character Accumulation
            if pred.is_stable and pred.stable_label and not pred.is_uncertain:
                active_char = pred.stable_label
                if active_char == last_committed_char:
                    current_stable_streak = 0
                else:
                    current_stable_streak += 1
                    if current_stable_streak >= auto_commit_frames:
                        sentence_engine.add_character(active_char)
                        last_committed_char = active_char
                        current_stable_streak = 0
            else:
                current_stable_streak = 0
                if not pred.hand_detected:
                    last_committed_char = ""
                    last_detected_phrase = ""

            # Check pause autospace (auto-logs finished word to file)
            if sentence_engine.current_word and (time.time() - sentence_engine.last_char_timestamp) > sentence_engine.pause_autospace_seconds:
                word_committed = sentence_engine.commit_current_word(autocorrect=False)
                if word_committed:
                    logger.log_prediction(word_committed, tag="WORD")
                    print(f"[*] Word saved to file: '{word_committed}'")

            # Step 5: Render Top HUD Banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (15, 23, 42), -1)
            cv2.rectangle(overlay, (0, h - 95), (w, h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

            # Color coding based on confidence & motion
            if pred.confidence_level == "HIGH" and not pred.is_uncertain:
                status_color = (60, 220, 60)   # Vibrant Green
            elif pred.confidence_level == "MEDIUM" and not pred.is_uncertain:
                status_color = (30, 180, 255)  # Amber
            else:
                status_color = (60, 60, 240)   # Red / Uncertain

            disp_sign = pred.display_label if pred.hand_detected else "NO HAND DETECTED"
            cv2.putText(
                frame,
                f"SIGN: {disp_sign}",
                (25, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                status_color,
                3,
                cv2.LINE_AA
            )

            # Dual-Hand Indicator & Shiver Status
            hands_mode_str = "🙌 DUAL HANDS (2)" if pred.num_hands == 2 else "✋ SINGLE HAND (1)"
            cv2.putText(
                frame,
                hands_mode_str,
                (420, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 100) if pred.num_hands == 2 else (200, 200, 200),
                2,
                cv2.LINE_AA
            )

            shiver_str = f"SHIVER FILTER: ACTIVE (J={pred.jitter_score:.1f}px)" if pred.is_shivering else f"TREMOR: OK (J={pred.jitter_score:.1f}px)"
            shiver_color = (255, 180, 255) if pred.is_shivering else (180, 220, 180)
            cv2.putText(
                frame,
                shiver_str,
                (420, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                shiver_color,
                1,
                cv2.LINE_AA
            )

            # Whole-phrase notice
            if pred.is_phrase and pred.phrase_text:
                prefix = "2-HAND PHRASE:" if pred.is_two_handed else "PHRASE:"
                cv2.putText(
                    frame,
                    f"{prefix} {pred.phrase_text}",
                    (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (50, 255, 120),
                    2,
                    cv2.LINE_AA
                )
            else:
                # Regular status text
                status_text = (
                    f"Prob: {pred.raw_probability:.1%} | "
                    f"Motion: {pred.motion_state} (E={pred.kinetic_energy:.2f}) | "
                    f"Window: [{pred.buffer_fill}/{predictor.sequence_length}] | "
                    f"FPS: {fps:.1f}"
                )
                cv2.putText(
                    frame,
                    status_text,
                    (25, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )

            # Auto-commit Progress Bar
            if current_stable_streak > 0:
                bar_width = int((current_stable_streak / auto_commit_frames) * 200)
                cv2.rectangle(frame, (w - 230, 70), (w - 230 + bar_width, 85), status_color, -1)
                cv2.rectangle(frame, (w - 230, 70), (w - 30, 85), (200, 200, 200), 1)
                cv2.putText(frame, "HOLD TO ADD", (w - 230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Step 6: Bottom Banner (Live Predicted Sentence & File Storage Indicator)
            full_sentence = sentence_engine.get_full_sentence(include_current=True)
            cv2.putText(
                frame,
                "PREDICTED SENTENCE (Auto-Logged to data/predicted_signs.txt):",
                (25, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (56, 189, 248),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                full_sentence if full_sentence else "(Perform signs to predict words & sentence...)",
                (25, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            instructions = "[SPC]=Commit  [BKSP]=Del  [C]=Clear  [S]=Speak+Save  [Q]=Quit"
            cv2.putText(
                frame,
                instructions,
                (w - 630, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 210, 210),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)

            # Step 7: Handle Keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c") or key == ord("C"):
                sentence_engine.reset()
                print("[*] Cleared sentence buffer.")
            elif key == 8:
                sentence_engine.backspace()
                print(f"[*] Text: {sentence_engine.get_full_sentence()}")
            elif key == ord(" ") or key == 32:
                word = sentence_engine.commit_current_word(autocorrect=False)
                if word:
                    logger.log_prediction(word, tag="WORD")
                    print(f"[*] Logged word: '{word}' -> {logger.file_path}")
            elif key == ord("a") or key == ord("A"):
                if pred.is_stable and pred.stable_label and not pred.is_uncertain:
                    sentence_engine.add_character(pred.stable_label)
                    print(f"[*] Added '{pred.stable_label}'. Text: {sentence_engine.get_full_sentence()}")
            elif key == ord("s") or key == ord("S"):
                spoken = sentence_engine.get_full_sentence(include_current=False)
                if spoken.strip():
                    logger.log_prediction(spoken, tag="SENTENCE")
                    print(f"[*] Logged complete sentence: '{spoken}' -> {logger.file_path}")
                    print(f"[*] Speaking: '{spoken}'")
                    speak_text(spoken)
                else:
                    print("[!] No text to speak.")

    finally:
        cap.release()
        predictor.close()
        cv2.destroyAllWindows()
        print("\n[*] Application terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Sign Language Prediction & Logging.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--lstm-model", type=str, default="models/sign_lstm.pth", help="LSTM checkpoint path")
    parser.add_argument("--encoder", type=str, default="models/label_encoder.pkl", help="Label encoder path")
    parser.add_argument("--hold-frames", type=int, default=15, help="Frames to hold stable before auto-commit")
    args = parser.parse_args()

    run_realtime_application(
        camera_index=args.camera,
        lstm_model_path=args.lstm_model,
        encoder_path=args.encoder,
        auto_commit_frames=args.hold_frames
    )
