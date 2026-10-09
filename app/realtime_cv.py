"""
Real-Time Motion-Aware ASL Recognition Desktop Application (Native OpenCV).

High-performance, 30+ FPS live webcam system featuring:
- MediaPipe 21 hand landmarks tracking
- Deep Learning PyTorch MotionSignLSTM recurrent sequence inference
- Instantaneous hand kinetic energy and motion state tracking (Static / Active Motion)
- Interactive sentence builder (Space, Backspace, Clear)
- On-demand offline Text-to-Speech (pyttsx3)
- Keyboard shortcuts: [SPACE], [BACKSPACE], [C]=Clear, [S]=Speak, [Q]=Quit
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
from utils.landmarks_util import draw_styled_landmarks
from utils.labels import format_label_for_display
from utils.speech import speak_text


def run_realtime_application(
    camera_index: int = 0,
    lstm_model_path: str = "models/sign_lstm.pth",
    encoder_path: str = "models/label_encoder.pkl",
    auto_commit_frames: int = 20
) -> None:
    """
    Main OpenCV real-time video processing loop with LSTM motion tracking.
    """
    print("\n=======================================================")
    print("  REAL-TIME MOTION-AWARE ASL SIGN RECOGNITION (LSTM)   ")
    print("=======================================================")
    print("[*] Keyboard Controls:")
    print("    - [SPACE]     : Insert Space into sentence")
    print("    - [BACKSPACE] : Delete last character")
    print("    - [A]         : Manually add current stable character")
    print("    - [C]         : Clear accumulated text")
    print("    - [S]         : Speak accumulated text (pyttsx3)")
    print("    - [Q] / [ESC] : Exit application")
    print("=======================================================\n")

    try:
        predictor = SignPredictor(
            lstm_model_path=lstm_model_path,
            encoder_path=encoder_path,
            sequence_length=20
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

    accumulated_text = ""
    current_stable_streak = 0
    last_committed_char = ""
    fps_time = time.time()
    fps = 0.0

    window_name = "Motion-Aware Sign Language Recognition - Deep LSTM HUD"
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

            # Step 1: Run LSTM prediction pipeline
            pred = predictor.predict_frame(frame)

            # Step 2: Draw hand landmarks
            if pred.hand_detected and pred.hand_landmarks_proto:
                draw_styled_landmarks(frame, pred.hand_landmarks_proto)

            # Step 3: Handle Automatic Character Accumulation
            if pred.is_stable and pred.stable_label and not pred.is_uncertain:
                active_char = pred.stable_label
                if active_char == last_committed_char:
                    current_stable_streak = 0
                else:
                    current_stable_streak += 1
                    if current_stable_streak >= auto_commit_frames:
                        if active_char == "space":
                            accumulated_text += " "
                        elif active_char == "del":
                            accumulated_text = accumulated_text[:-1]
                        elif active_char != "nothing":
                            accumulated_text += active_char
                        last_committed_char = active_char
                        current_stable_streak = 0
            else:
                current_stable_streak = 0
                if not pred.hand_detected:
                    last_committed_char = ""

            # Step 4: Render HUD Overlays
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 25), -1)
            cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Color coding based on confidence & motion
            if pred.confidence_level == "HIGH" and not pred.is_uncertain:
                status_color = (60, 220, 60)  # Green
            elif pred.confidence_level == "MEDIUM" and not pred.is_uncertain:
                status_color = (30, 180, 255)  # Amber
            else:
                status_color = (60, 60, 240)  # Red / Uncertain

            disp_sign = pred.display_label if pred.hand_detected else "NO HAND DETECTED"
            cv2.putText(
                frame,
                f"SIGN: {disp_sign}",
                (25, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                status_color,
                3,
                cv2.LINE_AA
            )

            # Display Motion & LSTM details
            status_text = (
                f"LSTM Prob: {pred.raw_probability:.1%} | "
                f"Motion: {pred.motion_state} (E={pred.kinetic_energy:.2f}) | "
                f"Window: [{pred.buffer_fill}/{predictor.sequence_length}] | "
                f"FPS: {fps:.1f}"
            )
            cv2.putText(
                frame,
                status_text,
                (25, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            # Display Auto-commit Progress Bar
            if current_stable_streak > 0:
                bar_width = int((current_stable_streak / auto_commit_frames) * 200)
                cv2.rectangle(frame, (w - 230, 70), (w - 230 + bar_width, 85), status_color, -1)
                cv2.rectangle(frame, (w - 230, 70), (w - 30, 85), (200, 200, 200), 1)
                cv2.putText(frame, "HOLD TO ADD", (w - 230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Bottom Banner: Accumulated Sentence
            cv2.putText(
                frame,
                "RECOGNIZED TEXT:",
                (25, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (170, 170, 170),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                accumulated_text if accumulated_text else "(Perform signs or hold stable to write...)",
                (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            instructions = "[SPC]=Space  [BKSP]=Del  [C]=Clear  [S]=Speak  [Q]=Quit"
            cv2.putText(
                frame,
                instructions,
                (w - 650, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 210, 210),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)

            # Handle Keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c") or key == ord("C"):
                accumulated_text = ""
                print("[*] Cleared text buffer.")
            elif key == 8:
                accumulated_text = accumulated_text[:-1]
                print(f"[*] Text: {accumulated_text}")
            elif key == ord(" ") or key == 32:
                accumulated_text += " "
                print(f"[*] Text: {accumulated_text}")
            elif key == ord("a") or key == ord("A"):
                if pred.is_stable and pred.stable_label and not pred.is_uncertain:
                    if pred.stable_label == "space":
                        accumulated_text += " "
                    elif pred.stable_label == "del":
                        accumulated_text = accumulated_text[:-1]
                    elif pred.stable_label != "nothing":
                        accumulated_text += pred.stable_label
                    print(f"[*] Added '{pred.stable_label}'. Text: {accumulated_text}")
            elif key == ord("s") or key == ord("S"):
                if accumulated_text.strip():
                    print(f"[*] Speaking: '{accumulated_text}'")
                    speak_text(accumulated_text)
                else:
                    print("[!] No text to speak.")

    finally:
        cap.release()
        predictor.close()
        cv2.destroyAllWindows()
        print("\n[*] Application terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Motion-Aware ASL Recognition with LSTM.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--lstm-model", type=str, default="models/sign_lstm.pth", help="LSTM checkpoint path")
    parser.add_argument("--encoder", type=str, default="models/label_encoder.pkl", help="Label encoder path")
    parser.add_argument("--hold-frames", type=int, default=20, help="Frames to hold stable before auto-commit")
    args = parser.parse_args()

    run_realtime_application(
        camera_index=args.camera,
        lstm_model_path=args.lstm_model,
        encoder_path=args.encoder,
        auto_commit_frames=args.hold_frames
    )
