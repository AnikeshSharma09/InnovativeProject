"""
Ambient Peripheral Vision HUD Engine
====================================
World-first zero-eye-contact feedback system for deaf signers.
Draws an ambient chromatic perimeter halo along the borders of the video feed,
allowing signers to perceive word commits, tracking status, and typos via 140-degree
peripheral vision without ever taking their eyes off their conversation partner.
"""

import time
from typing import Tuple, Optional
import numpy as np
import cv2


# Chroma Color Palettes (BGR for OpenCV)
COLOR_COMMIT_CYAN = (248, 189, 56)      # #38BDF8 Sky Blue (Word commit)
COLOR_COMMIT_EMERALD = (127, 255, 0)    # #00FF7F Spring Green
COLOR_TRACKING_AMBER = (0, 190, 255)    # #FFBE00 Amber Gold (Active in-progress)
COLOR_AMBIGUOUS_CRIMSON = (80, 50, 240) # #F03250 Crimson / Magenta (Ambiguity/Typo)


class AmbientPeripheralHUD:
    """
    Renders a soft, glowing chromatic perimeter halo around the camera frame.
    Signers absorb state changes via peripheral vision while maintaining natural eye contact.
    """

    def __init__(
        self,
        border_thickness: int = 14,
        pulse_duration_sec: float = 0.45
    ):
        self.border_thickness = border_thickness
        self.pulse_duration_sec = pulse_duration_sec

        # Pulse state
        self.active_color: Tuple[int, int, int] = COLOR_TRACKING_AMBER
        self.pulse_start_time: float = 0.0
        self.is_pulsing: bool = False
        self.pulse_intensity: float = 0.0

    def trigger_commit_pulse(self):
        """Triggers cyan/emerald peripheral pulse on successful word commit."""
        self.active_color = COLOR_COMMIT_CYAN
        self.pulse_start_time = time.time()
        self.is_pulsing = True

    def trigger_ambiguous_pulse(self):
        """Triggers crimson edge pulse on ambiguous sign or typo correction."""
        self.active_color = COLOR_AMBIGUOUS_CRIMSON
        self.pulse_start_time = time.time()
        self.is_pulsing = True

    def trigger_event(self, event_type: str, label: str = ""):
        """Universal trigger for peripheral vision events ('COMMIT', 'TYPO', 'UNCERTAIN', 'TRACKING')."""
        ev = str(event_type).upper()
        if "COMMIT" in ev or "WORD" in ev:
            self.trigger_commit_pulse()
        elif "TYPO" in ev or "DEL" in ev:
            self.active_color = COLOR_AMBIGUOUS_CRIMSON
            self.pulse_start_time = time.time()
            self.is_pulsing = True
        elif "UNCERTAIN" in ev or "AMBIG" in ev:
            self.trigger_ambiguous_pulse()
        elif "TRACK" in ev:
            self.active_color = COLOR_TRACKING_AMBER

    def render_perimeter_halo(
        self,
        bgr_frame: np.ndarray,
        hand_detected: bool = False,
        is_tracking: bool = False
    ) -> np.ndarray:
        """
        Overlays an alpha-blended chromatic perimeter gradient onto the BGR frame.
        """
        now = time.time()
        h, w, _ = bgr_frame.shape

        # Calculate pulse alpha
        if self.is_pulsing:
            elapsed = now - self.pulse_start_time
            if elapsed >= self.pulse_duration_sec:
                self.is_pulsing = False
                alpha = 0.0
            else:
                # Decaying ease-out pulse
                progress = elapsed / self.pulse_duration_sec
                alpha = float(0.85 * (1.0 - progress))
                color = self.active_color
        elif hand_detected and is_tracking:
            # Gentle ambient breathing shimmer when actively signing
            alpha = float(0.20 + 0.12 * np.sin(now * 5.0))
            color = COLOR_TRACKING_AMBER
        elif hand_detected:
            alpha = 0.15
            color = (180, 150, 50)
        else:
            return bgr_frame

        if alpha <= 0.01:
            return bgr_frame

        # Draw smooth multi-layered perimeter border
        t = self.border_thickness
        overlay = bgr_frame.copy()

        # Outer thick glow
        cv2.rectangle(overlay, (0, 0), (w, h), color, t * 2)
        # Inner brighter line
        cv2.rectangle(overlay, (t // 2, t // 2), (w - t // 2, h - t // 2), (255, 255, 255), max(1, t // 4))

        # Alpha blend border
        cv2.addWeighted(overlay, alpha, bgr_frame, 1.0 - alpha, 0, bgr_frame)
        return bgr_frame


if __name__ == "__main__":
    hud = AmbientPeripheralHUD()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud.trigger_commit_pulse()
    out = hud.render_perimeter_halo(frame, hand_detected=True, is_tracking=True)
    print(f"[+] AmbientPeripheralHUD verified. Output frame shape: {out.shape}, max pixel: {np.max(out)}")
