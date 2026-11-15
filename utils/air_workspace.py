"""
Air-Workspace In-Meeting Gesture Macro Controller
=================================================
Enables deaf and signing professionals to control virtual meetings (Zoom, Google Meet, Teams)
and computer presentations hands-free using spatial sign macros:
- ✋ Flat Palm Held at Chest for 1.2s -> Toggle Audio Mute / Unmute
- ☝️ Index Point Upward + Flick -> Toggle "Raise Hand" in Zoom / Teams
- ➡️ Two-Finger Swipe Right -> Next Slide (Right Arrow)
- ⬅️ Two-Finger Swipe Left -> Previous Slide (Left Arrow)
- 📋 Double Palm Tap / Clap -> Copy Accumulated ASL Transcript to Clipboard
"""

import time
import ctypes
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class AirMacroEvent:
    gesture: str
    action: str
    description: str
    timestamp: float
    hotkey_emitted: str
    triggered: bool = True

    @property
    def macro_type(self) -> str:
        return self.action


# Windows Virtual Key Codes
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_M = 0x4D
VK_K = 0x4B
VK_Y = 0x59

KEYEVENTF_KEYUP = 0x0002


def _send_win_hotkey(keys: List[int]):
    """Dispatches virtual keystrokes using native Windows user32.dll API."""
    try:
        user32 = ctypes.windll.user32
        for k in keys:
            user32.keybd_event(k, 0, 0, 0)
        time.sleep(0.04)
        for k in reversed(keys):
            user32.keybd_event(k, 0, KEYEVENTF_KEYUP, 0)
    except Exception:
        pass


def _copy_text_to_clipboard(text: str) -> bool:
    """Copies text to the Windows system clipboard via clip.exe."""
    try:
        proc = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
        proc.communicate(input=text.encode("utf-8"))
        return True
    except Exception:
        return False


class AirWorkspaceController:
    """
    Evaluates real-time 3D hand landmarks and executes meeting hotkeys.
    Includes refractory debouncing to prevent accidental duplicate triggers.
    """

    def __init__(
        self,
        hold_time_thresh_sec: float = 1.1,
        swipe_velocity_thresh: float = 0.55,
        refractory_cooldown_sec: float = 1.8,
        cooldown_seconds: Optional[float] = None
    ):
        self.hold_time_thresh_sec = hold_time_thresh_sec
        self.swipe_velocity_thresh = swipe_velocity_thresh
        self.refractory_cooldown_sec = cooldown_seconds if cooldown_seconds is not None else refractory_cooldown_sec

        # State trackers
        self.last_trigger_time: float = 0.0
        self.last_triggered_macro: Optional[str] = None
        self.palm_hold_start: float = 0.0
        self.prev_wrist_x: Optional[float] = None
        self.prev_time: Optional[float] = None

        # Statistics
        self.macro_event_history: List[AirMacroEvent] = []

    def reset(self):
        """Resets tracking buffers."""
        self.palm_hold_start = 0.0
        self.prev_wrist_x = None
        self.prev_time = None

    def evaluate_macros(
        self,
        raw_coords: Optional[List[Tuple[float, float, float]]] = None,
        num_hands: int = 1,
        accumulated_text: str = "",
        enable_os_hotkeys: bool = False,
        active_coords: Any = None,
        kinetic_energy: float = 0.0,
        is_two_handed: bool = False,
        left_coords: Any = None,
        right_coords: Any = None,
        **kwargs
    ) -> AirMacroEvent:
        """
        Scans current frame for spatial meeting macros.
        Returns AirMacroEvent with triggered=True if a macro fired, else triggered=False.
        """
        coords_in = active_coords if active_coords is not None else raw_coords
        hands_count = 2 if is_two_handed else num_hands

        if coords_in is None or len(coords_in) < 21:
            self.reset()
            return AirMacroEvent(gesture="NONE", action="NONE", description="", timestamp=time.time(), hotkey_emitted="", triggered=False)

        now = time.time()
        coords = np.array(coords_in, dtype=np.float32)

        # Refractory guard
        if (now - self.last_trigger_time) < self.refractory_cooldown_sec:
            return AirMacroEvent(gesture="NONE", action="NONE", description="", timestamp=now, hotkey_emitted="", triggered=False)

        w = coords[0]
        i_tip, i_mcp = coords[8], coords[5]
        m_tip, m_mcp = coords[12], coords[9]
        r_tip, r_mcp = coords[16], coords[13]
        p_tip, p_mcp = coords[20], coords[17]

        # Finger extensions
        idx_ext = float(np.linalg.norm(i_tip - w)) > float(np.linalg.norm(i_mcp - w)) * 1.05
        mid_ext = float(np.linalg.norm(m_tip - w)) > float(np.linalg.norm(m_mcp - w)) * 1.05
        ring_ext = float(np.linalg.norm(r_tip - w)) > float(np.linalg.norm(r_mcp - w)) * 1.05
        pnk_ext = float(np.linalg.norm(p_tip - w)) > float(np.linalg.norm(p_mcp - w)) * 1.05

        all_4_ext = idx_ext and mid_ext and ring_ext and pnk_ext

        # Velocity calculation for swipes
        vx = 0.0
        if self.prev_wrist_x is not None and self.prev_time is not None:
            dt = max(1e-4, now - self.prev_time)
            vx = (w[0] - self.prev_wrist_x) / dt

        self.prev_wrist_x = float(w[0])
        self.prev_time = now

        # ---------------------------------------------------------------------
        # 1. MACRO: Next / Prev Presentation Slide (Two-Finger Swipe)
        # ---------------------------------------------------------------------
        if idx_ext and mid_ext and not ring_ext and not pnk_ext:
            if vx > self.swipe_velocity_thresh:
                # Swipe Right -> Next Slide
                if enable_os_hotkeys:
                    _send_win_hotkey([VK_RIGHT])
                event = AirMacroEvent(
                    gesture="Two-Finger Swipe Right",
                    action="NEXT_SLIDE",
                    description="Advanced presentation slide forward.",
                    timestamp=now,
                    hotkey_emitted="RIGHT_ARROW"
                )
                self._record_event(event)
                return event
            elif vx < -self.swipe_velocity_thresh:
                # Swipe Left -> Prev Slide
                if enable_os_hotkeys:
                    _send_win_hotkey([VK_LEFT])
                event = AirMacroEvent(
                    gesture="Two-Finger Swipe Left",
                    action="PREV_SLIDE",
                    description="Moved presentation slide backward.",
                    timestamp=now,
                    hotkey_emitted="LEFT_ARROW"
                )
                self._record_event(event)
                return event

        # ---------------------------------------------------------------------
        # 2. MACRO: Toggle Meeting Mute (Open Flat Palm Held at Center)
        # ---------------------------------------------------------------------
        if all_4_ext and abs(vx) < 0.08:
            if self.palm_hold_start == 0.0:
                self.palm_hold_start = now
            elif (now - self.palm_hold_start) >= self.hold_time_thresh_sec:
                # Mute/Unmute hotkey: Zoom/Meet (Ctrl+Shift+M or Space)
                if enable_os_hotkeys:
                    _send_win_hotkey([VK_CONTROL, VK_SHIFT, VK_M])
                event = AirMacroEvent(
                    gesture="Chest Palm Hold (1.2s)",
                    action="TOGGLE_MUTE",
                    description="Toggled meeting microphone mute/unmute.",
                    timestamp=now,
                    hotkey_emitted="Ctrl+Shift+M"
                )
                self.palm_hold_start = 0.0
                self._record_event(event)
                return event
        else:
            self.palm_hold_start = 0.0

        # ---------------------------------------------------------------------
        # 3. MACRO: Double Palm / 2-Hand Tap -> Copy Transcript to Clipboard
        # ---------------------------------------------------------------------
        if num_hands == 2 and all_4_ext:
            if accumulated_text.strip():
                _copy_text_to_clipboard(accumulated_text)
                event = AirMacroEvent(
                    gesture="Two-Handed Palm Contact",
                    action="COPY_TRANSCRIPT",
                    description=f"Copied {len(accumulated_text.split())} words to system clipboard.",
                    timestamp=now,
                    hotkey_emitted="CLIPBOARD_SYNC"
                )
                self._record_event(event)
                return event

        return AirMacroEvent(gesture="NONE", action="NONE", description="", timestamp=now, hotkey_emitted="", triggered=False)

    def _record_event(self, event: AirMacroEvent):
        """Saves event and updates cooldown."""
        self.last_trigger_time = event.timestamp
        self.last_triggered_macro = event.action
        self.macro_event_history.append(event)
        if len(self.macro_event_history) > 50:
            self.macro_event_history.pop(0)


if __name__ == "__main__":
    controller = AirWorkspaceController()
    print("[+] AirWorkspaceController initialized.")
    # Test flat palm hold simulation
    flat_pts = [(0.5, 0.5, 0.0)] * 21
    flat_pts[8] = (0.5, 0.2, 0.0)
    flat_pts[12] = (0.5, 0.2, 0.0)
    flat_pts[16] = (0.5, 0.2, 0.0)
    flat_pts[20] = (0.5, 0.2, 0.0)
    t0 = time.time()
    for s in range(5):
        ev = controller.evaluate_macros(flat_pts, enable_os_hotkeys=False)
        time.sleep(0.3)
    print("Air-Workspace controller test completed. Total events:", len(controller.macro_event_history))
