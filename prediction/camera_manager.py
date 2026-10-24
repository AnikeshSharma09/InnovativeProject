"""
camera_manager.py
=================
Robust cross-platform camera initialization, automatic hardware detection,
and fail-safe backend fallback for OpenCV in Streamlit applications.

Handles:
- DirectShow, MSMF, AVFoundation, and V4L2 backends.
- Windows Hello Infrared (IR) camera index conflicts (e.g., IR at index 0, RGB at index 1).
- Sensor warm-up frame buffering (preventing instant teardown on frame 0).
- Hardware lock-up prevention and safe resolution negotiation.
"""

import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2


def get_platform_backends() -> List[Tuple[str, int]]:
    """Returns the ordered list of OpenCV video capture backends for the current OS."""
    if sys.platform.startswith("win"):
        return [
            ("DirectShow", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("Default", cv2.CAP_ANY),
        ]
    elif sys.platform.startswith("darwin"):
        return [
            ("AVFoundation", getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)),
            ("Default", cv2.CAP_ANY),
        ]
    else:  # Linux / Unix
        return [
            ("V4L2", getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)),
            ("Default", cv2.CAP_ANY),
        ]


def test_single_camera(index: int, backend_name: str, backend_api: int) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Safely tests whether a camera index can be opened and successfully yields an image frame.
    Returns (success, (width, height)).
    """
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend_api)
        if not cap.isOpened():
            return False, None

        # Warm-up attempts: some sensors take 100-300ms to initialize exposure
        for _ in range(4):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                return True, (w, h)
            time.sleep(0.04)

        return False, None
    except Exception:
        return False, None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def scan_all_cameras(max_indices: int = 4) -> List[Dict]:
    """
    Scans camera indices 0..max_indices-1 and reports all physically connected and active webcams.
    """
    results = []
    backends = get_platform_backends()

    for idx in range(max_indices):
        info = {
            "index": idx,
            "working": False,
            "backend": None,
            "resolution": None,
            "note": "Offline or in-use by another app"
        }
        for b_name, b_api in backends:
            success, res = test_single_camera(idx, b_name, b_api)
            if success:
                info["working"] = True
                info["backend"] = b_name
                info["resolution"] = f"{res[0]}x{res[1]}" if res else "Unknown"
                info["note"] = f"Online ({b_name}, {info['resolution']})"
                break
        results.append(info)

    return results


def open_robust_camera(
    requested_index: int = 0,
    preferred_backend: str = "Auto",
    auto_fallback: bool = True
) -> Tuple[Optional[cv2.VideoCapture], int, str, Optional[str]]:
    """
    Robustly opens a webcam with multi-backend fallback and secondary index scanning.

    Args:
        requested_index: Preferred webcam index (usually 0 or 1).
        preferred_backend: "Auto", "DirectShow", "MSMF", or "Default".
        auto_fallback: If True, tests alternative indices (1, 0, 2) if requested fails.

    Returns:
        (cap, actual_index, backend_used, notification_message)
    """
    backends = get_platform_backends()
    
    # Filter backends if user specified a particular one
    if preferred_backend and preferred_backend != "Auto":
        matched = [b for b in backends if b[0].lower() == preferred_backend.lower()]
        if matched:
            # Put preferred first, but keep others as fallbacks
            remaining = [b for b in backends if b[0].lower() != preferred_backend.lower()]
            backends = matched + remaining

    indices_to_try = [requested_index]
    if auto_fallback:
        # If user picked 0, try 1 then 2. If user picked 1, try 0 then 2.
        alternatives = [i for i in [0, 1, 2, 3] if i != requested_index]
        indices_to_try.extend(alternatives)

    notification = None

    for idx in indices_to_try:
        for b_name, b_api in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, b_api)
                if not cap.isOpened():
                    continue

                # Configure gentle properties (don't force if driver rejects)
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                # Warm-up check: read frames to verify image delivery
                delivered = False
                for _ in range(5):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        delivered = True
                        break
                    time.sleep(0.04)

                if delivered:
                    if idx != requested_index:
                        notification = (
                            f"⚠️ Camera {requested_index} was offline or non-responsive (e.g. Windows Hello IR sensor). "
                            f"Automatically switched to active Camera {idx} ({b_name})!"
                        )
                    return cap, idx, b_name, notification
                else:
                    cap.release()
            except Exception:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                continue

    return None, requested_index, "None", "Unable to open any active webcam."


def release_camera_safely(cap: Optional[cv2.VideoCapture]) -> None:
    """Safely closes camera capture and destroys OpenCV windows."""
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
