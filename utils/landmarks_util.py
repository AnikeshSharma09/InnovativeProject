"""
Landmark extraction and normalization utilities for Hand Gestures & Motions.
Supports:
1. Static 63 spatial features (21 landmarks * 3 coordinates).
2. Motion-enhanced 129 features:
   - 63 spatial normalized coordinates relative to wrist.
   - 63 first-order landmark velocities (finger articulation speed & direction).
   - 3 global wrist trajectory deltas (spatial path of whole hand).
   - Instantaneous hand kinetic energy.
Ensures 100% strict mathematical parity between dataset training and live video inference.
"""
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import cv2
import numpy as np

# Landmark constants
NUM_LANDMARKS = 21
NUM_COORDINATES = 3
TOTAL_SPATIAL_FEATURES = NUM_LANDMARKS * NUM_COORDINATES  # 63
TOTAL_FEATURES = TOTAL_SPATIAL_FEATURES  # Backward-compatible alias
TOTAL_VELOCITY_FEATURES = NUM_LANDMARKS * NUM_COORDINATES  # 63
TOTAL_TRAJECTORY_FEATURES = 3  # Wrist dx, dy, dz
TOTAL_MOTION_FEATURES = TOTAL_SPATIAL_FEATURES + TOTAL_VELOCITY_FEATURES + TOTAL_TRAJECTORY_FEATURES  # 129

DEFAULT_SEQUENCE_LENGTH = 20


def get_feature_names() -> List[str]:
    """
    Returns standardized column names for the 63 spatial landmark features.
    Format: ['x_0', 'y_0', 'z_0', ..., 'x_20', 'y_20', 'z_20']
    """
    feature_names = []
    for i in range(NUM_LANDMARKS):
        feature_names.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    return feature_names


def get_motion_feature_names() -> List[str]:
    """
    Returns standardized column names for the 129 motion features:
    [x_0..z_20, vx_0..vz_20, wrist_dx, wrist_dy, wrist_dz]
    """
    names = get_feature_names()
    for i in range(NUM_LANDMARKS):
        names.extend([f"vx_{i}", f"vy_{i}", f"vz_{i}"])
    names.extend(["wrist_dx", "wrist_dy", "wrist_dz"])
    return names


def normalize_landmarks(raw_landmarks: List[Tuple[float, float, float]]) -> List[float]:
    """
    Normalizes hand landmark coordinates to be translation and scale invariant.
    
    1. Translation Invariance: Landmark 0 (wrist) is set as origin (0, 0, 0).
       Relative coordinates: x_i' = x_i - x_0, y_i' = y_i - y_0, z_i' = z_i - z_0.
    2. Scale Invariance: Normalized by the maximum Euclidean distance from the wrist,
       preventing hand distance from the camera from skewing classification.
    
    Args:
        raw_landmarks: List of 21 (x, y, z) tuples.
        
    Returns:
        List of 63 normalized floating-point features.
    """
    if len(raw_landmarks) != NUM_LANDMARKS:
        raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {len(raw_landmarks)}")

    # Extract wrist landmark as origin reference point
    base_x, base_y, base_z = raw_landmarks[0]

    # Step 1: Shift coordinates relative to wrist
    relative_coords = []
    for x, y, z in raw_landmarks:
        relative_coords.append((x - base_x, y - base_y, z - base_z))

    # Step 2: Compute maximum Euclidean distance from the wrist
    max_distance = 0.0
    for rx, ry, rz in relative_coords:
        distance = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        if distance > max_distance:
            max_distance = distance

    # Avoid zero division if landmarks collapse
    if max_distance < 1e-6:
        max_distance = 1.0

    # Step 3: Scale relative coordinates by max_distance
    normalized_features: List[float] = []
    for rx, ry, rz in relative_coords:
        normalized_features.extend([
            float(rx / max_distance),
            float(ry / max_distance),
            float(rz / max_distance)
        ])

    return normalized_features


def extract_motion_features(
    current_raw_landmarks: List[Tuple[float, float, float]],
    previous_raw_landmarks: Optional[List[Tuple[float, float, float]]] = None
) -> Tuple[List[float], float]:
    """
    Extracts 129 spatiotemporal motion features:
    1. 63 spatial normalized features (relative to wrist and scaled).
    2. 63 joint velocities (first-order temporal differences).
    3. 3 global wrist movement deltas (trajectory through camera space).
    4. Instantaneous kinetic motion energy.
    
    Returns:
        Tuple of (129-dimensional feature vector, kinetic_energy_value).
    """
    # 1. Spatial normalized features (63)
    spatial_features = normalize_landmarks(current_raw_landmarks)

    velocities = []
    kinetic_energy = 0.0

    if previous_raw_landmarks is not None and len(previous_raw_landmarks) == NUM_LANDMARKS:
        # Calculate velocity for each landmark
        for i in range(NUM_LANDMARKS):
            cx, cy, cz = current_raw_landmarks[i]
            px, py, pz = previous_raw_landmarks[i]
            vx = cx - px
            vy = cy - py
            vz = cz - pz
            velocities.extend([float(vx), float(vy), float(vz)])
            kinetic_energy += float(vx ** 2 + vy ** 2 + vz ** 2)

        # Global wrist trajectory displacement
        cw_x, cw_y, cw_z = current_raw_landmarks[0]
        pw_x, pw_y, pw_z = previous_raw_landmarks[0]
        wrist_delta = [float(cw_x - pw_x), float(cw_y - pw_y), float(cw_z - pw_z)]
    else:
        # Initial frame in sequence has zero velocity
        velocities = [0.0] * TOTAL_VELOCITY_FEATURES
        wrist_delta = [0.0, 0.0, 0.0]
        kinetic_energy = 0.0

    # Combine: 63 spatial + 63 velocities + 3 trajectory = 129 features
    motion_vector = spatial_features + velocities + wrist_delta
    return motion_vector, kinetic_energy


def extract_features_from_mediapipe_landmarks(hand_landmarks) -> List[float]:
    """
    Extracts and normalizes 63 numerical features from a MediaPipe HandLandmarks object.
    """
    raw_coords = []
    for lm in hand_landmarks.landmark:
        raw_coords.append((lm.x, lm.y, lm.z))
    return normalize_landmarks(raw_coords)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


def robust_extract_landmarks_from_bgr_frame(
    bgr_frame: np.ndarray,
    primary_detector=None
) -> Tuple[Optional[List[float]], Optional[object], Optional[List[Tuple[float, float, float]]], str]:
    """
    Robust multi-stage cascading hand landmark extraction.
    Crucial for tightly-cropped images (e.g. Kaggle ASL dataset) and edge-of-frame hands:
    - Stage 1: Fast direct inference on raw frame (conf=0.40)
    - Stage 2: Reflection padding 50px (conf=0.30)
    - Stage 3: Reflection padding 80px (conf=0.25)
    - Stage 4: Replication padding 50px (conf=0.25)
    - Stage 5: Constant black padding 80px (conf=0.20)
    - Stage 6: Constant black padding 100px (conf=0.20)
    - Stage 7: CLAHE contrast enhancement (conf=0.15)
    - Stage 8: CLAHE + 60px reflection padding (conf=0.10)
    
    Inverse maps normalized coordinates back to the original image frame.
    
    Returns:
        Tuple of (features_63, hand_landmarks_proto, raw_coords_21, stage_name).
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None, None, None, "empty_frame"

    h, w = bgr_frame.shape[:2]
    import mediapipe as mp
    mp_hands = mp.solutions.hands

    # Define cascading inspection stages: (name, image, pad_t, pad_b, pad_l, pad_r, conf)
    stages = [
        ("raw", bgr_frame, 0, 0, 0, 0, 0.40),
        ("pad_reflect_50", cv2.copyMakeBorder(bgr_frame, 50, 50, 50, 50, cv2.BORDER_REFLECT), 50, 50, 50, 50, 0.30),
        ("pad_reflect_80", cv2.copyMakeBorder(bgr_frame, 80, 80, 80, 80, cv2.BORDER_REFLECT), 80, 80, 80, 80, 0.25),
        ("pad_replicate_50", cv2.copyMakeBorder(bgr_frame, 50, 50, 50, 50, cv2.BORDER_REPLICATE), 50, 50, 50, 50, 0.25),
        ("pad_const_80", cv2.copyMakeBorder(bgr_frame, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=[0, 0, 0]), 80, 80, 80, 80, 0.20),
        ("pad_const_100", cv2.copyMakeBorder(bgr_frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0]), 100, 100, 100, 100, 0.20),
    ]

    # Pre-generate CLAHE if previous stages fail
    clahe_generated = False
    clahe_img = None

    for idx, (s_name, s_img, pt, pb, pl, pr, conf) in enumerate(stages):
        sh, sw = s_img.shape[:2]
        rgb = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
        
        # Use primary_detector for raw stage if supplied
        if s_name == "raw" and primary_detector is not None:
            rgb.flags.writeable = False
            res = primary_detector.process(rgb)
            rgb.flags.writeable = True
        else:
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=conf) as det:
                res = det.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            raw_coords = []
            for lm in hand.landmark:
                orig_x = float((lm.x * sw - pl) / w)
                orig_y = float((lm.y * sh - pt) / h)
                orig_z = float(lm.z * (sw / w))
                raw_coords.append((orig_x, orig_y, orig_z))
                # Update in-place so standard drawing utils work on original image
                lm.x = orig_x
                lm.y = orig_y
                lm.z = orig_z

            norm_feats = normalize_landmarks(raw_coords)
            return norm_feats, hand, raw_coords, s_name

    # CLAHE Fallback Stages (for harsh lighting, high glare, or deep shadows)
    try:
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        clahe_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        clahe_stages = [
            ("clahe_raw", clahe_img, 0, 0, 0, 0, 0.10),
            ("clahe_pad_60", cv2.copyMakeBorder(clahe_img, 60, 60, 60, 60, cv2.BORDER_REFLECT), 60, 60, 60, 60, 0.10),
            ("clahe_pad_80", cv2.copyMakeBorder(clahe_img, 80, 80, 80, 80, cv2.BORDER_REFLECT), 80, 80, 80, 80, 0.10)
        ]
        for s_name, s_img, pt, pb, pl, pr, conf in clahe_stages:
            sh, sw = s_img.shape[:2]
            rgb = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=conf) as det:
                res = det.process(rgb)
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                raw_coords = []
                for lm in hand.landmark:
                    orig_x = float((lm.x * sw - pl) / w)
                    orig_y = float((lm.y * sh - pt) / h)
                    orig_z = float(lm.z * (sw / w))
                    raw_coords.append((orig_x, orig_y, orig_z))
                    lm.x = orig_x
                    lm.y = orig_y
                    lm.z = orig_z
                norm_feats = normalize_landmarks(raw_coords)
                return norm_feats, hand, raw_coords, s_name
    except Exception:
        pass

    return None, None, None, "not_detected"


def extract_landmarks_from_bgr_frame(
    bgr_frame: np.ndarray,
    hands_detector=None,
    robust: bool = True
) -> Tuple[Optional[List[float]], Optional[object]]:
    """
    Extracts normalized 63 landmarks from a BGR image frame (OpenCV format).
    Supports robust cascading fallback when direct detection fails.
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None, None

    if robust:
        feats, hand_proto, _, _ = robust_extract_landmarks_from_bgr_frame(bgr_frame, hands_detector)
        return feats, hand_proto

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands_detector.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.multi_hand_landmarks:
        primary_hand = results.multi_hand_landmarks[0]
        features = extract_features_from_mediapipe_landmarks(primary_hand)
        return features, primary_hand

    return None, None


def extract_motion_frame_data(
    bgr_frame: np.ndarray,
    hands_detector,
    prev_raw_coords: Optional[List[Tuple[float, float, float]]] = None
) -> Tuple[Optional[List[float]], Optional[List[Tuple[float, float, float]]], Optional[object], float]:
    """
    Extracts 129 motion-aware features from a BGR image frame.
    
    Returns:
        Tuple of (motion_features_129, current_raw_coords, hand_landmarks_proto, kinetic_energy).
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None, None, None, 0.0

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands_detector.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.multi_hand_landmarks:
        primary_hand = results.multi_hand_landmarks[0]
        curr_raw = [(lm.x, lm.y, lm.z) for lm in primary_hand.landmark]
        motion_feats, ke = extract_motion_features(curr_raw, prev_raw_coords)
        return motion_feats, curr_raw, primary_hand, ke

    return None, None, None, 0.0


def draw_styled_landmarks(
    image: np.ndarray,
    hand_landmarks,
    mp_hands=None,
    mp_drawing=None,
    mp_drawing_styles=None
) -> np.ndarray:
    """
    Draws visually appealing landmark connections on the target BGR image.
    Supports MediaPipe LandmarkList proto and raw coordinate tuples.
    """
    if hand_landmarks is None or image is None:
        return image

    h, w = image.shape[:2]

    # If it's a list of raw (x, y, z) coordinates:
    if isinstance(hand_landmarks, list):
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand_landmarks]
        # Draw skeleton connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if 0 <= start_idx < len(pts) and 0 <= end_idx < len(pts):
                cv2.line(image, pts[start_idx], pts[end_idx], (0, 255, 128), 2, cv2.LINE_AA)
        # Draw landmark nodes
        for px, py in pts:
            cv2.circle(image, (px, py), 4, (0, 180, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
        return image

    # If it's a MediaPipe proto
    import mediapipe as mp
    mp_hands = mp_hands or mp.solutions.hands
    mp_drawing = mp_drawing or mp.solutions.drawing_utils
    mp_drawing_styles = mp_drawing_styles or mp.solutions.drawing_styles

    try:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    except Exception:
        # Fallback to OpenCV drawing if proto drawing fails
        try:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            for start_idx, end_idx in HAND_CONNECTIONS:
                if 0 <= start_idx < len(pts) and 0 <= end_idx < len(pts):
                    cv2.line(image, pts[start_idx], pts[end_idx], (0, 255, 128), 2, cv2.LINE_AA)
            for px, py in pts:
                cv2.circle(image, (px, py), 4, (0, 180, 255), -1, cv2.LINE_AA)
        except Exception:
            pass

    return image


@dataclass
class DualHandDetectionResult:
    """Encapsulates synchronized detection of Left & Right hands."""
    has_hands: bool
    num_hands: int
    left_hand_proto: Optional[object]
    right_hand_proto: Optional[object]
    left_coords: Optional[List[Tuple[float, float, float]]]
    right_coords: Optional[List[Tuple[float, float, float]]]
    left_normalized_63: Optional[List[float]]
    right_normalized_63: Optional[List[float]]
    wrist_dist: float = 0.0
    index_dist: float = 0.0
    thumb_dist: float = 0.0
    palms_touching: bool = False
    is_bimanual: bool = False


def extract_dual_hands_from_bgr_frame(
    bgr_frame: np.ndarray,
    hands_detector
) -> DualHandDetectionResult:
    """
    Extracts up to 2 hands (Left & Right) with inter-hand relational geometry.
    If 1 hand is present, identifies left/right and leaves the other as None.
    If 2 hands are present, computes wrist-to-wrist and fingertip distances.
    """
    if bgr_frame is None or bgr_frame.size == 0 or hands_detector is None:
        return DualHandDetectionResult(
            has_hands=False, num_hands=0,
            left_hand_proto=None, right_hand_proto=None,
            left_coords=None, right_coords=None,
            left_normalized_63=None, right_normalized_63=None
        )

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands_detector.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if not results.multi_hand_landmarks:
        return DualHandDetectionResult(
            has_hands=False, num_hands=0,
            left_hand_proto=None, right_hand_proto=None,
            left_coords=None, right_coords=None,
            left_normalized_63=None, right_normalized_63=None
        )

    num_hands = len(results.multi_hand_landmarks)
    left_proto = None
    right_proto = None
    left_coords = None
    right_coords = None

    if num_hands == 1:
        hand = results.multi_hand_landmarks[0]
        label = "Right"
        if results.multi_handedness and len(results.multi_handedness) > 0:
            label = results.multi_handedness[0].classification[0].label

        coords = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
        if label == "Left":
            left_proto = hand
            left_coords = coords
        else:
            right_proto = hand
            right_coords = coords
    else:
        # 2 hands detected - classify by handedness or horizontal position fallback
        hand_0 = results.multi_hand_landmarks[0]
        hand_1 = results.multi_hand_landmarks[1]
        coords_0 = [(lm.x, lm.y, lm.z) for lm in hand_0.landmark]
        coords_1 = [(lm.x, lm.y, lm.z) for lm in hand_1.landmark]

        label_0 = "Right"
        label_1 = "Left"
        if results.multi_handedness and len(results.multi_handedness) >= 2:
            label_0 = results.multi_handedness[0].classification[0].label
            label_1 = results.multi_handedness[1].classification[0].label

        if label_0 != label_1:
            if label_0 == "Left":
                left_proto, left_coords = hand_0, coords_0
                right_proto, right_coords = hand_1, coords_1
            else:
                right_proto, right_coords = hand_0, coords_0
                left_proto, left_coords = hand_1, coords_1
        else:
            # Fallback: Leftmost hand on screen is Left
            if coords_0[0][0] < coords_1[0][0]:
                left_proto, left_coords = hand_0, coords_0
                right_proto, right_coords = hand_1, coords_1
            else:
                left_proto, left_coords = hand_1, coords_1
                right_proto, right_coords = hand_0, coords_0

    # Compute normalized spatial coordinates (63 each)
    left_norm = normalize_landmarks(left_coords) if left_coords else None
    right_norm = normalize_landmarks(right_coords) if right_coords else None

    # Compute inter-hand relational features if both hands present
    wrist_dist = 0.0
    index_dist = 0.0
    thumb_dist = 0.0
    palms_touching = False
    is_bimanual = (left_coords is not None and right_coords is not None)

    if is_bimanual:
        wl = np.array(left_coords[0])
        wr = np.array(right_coords[0])
        il = np.array(left_coords[8])
        ir = np.array(right_coords[8])
        tl = np.array(left_coords[4])
        tr = np.array(right_coords[4])

        wrist_dist = float(np.linalg.norm(wl - wr))
        index_dist = float(np.linalg.norm(il - ir))
        thumb_dist = float(np.linalg.norm(tl - tr))
        palms_touching = bool(wrist_dist < 0.20 and index_dist < 0.20)

    return DualHandDetectionResult(
        has_hands=True,
        num_hands=num_hands,
        left_hand_proto=left_proto,
        right_hand_proto=right_proto,
        left_coords=left_coords,
        right_coords=right_coords,
        left_normalized_63=left_norm,
        right_normalized_63=right_norm,
        wrist_dist=wrist_dist,
        index_dist=index_dist,
        thumb_dist=thumb_dist,
        palms_touching=palms_touching,
        is_bimanual=is_bimanual
    )


def draw_dual_styled_landmarks(
    image: np.ndarray,
    dual_result: DualHandDetectionResult
) -> np.ndarray:
    """
    Renders Left hand (Cyan/Blue) and Right hand (Green/Orange) landmarks,
    with an inter-hand distance guide when both hands are visible.
    """
    if not dual_result.has_hands or image is None:
        return image

    h, w = image.shape[:2]

    # 1. Draw Left Hand (Cyan nodes, Blue skeleton)
    if dual_result.left_coords:
        pts_l = [(int(x * w), int(y * h)) for x, y, _ in dual_result.left_coords]
        for start_idx, end_idx in HAND_CONNECTIONS:
            if 0 <= start_idx < len(pts_l) and 0 <= end_idx < len(pts_l):
                cv2.line(image, pts_l[start_idx], pts_l[end_idx], (230, 180, 50), 2, cv2.LINE_AA)
        for px, py in pts_l:
            cv2.circle(image, (px, py), 4, (255, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(image, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
        # Left hand label
        cv2.putText(image, "L", (pts_l[0][0] - 15, pts_l[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

    # 2. Draw Right Hand (Green nodes, Emerald skeleton)
    if dual_result.right_coords:
        pts_r = [(int(x * w), int(y * h)) for x, y, _ in dual_result.right_coords]
        for start_idx, end_idx in HAND_CONNECTIONS:
            if 0 <= start_idx < len(pts_r) and 0 <= end_idx < len(pts_r):
                cv2.line(image, pts_r[start_idx], pts_r[end_idx], (50, 220, 120), 2, cv2.LINE_AA)
        for px, py in pts_r:
            cv2.circle(image, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(image, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
        # Right hand label
        cv2.putText(image, "R", (pts_r[0][0] + 10, pts_r[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

    # 3. If Bimanual (Both hands present): draw inter-hand relationship line between wrists
    if dual_result.is_bimanual and dual_result.left_coords and dual_result.right_coords:
        wl_pt = (int(dual_result.left_coords[0][0] * w), int(dual_result.left_coords[0][1] * h))
        wr_pt = (int(dual_result.right_coords[0][0] * w), int(dual_result.right_coords[0][1] * h))
        cv2.line(image, wl_pt, wr_pt, (200, 200, 255), 1, cv2.LINE_AA)
        mid_x = (wl_pt[0] + wr_pt[0]) // 2
        mid_y = (wl_pt[1] + wr_pt[1]) // 2
        cv2.putText(image, f"2-HANDS (d={dual_result.wrist_dist:.2f})", (mid_x - 55, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 255), 1, cv2.LINE_AA)

    return image


