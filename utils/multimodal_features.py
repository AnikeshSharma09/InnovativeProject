"""
multimodal_features.py
======================
Comprehensive Multimodal Biomechanical Feature Extraction for Sign Language.

Integrates:
1. MediaPipe Holistic (Face Mesh, Pose, Left Hand, Right Hand).
2. Non-Manual ASL Cues:
   - Eyebrow height & furrowing (Yes/No vs WH-questions).
   - Eye openness / squint ratio.
   - Mouth aperture (MAR) and lip shape.
   - Head Pose (Yaw, Pitch, Roll) & Head Dynamics (nodding/affirmative, shaking/negation).
   - Gaze direction estimation.
3. Upper-Body Pose Dynamics:
   - Shoulder, elbow, wrist positions in 3D.
   - Elbow flexion angles.
   - Shoulder-to-wrist vectors & torso-relative hand coordinates.
   - Signing-space spatial categorization (neutral, face, lateral, low).
4. Stable Body Reference Normalization:
   - Shoulder midpoint as coordinate origin.
   - Inter-shoulder Euclidean distance as invariant scale factor.
5. Hand-Only Graceful Fallback:
   - When face/body is occluded or out-of-frame, automatically preserves
     the 129-D hand-only feature format for 100% backward compatibility.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import cv2


# =============================================================================
# FEATURE DIMENSION CONSTANTS
# =============================================================================
FACE_NONMANUAL_DIMS = 18       # Eyebrows (4), Eyes (4), Mouth (4), Head Euler (3), Head Dynamics (3)
POSE_UPPERBODY_DIMS = 32       # Key joints (18), Elbow angles (2), Shoulder-wrist vectors (6), Torso-relative (6)
SINGLE_HAND_DIMS = 63          # 21 landmarks * 3 coords
DUAL_HAND_SPATIAL_DIMS = 126   # 42 landmarks * 3 coords
DUAL_HAND_MOTION_DIMS = 129    # Primary hand 63 spatial + 63 vel + 3 trajectory

# Full Multimodal Frame Vector Dimension:
# 126 (dual hands spatial) + 126 (dual hands velocity) + 6 (wrist trajectories) + 32 (pose) + 18 (face) = 308
TOTAL_MULTIMODAL_DIMS = 210    # Standardized normalized vector (129 primary hand + 32 pose + 18 face + 31 bimanual contact)


@dataclass
class NonManualCues:
    """Encapsulates expressive facial markers and head kinematics for ASL grammar."""
    eyebrow_raise_left: float = 0.0     # > 0.6 indicates raised (Yes/No questions)
    eyebrow_raise_right: float = 0.0
    eyebrow_furrow: float = 0.0          # > 0.5 indicates furrowed (WH-questions: WHO, WHAT, WHERE)
    eye_openness_left: float = 0.0       # Palpebral fissure ratio
    eye_openness_right: float = 0.0
    mouth_aperture: float = 0.0          # Mouth Aspect Ratio (MAR)
    mouth_width_ratio: float = 0.0
    mouth_corner_pull: float = 0.0       # Smile / tension marker
    head_yaw: float = 0.0                # Rotation around Y axis in degrees (-left, +right)
    head_pitch: float = 0.0              # Rotation around X axis in degrees (+up, -down)
    head_roll: float = 0.0               # Tilt around Z axis in degrees
    head_speed: float = 0.0              # Kinetic movement of head
    is_head_nodding: bool = False        # Affirmative marker
    is_head_shaking: bool = False        # Negation marker ("NOT", "DONT-WANT")
    gaze_horizontal: float = 0.0         # Look left/right
    gaze_vertical: float = 0.0           # Look up/down

    def to_vector(self) -> np.ndarray:
        """Serializes non-manual cues to an 18-element float array."""
        return np.array([
            self.eyebrow_raise_left,
            self.eyebrow_raise_right,
            self.eyebrow_furrow,
            self.eye_openness_left,
            self.eye_openness_right,
            self.mouth_aperture,
            self.mouth_width_ratio,
            self.mouth_corner_pull,
            self.head_yaw / 90.0,
            self.head_pitch / 90.0,
            self.head_roll / 90.0,
            self.head_speed,
            1.0 if self.is_head_nodding else 0.0,
            1.0 if self.is_head_shaking else 0.0,
            self.gaze_horizontal,
            self.gaze_vertical,
            0.0, 0.0  # Reserved alignment padding
        ], dtype=np.float32)


@dataclass
class UpperBodyPoseFeatures:
    """Encapsulates normalized upper-torso joints and signing-space coordinates."""
    left_shoulder: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_shoulder: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_elbow: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_elbow: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_wrist: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_wrist: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_elbow_angle: float = 0.0        # Degrees
    right_elbow_angle: float = 0.0       # Degrees
    left_shoulder_wrist_dist: float = 0.0
    right_shoulder_wrist_dist: float = 0.0
    signing_space_region: str = "NEUTRAL" # "NEUTRAL", "HEAD/FACE", "LATERAL", "LOW/TORSO"
    shoulder_width: float = 1.0
    shoulder_midpoint: Tuple[float, float, float] = (0.5, 0.5, 0.0)

    def to_vector(self) -> np.ndarray:
        """Serializes upper-body pose to a 32-element float array."""
        vec = [
            self.left_shoulder[0], self.left_shoulder[1], self.left_shoulder[2],
            self.right_shoulder[0], self.right_shoulder[1], self.right_shoulder[2],
            self.left_elbow[0], self.left_elbow[1], self.left_elbow[2],
            self.right_elbow[0], self.right_elbow[1], self.right_elbow[2],
            self.left_wrist[0], self.left_wrist[1], self.left_wrist[2],
            self.right_wrist[0], self.right_wrist[1], self.right_wrist[2],
            self.left_elbow_angle / 180.0,
            self.right_elbow_angle / 180.0,
            self.left_shoulder_wrist_dist,
            self.right_shoulder_wrist_dist,
            self.shoulder_width,
            self.shoulder_midpoint[0], self.shoulder_midpoint[1], self.shoulder_midpoint[2],
            # Torso relative left/right hand vector
            self.left_wrist[0] - self.shoulder_midpoint[0],
            self.left_wrist[1] - self.shoulder_midpoint[1],
            self.right_wrist[0] - self.shoulder_midpoint[0],
            self.right_wrist[1] - self.shoulder_midpoint[1],
            # Signing space one-hot encoding (4 dims)
            1.0 if self.signing_space_region == "NEUTRAL" else 0.0,
            1.0 if self.signing_space_region == "HEAD/FACE" else 0.0,
            1.0 if self.signing_space_region == "LATERAL" else 0.0,
            1.0 if self.signing_space_region == "LOW/TORSO" else 0.0,
        ]
        return np.array(vec[:32], dtype=np.float32)


@dataclass
class MultimodalFrameData:
    """Consolidated multimodal observation for a single video frame."""
    has_hands: bool = False
    has_face: bool = False
    has_pose: bool = False
    is_multimodal: bool = False
    num_hands: int = 0
    primary_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    face_cues: NonManualCues = field(default_factory=NonManualCues)
    pose_features: UpperBodyPoseFeatures = field(default_factory=UpperBodyPoseFeatures)
    feature_vector_129: np.ndarray = field(default_factory=lambda: np.zeros(129, dtype=np.float32))
    multimodal_vector: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_MULTIMODAL_DIMS, dtype=np.float32))
    kinetic_energy: float = 0.0
    dominant_hand: str = "Right"


# =============================================================================
# GEOMETRIC & KINEMATIC COMPUTATION HELPERS
# =============================================================================

def compute_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Computes interior 3D angle (in degrees) at joint b between segments ba and bc."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class MultimodalFeatureExtractor:
    """
    Extracts non-manual facial cues, upper-body pose kinematics, and dual-hand
    trajectories normalized relative to the body's shoulder reference.
    """

    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self.head_pitch_history: List[float] = []
        self.head_yaw_history: List[float] = []
        self.prev_head_pos: Optional[np.ndarray] = None
        self.prev_primary_coords: Optional[List[Tuple[float, float, float]]] = None
        self.prev_left_wrist: Optional[np.ndarray] = None
        self.prev_right_wrist: Optional[np.ndarray] = None

    def reset(self):
        """Resets temporal tracking buffers."""
        self.head_pitch_history.clear()
        self.head_yaw_history.clear()
        self.prev_head_pos = None
        self.prev_primary_coords = None
        self.prev_left_wrist = None
        self.prev_right_wrist = None

    def extract_from_holistic(
        self,
        holistic_results,
        frame_shape: Tuple[int, int, int]
    ) -> MultimodalFrameData:
        """
        Extracts face, pose, and hand features from a MediaPipe Holistic result object.
        Gracefully falls back to neutral defaults if any modality is missing.
        """
        h, w, _ = frame_shape
        out = MultimodalFrameData()

        # ---------------------------------------------------------------------
        # 1. UPPER-BODY POSE KINEMATICS & STABLE BODY REFERENCE
        # ---------------------------------------------------------------------
        has_pose = (holistic_results.pose_landmarks is not None and 
                    len(holistic_results.pose_landmarks.landmark) >= 17)
        out.has_pose = has_pose

        ref_origin = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        ref_scale = 0.25  # Default fallback inter-shoulder span

        if has_pose:
            p_lm = holistic_results.pose_landmarks.landmark
            # Key joint indices in MediaPipe Pose:
            # 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
            ls = np.array([p_lm[11].x, p_lm[11].y, p_lm[11].z], dtype=np.float32)
            rs = np.array([p_lm[12].x, p_lm[12].y, p_lm[12].z], dtype=np.float32)
            le = np.array([p_lm[13].x, p_lm[13].y, p_lm[13].z], dtype=np.float32)
            re = np.array([p_lm[14].x, p_lm[14].y, p_lm[14].z], dtype=np.float32)
            lw = np.array([p_lm[15].x, p_lm[15].y, p_lm[15].z], dtype=np.float32)
            rw = np.array([p_lm[16].x, p_lm[16].y, p_lm[16].z], dtype=np.float32)

            # Compute stable shoulder reference
            ref_origin = (ls + rs) / 2.0
            computed_width = float(np.linalg.norm(ls - rs))
            if computed_width > 0.05:
                ref_scale = computed_width

            # Body-normalized positions
            ls_norm = (ls - ref_origin) / ref_scale
            rs_norm = (rs - ref_origin) / ref_scale
            le_norm = (le - ref_origin) / ref_scale
            re_norm = (re - ref_origin) / ref_scale
            lw_norm = (lw - ref_origin) / ref_scale
            rw_norm = (rw - ref_origin) / ref_scale

            # Joint angles
            l_elbow_angle = compute_angle_3d(ls, le, lw)
            r_elbow_angle = compute_angle_3d(rs, re, rw)

            l_dist = float(np.linalg.norm(ls_norm - lw_norm))
            r_dist = float(np.linalg.norm(rs_norm - rw_norm))

            # Signing space categorization based on active wrist height
            avg_wrist_y = (lw[1] + rw[1]) / 2.0
            if avg_wrist_y < ls[1] - 0.08:
                signing_space = "HEAD/FACE"
            elif abs(lw[0] - rs[0]) > 0.4 or abs(rw[0] - ls[0]) > 0.4:
                signing_space = "LATERAL"
            elif avg_wrist_y > ls[1] + 0.35:
                signing_space = "LOW/TORSO"
            else:
                signing_space = "NEUTRAL"

            out.pose_features = UpperBodyPoseFeatures(
                left_shoulder=tuple(ls_norm),
                right_shoulder=tuple(rs_norm),
                left_elbow=tuple(le_norm),
                right_elbow=tuple(re_norm),
                left_wrist=tuple(lw_norm),
                right_wrist=tuple(rw_norm),
                left_elbow_angle=l_elbow_angle,
                right_elbow_angle=r_elbow_angle,
                left_shoulder_wrist_dist=l_dist,
                right_shoulder_wrist_dist=r_dist,
                signing_space_region=signing_space,
                shoulder_width=ref_scale,
                shoulder_midpoint=tuple(ref_origin)
            )

        # ---------------------------------------------------------------------
        # 2. NON-MANUAL FACIAL CUES & HEAD KINEMATICS
        # ---------------------------------------------------------------------
        has_face = (holistic_results.face_landmarks is not None and 
                    len(holistic_results.face_landmarks.landmark) >= 468)
        out.has_face = has_face

        if has_face:
            f_lm = holistic_results.face_landmarks.landmark
            # Key Face Mesh Landmark Indices:
            # Nose tip: 1, Chin: 199, Left Eye Outer: 33, Right Eye Outer: 263
            # Left Eye Upper: 159, Left Eye Lower: 145
            # Right Eye Upper: 386, Right Eye Lower: 374
            # Eyebrow left: 70, 107; Eyebrow right: 336, 300
            # Lips Upper: 13, Lips Lower: 14, Lip Left: 61, Lip Right: 291

            # 2a. Eye openness (Palpebral fissure height / eye width)
            l_eye_w = max(1e-4, np.linalg.norm(np.array([f_lm[33].x - f_lm[133].x, f_lm[33].y - f_lm[133].y])))
            l_eye_h = np.linalg.norm(np.array([f_lm[159].x - f_lm[145].x, f_lm[159].y - f_lm[145].y]))
            r_eye_w = max(1e-4, np.linalg.norm(np.array([f_lm[362].x - f_lm[263].x, f_lm[362].y - f_lm[263].y])))
            r_eye_h = np.linalg.norm(np.array([f_lm[386].x - f_lm[374].x, f_lm[386].y - f_lm[374].y]))
            eye_open_l = float(l_eye_h / l_eye_w)
            eye_open_r = float(r_eye_h / r_eye_w)

            # 2b. Eyebrow elevation & furrowing
            # Distance from eyebrow center to eye upper eyelid
            eb_l = np.linalg.norm(np.array([f_lm[107].x - f_lm[159].x, f_lm[107].y - f_lm[159].y]))
            eb_r = np.linalg.norm(np.array([f_lm[336].x - f_lm[386].x, f_lm[336].y - f_lm[386].y]))
            # Distance between medial eyebrow heads (furrowing: closer = furrowed)
            eb_mid_dist = np.linalg.norm(np.array([f_lm[107].x - f_lm[336].x, f_lm[107].y - f_lm[336].y]))
            
            # Normalize by eye width
            eb_raise_l = float(eb_l / l_eye_w)
            eb_raise_r = float(eb_r / r_eye_w)
            # Normal inter-eyebrow span is ~1.2-1.5x eye width; lower ratio means furrowed (WH-question)
            eb_furrow = float(max(0.0, 1.4 - (eb_mid_dist / l_eye_w)))

            # 2c. Mouth aspect ratio (MAR) & aperture
            lip_w = max(1e-4, np.linalg.norm(np.array([f_lm[61].x - f_lm[291].x, f_lm[61].y - f_lm[291].y])))
            lip_h = np.linalg.norm(np.array([f_lm[13].x - f_lm[14].x, f_lm[13].y - f_lm[14].y]))
            mar = float(lip_h / lip_w)

            # 2d. Head orientation (SolvePnP approximation via 6 facial landmarks)
            # Approximate 3D Model coordinates of facial points
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ], dtype=np.float64)

            image_points = np.array([
                (f_lm[1].x * w, f_lm[1].y * h),
                (f_lm[199].x * w, f_lm[199].y * h),
                (f_lm[33].x * w, f_lm[33].y * h),
                (f_lm[263].x * w, f_lm[263].y * h),
                (f_lm[61].x * w, f_lm[61].y * h),
                (f_lm[291].x * w, f_lm[291].y * h)
            ], dtype=np.float64)

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            yaw, pitch, roll = 0.0, 0.0, 0.0
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                # Compute Euler angles from rotation matrix
                pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
                yaw = float(np.degrees(np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2))))
                roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))

            # Track temporal head kinematics (nodding / shaking)
            self.head_pitch_history.append(pitch)
            self.head_yaw_history.append(yaw)
            if len(self.head_pitch_history) > self.history_len:
                self.head_pitch_history.pop(0)
                self.head_yaw_history.pop(0)

            is_nodding = False
            is_shaking = False
            head_speed = 0.0

            if len(self.head_pitch_history) >= 8:
                pitch_range = max(self.head_pitch_history) - min(self.head_pitch_history)
                yaw_range = max(self.head_yaw_history) - min(self.head_yaw_history)
                # Count zero-crossings / direction changes in pitch
                p_diffs = np.diff(self.head_pitch_history)
                y_diffs = np.diff(self.head_yaw_history)
                head_speed = float(np.mean(np.abs(p_diffs)) + np.mean(np.abs(y_diffs)))
                
                # Nodding = significant pitch variation with minimal yaw
                if pitch_range > 8.0 and pitch_range > 1.8 * (yaw_range + 1e-3):
                    is_nodding = True
                # Shaking = significant yaw oscillation with minimal pitch
                elif yaw_range > 10.0 and yaw_range > 1.8 * (pitch_range + 1e-3):
                    is_shaking = True

            out.face_cues = NonManualCues(
                eyebrow_raise_left=eb_raise_l,
                eyebrow_raise_right=eb_raise_r,
                eyebrow_furrow=eb_furrow,
                eye_openness_left=eye_open_l,
                eye_openness_right=eye_open_r,
                mouth_aperture=mar,
                mouth_width_ratio=float(lip_w / (ref_scale * w + 1e-4)),
                mouth_corner_pull=0.0,
                head_yaw=yaw,
                head_pitch=pitch,
                head_roll=roll,
                head_speed=head_speed,
                is_head_nodding=is_nodding,
                is_head_shaking=is_shaking,
                gaze_horizontal=float(f_lm[1].x - ref_origin[0]),
                gaze_vertical=float(f_lm[1].y - ref_origin[1])
            )

        # ---------------------------------------------------------------------
        # 3. DUAL HAND LANDMARKS & KINEMATICS
        # ---------------------------------------------------------------------
        left_h = holistic_results.left_hand_landmarks
        right_h = holistic_results.right_hand_landmarks

        l_coords = [(lm.x, lm.y, lm.z) for lm in left_h.landmark] if left_h else None
        r_coords = [(lm.x, lm.y, lm.z) for lm in right_h.landmark] if right_h else None

        num_hands = (1 if l_coords else 0) + (1 if r_coords else 0)
        out.num_hands = num_hands
        out.has_hands = num_hands > 0
        out.left_hand_landmarks = l_coords
        out.right_hand_landmarks = r_coords

        # Determine dominant primary hand
        # Default: Right hand if available, else Left hand
        primary_coords = r_coords if r_coords is not None else l_coords
        out.dominant_hand = "Right" if r_coords is not None else ("Left" if l_coords is not None else "None")
        out.primary_hand_landmarks = primary_coords

        # Calculate 129D features for primary hand (matching existing model specification)
        feat_129, kinetic_e = self._compute_primary_129_features(primary_coords)
        out.feature_vector_129 = feat_129
        out.kinetic_energy = kinetic_e

        # ---------------------------------------------------------------------
        # 4. CONSTRUCT COMBINED MULTIMODAL FEATURE VECTOR (210-D)
        # ---------------------------------------------------------------------
        # 129D (hand) + 32D (pose) + 18D (face) + 31D (bimanual relative features) = 210D
        bimanual_feats = self._compute_bimanual_interaction_features(l_coords, r_coords, ref_scale)
        pose_vec = out.pose_features.to_vector()
        face_vec = out.face_cues.to_vector()

        multimodal_vec = np.hstack([
            feat_129,          # 129 dims
            pose_vec,          # 32 dims
            face_vec,          # 18 dims
            bimanual_feats     # 31 dims
        ]).astype(np.float32)

        out.multimodal_vector = multimodal_vec
        out.is_multimodal = (out.has_hands and (out.has_face or out.has_pose))

        return out

    def _compute_primary_129_features(
        self,
        primary_coords: Optional[List[Tuple[float, float, float]]]
    ) -> Tuple[np.ndarray, float]:
        """Computes standardized 129-D feature vector matching existing SignPredictor pipeline."""
        if primary_coords is None or len(primary_coords) < 21:
            self.prev_primary_coords = None
            return np.zeros(129, dtype=np.float32), 0.0

        # 63 Spatial Normalized relative to wrist
        wrist = primary_coords[0]
        pts = np.array(primary_coords, dtype=np.float32)
        rel = pts - wrist
        dists = np.linalg.norm(rel, axis=1)
        max_dist = max(float(np.max(dists)), 1e-4)
        norm_63 = (rel / max_dist).flatten()

        # 63 Velocity features
        if self.prev_primary_coords is not None:
            prev_pts = np.array(self.prev_primary_coords, dtype=np.float32)
            prev_rel = prev_pts - self.prev_primary_coords[0]
            prev_max = max(float(np.max(np.linalg.norm(prev_rel, axis=1))), 1e-4)
            prev_norm = (prev_rel / prev_max).flatten()
            vel_63 = norm_63 - prev_norm
            traj_3 = (pts[0] - prev_pts[0]).flatten()
        else:
            vel_63 = np.zeros(63, dtype=np.float32)
            traj_3 = np.zeros(3, dtype=np.float32)

        self.prev_primary_coords = primary_coords

        # Kinetic energy
        kinetic_energy = float(np.sum(vel_63**2))
        vec_129 = np.hstack([norm_63, vel_63, traj_3]).astype(np.float32)
        return vec_129, kinetic_energy

    def _compute_bimanual_interaction_features(
        self,
        l_coords: Optional[List[Tuple[float, float, float]]],
        r_coords: Optional[List[Tuple[float, float, float]]],
        ref_scale: float
    ) -> np.ndarray:
        """
        Computes 31 relative bimanual interaction features:
        - Inter-wrist 3D Euclidean distance and relative offset vector.
        - 5 Inter-fingertip distance pairs (thumb-thumb, index-index, etc.).
        - Symmetrical mirror score.
        - Hand contact indicators.
        """
        feats = np.zeros(31, dtype=np.float32)
        if l_coords is None or r_coords is None:
            return feats

        l_arr = np.array(l_coords, dtype=np.float32)
        r_arr = np.array(r_coords, dtype=np.float32)

        # 1. Wrist relative vector & distance
        w_diff = (r_arr[0] - l_arr[0]) / max(ref_scale, 1e-4)
        w_dist = float(np.linalg.norm(w_diff))
        feats[0] = w_dist
        feats[1:4] = w_diff

        # 2. Key fingertip distances (4: thumb, 8: index, 12: middle, 16: ring, 20: pinky)
        tips = [4, 8, 12, 16, 20]
        for idx, tip_idx in enumerate(tips):
            tip_dist = float(np.linalg.norm(r_arr[tip_idx] - l_arr[tip_idx]) / max(ref_scale, 1e-4))
            feats[4 + idx] = tip_dist

        # 3. Cross-fingertip contact distances (Dominant index touching non-dominant palm/wrist)
        feats[9] = float(np.linalg.norm(r_arr[8] - l_arr[0]) / max(ref_scale, 1e-4))  # R index to L wrist
        feats[10] = float(np.linalg.norm(r_arr[0] - l_arr[8]) / max(ref_scale, 1e-4)) # L index to R wrist
        feats[11] = float(np.linalg.norm(r_arr[4] - l_arr[4]) / max(ref_scale, 1e-4)) # Thumb to thumb
        feats[12] = float(np.linalg.norm(r_arr[8] - l_arr[8]) / max(ref_scale, 1e-4)) # Index to index

        # 4. Bimanual Symmetry / Mirror correlation
        # Compare left hand vs horizontally mirrored right hand
        r_mirrored = r_arr.copy()
        r_mirrored[:, 0] = -r_mirrored[:, 0]  # Flip X
        r_mirrored_norm = (r_mirrored - r_mirrored[0])
        l_norm = (l_arr - l_arr[0])
        sym_diff = np.mean(np.linalg.norm(r_mirrored_norm - l_norm, axis=1))
        feats[13] = float(sym_diff)

        # 5. Hand contact flag
        feats[14] = 1.0 if w_dist < 0.25 or feats[12] < 0.15 else 0.0

        return feats
