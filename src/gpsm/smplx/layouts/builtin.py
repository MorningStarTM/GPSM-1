"""Built-in correspondence layouts for common keypoint/skeleton formats.

All correspondences here are `JointCorr` (point -> nearest SMPL-X joint
center). For formats whose landmarks sit on the body *surface* rather than
at true joint centers — MediaPipe BlazePose in particular — this is a
documented simplification (see doc/Mocap_Unification_Design.docx section
4.2): those points are given a reduced structural `weight` so the solver
trusts them less than a true joint-center correspondence (e.g. OpenPose's
auxiliary joints, which SMPL-X's own landmark regressors are built to
match exactly). Upgrading MediaPipe to true vertex-surface correspondences
is future work, flagged at the bottom of this file.
"""
from __future__ import annotations

from src.gpsm.smplx.layouts.correspondence import JointCorr, MotionLayout, VertexCorr

# ---------------------------------------------------------------------------
# COCO-17 (standard pose-estimation keypoint order)
# ---------------------------------------------------------------------------
COCO_17_POINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO_17_TO_SMPLX = [
    JointCorr("nose"), JointCorr("left_eye"), JointCorr("right_eye"),
    JointCorr("left_ear"), JointCorr("right_ear"),
    JointCorr("left_shoulder"), JointCorr("right_shoulder"),
    JointCorr("left_elbow"), JointCorr("right_elbow"),
    JointCorr("left_wrist"), JointCorr("right_wrist"),
    JointCorr("left_hip"), JointCorr("right_hip"),
    JointCorr("left_knee"), JointCorr("right_knee"),
    JointCorr("left_ankle"), JointCorr("right_ankle"),
]
COCO_17 = MotionLayout(
    name="coco_17",
    point_names=COCO_17_POINT_NAMES,
    correspondences=COCO_17_TO_SMPLX,
    description="Standard 17-keypoint COCO pose format (e.g. HRNet, most 2D detectors).",
)

# ---------------------------------------------------------------------------
# OpenPose BODY_25
# ---------------------------------------------------------------------------
OPENPOSE_25_POINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
]
OPENPOSE_25_TO_SMPLX = [
    JointCorr("nose"), JointCorr("neck"),
    JointCorr("right_shoulder"), JointCorr("right_elbow"), JointCorr("right_wrist"),
    JointCorr("left_shoulder"), JointCorr("left_elbow"), JointCorr("left_wrist"),
    JointCorr("pelvis"),
    JointCorr("right_hip"), JointCorr("right_knee"), JointCorr("right_ankle"),
    JointCorr("left_hip"), JointCorr("left_knee"), JointCorr("left_ankle"),
    JointCorr("right_eye"), JointCorr("left_eye"), JointCorr("right_ear"), JointCorr("left_ear"),
    JointCorr("left_big_toe"), JointCorr("left_small_toe"), JointCorr("left_heel"),
    JointCorr("right_big_toe"), JointCorr("right_small_toe"), JointCorr("right_heel"),
]
OPENPOSE_25 = MotionLayout(
    name="openpose_25",
    point_names=OPENPOSE_25_POINT_NAMES,
    correspondences=OPENPOSE_25_TO_SMPLX,
    description="OpenPose BODY_25 output format.",
)

# ---------------------------------------------------------------------------
# MediaPipe BlazePose (33 landmarks) — surface points, reduced weight.
# ---------------------------------------------------------------------------
MEDIAPIPE_33_POINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
_SURFACE_WEIGHT = 0.5  # BlazePose landmarks approximate, not exact, joint centers
MEDIAPIPE_33_TO_SMPLX = [
    JointCorr("nose", _SURFACE_WEIGHT),
    JointCorr("left_eye", _SURFACE_WEIGHT), JointCorr("left_eye", _SURFACE_WEIGHT), JointCorr("left_eye", _SURFACE_WEIGHT),
    JointCorr("right_eye", _SURFACE_WEIGHT), JointCorr("right_eye", _SURFACE_WEIGHT), JointCorr("right_eye", _SURFACE_WEIGHT),
    JointCorr("left_ear", _SURFACE_WEIGHT), JointCorr("right_ear", _SURFACE_WEIGHT),
    JointCorr("nose", _SURFACE_WEIGHT * 0.5), JointCorr("nose", _SURFACE_WEIGHT * 0.5),  # mouth corners: coarse jaw-area proxy
    JointCorr("left_shoulder", _SURFACE_WEIGHT), JointCorr("right_shoulder", _SURFACE_WEIGHT),
    JointCorr("left_elbow", _SURFACE_WEIGHT), JointCorr("right_elbow", _SURFACE_WEIGHT),
    JointCorr("left_wrist", _SURFACE_WEIGHT), JointCorr("right_wrist", _SURFACE_WEIGHT),
    JointCorr("left_pinky", _SURFACE_WEIGHT), JointCorr("right_pinky", _SURFACE_WEIGHT),
    JointCorr("left_index", _SURFACE_WEIGHT), JointCorr("right_index", _SURFACE_WEIGHT),
    JointCorr("left_thumb", _SURFACE_WEIGHT), JointCorr("right_thumb", _SURFACE_WEIGHT),
    JointCorr("left_hip", _SURFACE_WEIGHT), JointCorr("right_hip", _SURFACE_WEIGHT),
    JointCorr("left_knee", _SURFACE_WEIGHT), JointCorr("right_knee", _SURFACE_WEIGHT),
    JointCorr("left_ankle", _SURFACE_WEIGHT), JointCorr("right_ankle", _SURFACE_WEIGHT),
    JointCorr("left_heel", _SURFACE_WEIGHT), JointCorr("right_heel", _SURFACE_WEIGHT),
    JointCorr("left_big_toe", _SURFACE_WEIGHT), JointCorr("right_big_toe", _SURFACE_WEIGHT),
]
MEDIAPIPE_33 = MotionLayout(
    name="mediapipe_33",
    point_names=MEDIAPIPE_33_POINT_NAMES,
    correspondences=MEDIAPIPE_33_TO_SMPLX,
    description=(
        "MediaPipe BlazePose 33-landmark output. Landmarks are surface "
        "points, not true joint centers, so all correspondences carry a "
        "reduced structural weight; treat fits from this layout as "
        "DEGRADED-tier by default (see quality.py)."
    ),
)

# ---------------------------------------------------------------------------
# Human3.6M 17-joint order
# ---------------------------------------------------------------------------
H36M_17_POINT_NAMES = [
    "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck/Nose", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]
H36M_17_TO_SMPLX = [
    JointCorr("pelvis"),
    JointCorr("right_hip"), JointCorr("right_knee"), JointCorr("right_ankle"),
    JointCorr("left_hip"), JointCorr("left_knee"), JointCorr("left_ankle"),
    JointCorr("spine2"), JointCorr("neck"), JointCorr("nose"), JointCorr("head"),
    JointCorr("left_shoulder"), JointCorr("left_elbow"), JointCorr("left_wrist"),
    JointCorr("right_shoulder"), JointCorr("right_elbow"), JointCorr("right_wrist"),
]
H36M_17 = MotionLayout(
    name="h36m_17",
    point_names=H36M_17_POINT_NAMES,
    correspondences=H36M_17_TO_SMPLX,
    description="Human3.6M-style 17-joint skeleton, common in 3D pose benchmarks.",
)

# ---------------------------------------------------------------------------
# VICON 50-marker optical mocap protocol (the marker set used by AMASS's
# SOMA/MoSh++ pipeline for BMLrub and related capture sessions).
#
# These are REAL vertex ids, not placeholders: extracted directly from a
# real AMASS SMPL-X release file's `markers_latent_vids` field (see
# doc/Mocap_Unification_Design.docx section 6.1 — this is the same
# `12_L_2_stageii.npz` sample analyzed there), which stores the exact
# marker-name -> SMPL-X-mesh-vertex-id mapping used to generate that
# capture's "latent markers". This is the correspondence table needed to
# fit raw C3D marker clouds using this same protocol onto the SMPL-X
# surface (VertexCorr, not JointCorr — markers sit on the skin, not at
# joint centers).
# ---------------------------------------------------------------------------
VICON_50_VERTEX_IDS = {
    "C7": 6607, "CLAV": 5656, "LAEL": 4351, "LANK": 5761, "LAOL": 4342,
    "LBAK": 3362, "LBHD": 2041, "LBWT": 5517, "LFHD": 1974, "LFRM": 4585,
    "LFWT": 3295, "LHEE": 8846, "LHPS": 4618, "LHTS": 4608, "LKNE": 3641,
    "LKNI": 3999, "LMT1": 5906, "LMT5": 5836, "LSHO": 3413, "LTHI": 3578,
    "LTIB": 3752, "LTOE": 5801, "LUPA": 4023, "LWPS": 4558, "LWTS": 4540,
    "RAEL": 7114, "RANK": 8576, "RAOL": 7078, "RBAK": 6813, "RBHD": 989,
    "RBWT": 7137, "RFHD": 2609, "RFRM": 6950, "RFWT": 6048, "RHEE": 8635,
    "RHPS": 7351, "RHTS": 8048, "RKNE": 6400, "RKNI": 6540, "RMT1": 8598,
    "RMT5": 8566, "RSHO": 6629, "RTHI": 6855, "RTIB": 6464, "RTOE": 8480,
    "RUPA": 7228, "RWPS": 7594, "RWTS": 7368, "STRN": 5449, "T10": 5496,
}
VICON_50_POINT_NAMES = list(VICON_50_VERTEX_IDS.keys())
VICON_50_TO_SMPLX = [VertexCorr(vid) for vid in VICON_50_VERTEX_IDS.values()]
VICON_50 = MotionLayout(
    name="vicon_50",
    point_names=VICON_50_POINT_NAMES,
    correspondences=VICON_50_TO_SMPLX,
    description=(
        "50-marker Vicon optical mocap protocol (AMASS SOMA/MoSh++ marker "
        "set), mapped to real SMPL-X mesh vertex ids. Use for C3D files "
        "whose marker labels match this convention."
    ),
)

ALL_BUILTIN_LAYOUTS = [COCO_17, OPENPOSE_25, MEDIAPIPE_33, H36M_17, VICON_50]
