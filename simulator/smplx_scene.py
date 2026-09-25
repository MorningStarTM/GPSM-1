"""pyrender scene construction/update helpers for rendering a posed SMPL-X
body mesh.

Up-axis note: SMPL-X joint/vertex output from this repo's data + model
(model/SMPLX_FEMALE.npz, data/*.npz) is Z-up — verified empirically by
recentring a real forward-kinematics output on the pelvis joint and checking
per-axis range (Z spans ~1.5 m, matching body height; X/Y span ~0.2-0.5 m,
matching torso width/depth). The ground plane and camera below are built
around world +Z as "up" accordingly, not pyrender/OpenGL's usual default +Y.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh
import pyrender

UP = np.array([0.0, 0.0, 1.0])

GROUND_COLOR = np.array([0.55, 0.55, 0.62, 1.0])
BODY_COLOR = np.array([0.75, 0.78, 0.95, 1.0])


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = UP) -> np.ndarray:
    """Camera-to-world 4x4 pose matrix (pyrender convention: the camera looks
    along its own local -Z axis, with local +Y as "up")."""
    forward = target - eye
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-8 else np.array([0.0, 0.0, -1.0])
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        # `up` is (near-)parallel to the view direction — fall back to an
        # arbitrary perpendicular so the matrix stays well-defined.
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def make_ground_plane(center_xy: np.ndarray, z: float, size: float = 4.0) -> pyrender.Mesh:
    """A flat quad in the world XY plane (Z-up) at height `z` — a visual floor
    reference only, not physically meaningful."""
    half = size / 2.0
    cx, cy = float(center_xy[0]), float(center_xy[1])
    vertices = np.array([
        [cx - half, cy - half, z],
        [cx + half, cy - half, z],
        [cx + half, cy + half, z],
        [cx - half, cy + half, z],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.visual.face_colors = np.tile((GROUND_COLOR * 255).astype(np.uint8), (2, 1))
    return pyrender.Mesh.from_trimesh(tm, smooth=False)


def make_checkerboard_ground(size: float = 60.0, tile: float = 1.0, z: float = 0.0) -> pyrender.Mesh:
    """A large checkerboard floor on the world XY plane (Z-up).

    A single flat-coloured plane makes movement impossible to see — a
    character walking across it looks stationary, because nothing in the
    image changes. The alternating tiles give the eye something to move
    against, which is the whole point when the thing being judged is
    whether the character goes where it was told.

    Args:
        size: Total width of the floor, metres.
        tile: Width of one square, metres.
        z:    Floor height.
    """
    steps = int(size / tile)
    start = -size / 2.0
    vertices, faces, colors = [], [], []

    for ix in range(steps):
        for iy in range(steps):
            x0, y0 = start + ix * tile, start + iy * tile
            x1, y1 = x0 + tile, y0 + tile
            base = len(vertices)
            vertices += [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            shade = 0.62 if (ix + iy) % 2 == 0 else 0.48
            colors += [[shade, shade, shade + 0.04, 1.0]] * 4

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int64),
        vertex_colors=(np.array(colors) * 255).astype(np.uint8),
        process=False,
    )
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def follow_camera_pose(target: np.ndarray, distance: float = 4.5, height: float = 2.2,
                        behind: np.ndarray = np.array([1.0, -1.0, 0.0])) -> np.ndarray:
    """Camera pose that keeps ``target`` in view from a fixed offset.

    Needed because the character travels: a static camera loses it within a
    second or two, and then there is nothing to judge.

    Args:
        target:   ``(3,)`` world position to look at (the character's root).
        distance: How far away to sit, metres.
        height:   How far above the target, metres.
        behind:   Horizontal direction to sit in, relative to the target.
    """
    direction = behind[:2] / (np.linalg.norm(behind[:2]) + 1e-8)
    eye = np.array([
        target[0] + direction[0] * distance,
        target[1] + direction[1] * distance,
        target[2] + height,
    ])
    return look_at(eye, target)


def body_mesh_from_vertices(vertices: np.ndarray, faces: np.ndarray) -> pyrender.Mesh:
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.visual.vertex_colors = np.tile((BODY_COLOR * 255).astype(np.uint8), (vertices.shape[0], 1))
    return pyrender.Mesh.from_trimesh(tm, smooth=True)


def build_scene(
    vertices0: np.ndarray,
    faces: np.ndarray,
    viewport_size: Tuple[int, int] = (960, 720),
    with_ground: bool = True,
):
    """
    Builds a scene containing a ground plane sized to the body, the posed
    body mesh (frame 0), a camera framing the body, and two directional
    lights.

    Args:
        vertices0     : (V, 3) first-frame body vertices.
        faces         : (F, 3) SMPL-X mesh topology (constant across frames).
        viewport_size : (width, height) used only to set the camera aspect ratio.
        with_ground   : Add the small body-sized ground patch. Set False when
                        the caller supplies its own floor (e.g. the large
                        checkerboard the play harness uses) — two overlapping
                        floors z-fight and look like a rendering fault.

    Returns:
        (scene, body_node, camera_node)
    """
    if vertices0.ndim != 2 or vertices0.shape[1] != 3:
        raise ValueError(f"vertices0 must be (V, 3), got {vertices0.shape}")
    if not np.isfinite(vertices0).all():
        raise ValueError("vertices0 contains non-finite values.")

    scene = pyrender.Scene(bg_color=[0.04, 0.04, 0.07, 1.0], ambient_light=[0.35, 0.35, 0.38])

    lo = vertices0.min(axis=0)
    hi = vertices0.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.linalg.norm(hi - lo)) / 2.0, 0.5)

    ground_z = float(lo[2])
    if with_ground:
        scene.add(make_ground_plane(center_xy=center[:2], z=ground_z, size=max(radius * 6.0, 4.0)))

    body_mesh = body_mesh_from_vertices(vertices0, faces)
    body_node = scene.add(body_mesh)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, aspectRatio=viewport_size[0] / viewport_size[1])
    eye = center + np.array([radius * 2.0, -radius * 2.6, radius * 1.4])
    camera_node = scene.add(cam, pose=look_at(eye, center))

    key_pose = look_at(center + np.array([2.5, -2.5, 3.5]), center)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.5), pose=key_pose)
    fill_pose = look_at(center + np.array([-2.5, 2.0, 1.5]), center)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=fill_pose)

    return scene, body_node, camera_node


def update_body_mesh(scene: pyrender.Scene, body_node, vertices: np.ndarray, faces: np.ndarray):
    """Swaps the body node's mesh for a new pose and returns the new node.

    pyrender doesn't support mutating a Node's mesh vertex buffers in place,
    so each frame we remove the old node and add a freshly built one — cheap
    enough for a single body-sized mesh at interactive frame rates.
    """
    if not np.isfinite(vertices).all():
        raise ValueError("update_body_mesh: vertices contain non-finite values.")
    new_mesh = body_mesh_from_vertices(vertices, faces)
    scene.remove_node(body_node)
    return scene.add(new_mesh)
