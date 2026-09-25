"""A game-engine-editor-style fly camera for the live viewer.

pyrender.Viewer's default camera is a "trackball": dragging the mouse
orbits the *entire scene* (ground included) around a pivot point. That's
fine for inspecting a single static object, but it's not how a game engine
viewport behaves — in Unity's Scene view or Unreal's editor viewport, the
camera moves through a fixed world; the ground never rotates.

This module replaces the trackball with a standard yaw/pitch fly camera:
hold the left mouse button and drag to look around (camera rotates in
place), WASD moves along the view direction, Space/Left-Ctrl move along
world up/down, and scrolling adjusts move speed. The world itself is never
touched.

Implemented as a `pyrender.Viewer` subclass (`GameViewer`) rather than a
full custom renderer, since pyrender's Viewer already handles the OpenGL
context, window, and render loop correctly — only the camera-control layer
needed replacing.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pyglet
import pyrender
from pyrender.constants import RenderFlags

from simulator.smplx_scene import UP, look_at


class FlyCamera:
    """Yaw/pitch fly camera, Z-up (matches smplx_scene.py's world convention).

    Position + yaw (rotation around world Z) + pitch (rotation away from the
    horizontal plane, clamped to avoid flipping past straight up/down).
    """

    def __init__(self, position, yaw: float, pitch: float, up: np.ndarray = UP):
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.up = np.asarray(up, dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "FlyCamera":
        """Builds an equivalent FlyCamera from an existing camera-to-world
        pose matrix, so the fly camera starts exactly where pyrender's
        auto-computed initial view was, instead of snapping elsewhere."""
        position = matrix[:3, 3].copy()
        z_axis = matrix[:3, 2]  # pyrender convention: camera looks along local -Z
        forward = -z_axis
        norm = np.linalg.norm(forward)
        forward = forward / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
        yaw = float(np.arctan2(forward[1], forward[0]))
        pitch = float(np.arcsin(np.clip(forward[2], -1.0, 1.0)))
        return cls(position, yaw, pitch)

    def forward_vector(self) -> np.ndarray:
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        return np.array([cp * cy, cp * sy, sp])

    def right_vector(self) -> np.ndarray:
        f = self.forward_vector()
        r = np.cross(f, self.up)
        n = np.linalg.norm(r)
        return r / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])

    def look(self, dyaw: float, dpitch: float, pitch_limit: float = 1.4835) -> None:
        """pitch_limit default ~85 degrees — stops just short of straight
        up/down, where yaw becomes degenerate."""
        self.yaw = float((self.yaw + dyaw) % (2.0 * np.pi))
        self.pitch = float(np.clip(self.pitch + dpitch, -pitch_limit, pitch_limit))

    def move(self, forward_amount: float, right_amount: float, up_amount: float) -> None:
        self.position = (
            self.position
            + self.forward_vector() * forward_amount
            + self.right_vector() * right_amount
            + self.up * up_amount
        )

    def to_matrix(self) -> np.ndarray:
        target = self.position + self.forward_vector()
        return look_at(self.position, target, up=self.up)


class GameViewer(pyrender.Viewer):
    """pyrender.Viewer with an editor-style fly camera instead of a trackball.

    Controls:
        left-drag         look around (camera rotates in place)
        W / A / S / D      move forward / left / backward / right
        Space / Left-Ctrl   move up / down (world axis)
        scroll             adjust move speed
        ESC                quit (in addition to pyrender's default 'q')

    All other pyrender.Viewer hotkeys (f = fullscreen, l = lighting, z =
    reset view, ...) still work — only mouse control and W/A/S/D/Space/
    Left-Ctrl are overridden, since those collide with pyrender's own
    letter-key hotkeys (e.g. 'w' otherwise toggles wireframe mode).
    """

    MOVE_SPEED_MIN = 0.2
    MOVE_SPEED_MAX = 30.0
    LOOK_SENSITIVITY = 0.0035  # radians of yaw/pitch per pixel of drag

    def __init__(self, *args, move_speed: float = 3.0, external_camera: bool = False, **kwargs):
        # external_camera=True hands the camera to the caller: this class
        # stops consuming W/A/S/D and stops writing the camera pose, so
        # something else (a third-person follow camera, say) can own both.
        # The mouse-look and movement handlers below simply go quiet.
        self._external_camera = external_camera
        self._fly_cam: Optional[FlyCamera] = None
        self._move_speed = float(move_speed)
        self._last_frame_time = time.time()
        self._keys = pyglet.window.key.KeyStateHandler()
        super().__init__(*args, **kwargs)
        self.push_handlers(self._keys)

    @property
    def held_keys(self):
        """Which keys are currently held down.

        Index it with pyglet key codes, e.g. ``viewer.held_keys[key.UP]``.
        Exposed so something outside the camera (a character being driven by
        the player) can read the keyboard without reaching into internals.
        Keys the camera itself uses (W/A/S/D, Space, Ctrl) appear here too,
        so a caller should pick keys that do not collide with those.
        """
        return self._keys

    # ------------------------------------------------------------------
    # Mouse: look only, never orbits/pans/zooms the world
    # ------------------------------------------------------------------

    def on_mouse_press(self, x, y, buttons, modifiers):
        pass  # swallow — no trackball state machine to arm

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._external_camera:
            return
        if buttons & pyglet.window.mouse.LEFT and self._fly_cam is not None:
            self._fly_cam.look(-dx * self.LOOK_SENSITIVITY, dy * self.LOOK_SENSITIVITY)

    def on_mouse_release(self, x, y, button, modifiers):
        pass

    def on_mouse_scroll(self, x, y, dx, dy):
        factor = 1.15 if dy > 0 else (1.0 / 1.15 if dy < 0 else 1.0)
        self._move_speed = float(np.clip(self._move_speed * factor, self.MOVE_SPEED_MIN, self.MOVE_SPEED_MAX))
        self._message_text = f"Move speed: {self._move_speed:.1f} m/s"
        self._message_opac = 1.0 + self._ticks_till_fade

    # ------------------------------------------------------------------
    # Keyboard: WASD/Space/Ctrl are movement (handled continuously in
    # _render via KeyStateHandler), everything else falls through to
    # pyrender.Viewer's normal hotkeys.
    # ------------------------------------------------------------------

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()
            return
        if symbol in GameViewer._movement_keys():
            return
        super().on_key_press(symbol, modifiers)

    @staticmethod
    def _movement_keys():
        k = pyglet.window.key
        return (k.W, k.A, k.S, k.D, k.SPACE, k.LCTRL, k.RCTRL)

    # ------------------------------------------------------------------
    # Render: apply continuous WASD movement, then use the fly camera's
    # pose instead of pyrender's trackball pose.
    # ------------------------------------------------------------------

    def _render(self):
        now = time.time()
        dt = max(now - self._last_frame_time, 0.0)
        self._last_frame_time = now

        if self._external_camera:
            # The caller owns the camera; fall through to the rendering half
            # below without touching camera_node.matrix.
            return self._render_scene()

        if self._fly_cam is None:
            # First frame — camera_node.matrix was set by pyrender's own
            # _reset_view()/_compute_initial_camera_pose() before any draw
            # call, so this is always a valid pose to seed from.
            self._fly_cam = FlyCamera.from_matrix(self._camera_node.matrix)

        step = self._move_speed * dt
        keys = self._keys
        k = pyglet.window.key
        forward_amt = (step if keys[k.W] else 0.0) - (step if keys[k.S] else 0.0)
        right_amt = (step if keys[k.D] else 0.0) - (step if keys[k.A] else 0.0)
        up_amt = (step if keys[k.SPACE] else 0.0) - (step if (keys[k.LCTRL] or keys[k.RCTRL]) else 0.0)
        if forward_amt or right_amt or up_amt:
            self._fly_cam.move(forward_amt, right_amt, up_amt)

        self._camera_node.matrix = self._fly_cam.to_matrix()
        return self._render_scene()

    def _render_scene(self):
        """The rendering half, shared by both camera modes.

        Mirrors pyrender.Viewer._render from the lighting setup onward;
        kept in sync with pyrender==0.1.45.
        """
        scene = self.scene

        vli = self.viewer_flags['lighting_intensity']
        if self.viewer_flags['use_raymond_lighting']:
            for n in self._raymond_lights:
                n.light.intensity = vli / 3.0
                if not self.scene.has_node(n):
                    scene.add_node(n, parent_node=self._camera_node)
        else:
            self._direct_light.light.intensity = vli
            for n in self._raymond_lights:
                if self.scene.has_node(n):
                    self.scene.remove_node(n)

        if self.viewer_flags['use_direct_lighting']:
            if not self.scene.has_node(self._direct_light):
                scene.add_node(self._direct_light, parent_node=self._camera_node)
        elif self.scene.has_node(self._direct_light):
            self.scene.remove_node(self._direct_light)

        flags = RenderFlags.NONE
        if self.render_flags['flip_wireframe']:
            flags |= RenderFlags.FLIP_WIREFRAME
        elif self.render_flags['all_wireframe']:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_flags['all_solid']:
            flags |= RenderFlags.ALL_SOLID

        if self.render_flags['shadows']:
            flags |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
        if self.render_flags['vertex_normals']:
            flags |= RenderFlags.VERTEX_NORMALS
        if self.render_flags['face_normals']:
            flags |= RenderFlags.FACE_NORMALS
        if not self.render_flags['cull_faces']:
            flags |= RenderFlags.SKIP_CULL_FACES

        self._renderer.render(self.scene, flags)
