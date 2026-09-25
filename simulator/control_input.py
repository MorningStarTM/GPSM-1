"""Turn player input into the control vector the model was trained on.

This is the piece that makes the model playable. Training derived control
from the *future of a recording* ("where did this person actually go next");
at play time nobody knows the future, so the same 14 numbers have to be
produced from what the player is asking for right now.

The translation is the standard one used for steerable locomotion:

1. Held keys become a *desired* forward speed and turn rate.
2. Those are eased toward smoothly, so a key press produces a believable
   ramp rather than an instant jump from standing to full speed.
3. The resulting motion is projected forward in time to say where the root
   would be at +0.1s, +0.25s and +0.5s — which is exactly what the control
   vector describes.

Step 3 assumes the character keeps its current speed and turn rate for the
next half second (a "unicycle" model: drive forward, steer). That is not
what the player will really do, but it is the same *kind* of short-horizon
guess the training data contained, which is what matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from src.gpsm.control.control_features import DEFAULT_HORIZONS_SECONDS

#: Speeds a key press asks for. Chosen to sit inside the range the training
#: clips actually contain (see the EDA notebook: forward speed runs to about
#: 3.5 m/s, backwards to about -2 m/s).
WALK_SPEED = 1.4          # m/s when "forward" is held
BACK_SPEED = -0.8         # m/s when "back" is held
STRAFE_SPEED = 0.9        # m/s sideways when "strafe left/right" is held
TURN_RATE = 1.6           # rad/s at full mouse deflection (~90 deg/s)

#: How quickly held keys are eased toward, in seconds to cover most of the
#: gap. Larger is smoother and more sluggish.
SPEED_SMOOTHING = 0.25
TURN_SMOOTHING = 0.15


@dataclass
class PlayerIntent:
    """What the player is asking for this frame.

    ``forward`` and ``strafe`` come from held keys; ``turn`` comes from the
    mouse, so it is continuous rather than just -1/0/+1.
    """

    forward: float = 0.0   # +1 forward, -1 back, 0 neither
    turn: float = 0.0      # + turns left, - turns right; 1.0 is full rate
    strafe: float = 0.0    # +1 step left, -1 step right, 0 neither

    @property
    def target_speed(self) -> float:
        if self.forward > 0:
            return WALK_SPEED
        if self.forward < 0:
            return BACK_SPEED
        return 0.0

    @property
    def target_strafe_speed(self) -> float:
        return float(np.clip(self.strafe, -1.0, 1.0)) * STRAFE_SPEED

    @property
    def target_turn_rate(self) -> float:
        return float(np.clip(self.turn, -1.0, 1.0)) * TURN_RATE


@dataclass
class ControlDriver:
    """Keeps the smoothed motion state and emits control vectors.

    One instance lives for the whole play session: it remembers the current
    speed and turn rate so they can ease toward what the player is asking
    for, instead of snapping.
    """

    horizons: Sequence[float] = DEFAULT_HORIZONS_SECONDS
    speed: float = 0.0             # current forward speed, m/s
    strafe_speed: float = 0.0      # current sideways speed, m/s (+ is left)
    turn_rate: float = 0.0         # current turn rate, rad/s (+ is left)
    names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        self.names = ["vel_fwd", "vel_left"]
        for horizon in self.horizons:
            ms = int(round(horizon * 1000))
            self.names += [f"pos_fwd_{ms}ms", f"pos_left_{ms}ms",
                            f"turn_cos_{ms}ms", f"turn_sin_{ms}ms"]

    @property
    def control_dim(self) -> int:
        return len(self.names)

    def update(self, intent: PlayerIntent, dt: float) -> np.ndarray:
        """Advance the smoothing by ``dt`` seconds and return the control.

        Args:
            intent: What the player is holding this frame.
            dt:     Seconds since the last update.

        Returns:
            ``(control_dim,)`` float32 — the same layout, in the same order
            and units, as the control MotionPrep extracts from recordings.
        """
        self.speed = _ease(self.speed, intent.target_speed, dt, SPEED_SMOOTHING)
        self.strafe_speed = _ease(self.strafe_speed, intent.target_strafe_speed, dt, SPEED_SMOOTHING)
        self.turn_rate = _ease(self.turn_rate, intent.target_turn_rate, dt, TURN_SMOOTHING)
        return self.control_vector()

    def control_vector(self) -> np.ndarray:
        """The current motion state, written as the 14-number control."""
        values = [self.speed, self.strafe_speed]

        for horizon in self.horizons:
            forward, left = _arc_offset(self.speed, self.turn_rate, horizon)
            # Strafing adds a straight sideways offset on top of the arc the
            # forward motion traces.
            left += self.strafe_speed * horizon
            turn = self.turn_rate * horizon
            values += [forward, left, float(np.cos(turn)), float(np.sin(turn))]

        return np.asarray(values, dtype=np.float32)


def _ease(current: float, target: float, dt: float, smoothing: float) -> float:
    """Move ``current`` toward ``target``, covering most of the gap in
    roughly ``smoothing`` seconds. Framerate-independent, so a slow frame
    does not produce a smaller step than it should."""
    if smoothing <= 0:
        return target
    alpha = 1.0 - float(np.exp(-dt / smoothing))
    return current + (target - current) * alpha


def _arc_offset(speed: float, turn_rate: float, horizon: float) -> "tuple[float, float]":
    """Where the root ends up after ``horizon`` seconds, in the character's
    own frame, if it keeps this speed and turn rate.

    Driving forward while turning traces a circular arc, so this is the
    standard unicycle result. When barely turning, the arc formula divides
    by a near-zero turn rate, so the straight-line case is handled directly.

    Returns:
        ``(forward, left)`` displacement in metres.
    """
    angle = turn_rate * horizon
    if abs(turn_rate) < 1e-4:
        return speed * horizon, 0.0
    radius = speed / turn_rate
    return radius * float(np.sin(angle)), radius * (1.0 - float(np.cos(angle)))
