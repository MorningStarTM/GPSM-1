"""MotionMamba: GPSM's SSM-based motion model (scaffold).

This module defines ``MotionMamba``, GPSM's Mamba/SSM-based alternative to
``StateMachineGPT`` (``sm_gpt.py``). It mirrors ``StateMachineGPT``'s public
interface on purpose (same config pattern, same ``forward``/``predict_next``/
``rollout`` signatures, same checkpoint file format) so every existing tool
built around ``StateMachineGPT`` — the trainer, the SMPL-X visualization
scripts, the 3D simulator — keeps working, just pointed at a different
checkpoint.

We call this class "MotionMamba", not "...GPT", on purpose: it has no
attention and no transformer block in it anywhere. Its sequence-mixing
mechanism is a state space model (SSM), so the name should say that, not
imply a transformer.

Implementation status (see ``doc/Mamba_Implementation_Task_Breakdown.docx``
for the full phased plan this file follows):

* **Task 1.1 (done)** — the model *scaffold*: config validation, the
  ``forward``/``predict_next``/``rollout`` interface, and checkpoint
  save/load. (The original placeholder block used here, ``_ScaffoldBlock``,
  has since been removed now that Task 1.3 supplies a real one — see git
  history if you want to see what it looked like.)
* **Task 1.2 (done)** — :class:`SelectiveScan`: the core recurrence math,
  written and unit-tested on its own (``tests/test_selective_scan.py``).
* **Task 1.3 (done)** — :class:`MambaBlock` assembles the real block around
  :class:`SelectiveScan` (input projection, causal convolution, gating,
  output projection), and :class:`ResidualBlock` wraps it with
  :class:`RMSNorm` and a residual connection. :class:`MotionMamba` now
  uses a stack of these — the model is a real (if not yet speed-optimized
  or carefully initialized) Mamba model end to end.
* **Task 1.4 (done)** — :func:`_init_dt_proj` gives ``MambaBlock.dt_proj``
  a targeted initialization (ported from ``mamba_simple.py``, Task 0.1)
  instead of the generic default every other ``nn.Linear`` gets.
  ``MotionMamba._init_weights`` was updated to not clobber it afterwards
  (see its docstring — this was a real, easy-to-miss bug, not a
  hypothetical one).
* **Task 1.5 (done, pulled forward into 1.3)** — the Mamba-specific config
  keys (``d_state``, ``d_conv``, ``expand``, ``dt_rank``) — see
  :func:`_validate_config` for why this could not wait.
* **Task 1.6 (done)** — trained via ``main_motion_mamba.py`` /
  ``motion_mamba_trainer.py`` on the real GPSM data
  (``checkpoints/best_motion_mamba``): train_loss fell smoothly from
  0.576 to 0.069 over 15 epochs, val_loss from 0.467 to 0.061, no
  NaN/Inf batches skipped. Checkpoint save/load and a real rollout on a
  genuine seed frame were both verified afterwards. **Phase 1 is
  complete** — see Phase 2 in ``doc/Mamba_Implementation_Task_Breakdown
  .docx`` for what comes next (correctness/visual comparison against
  ``StateMachineGPT``).

This is GPSM's own implementation. We do not depend on ``mamba_ssm`` and do
not vendor ``mamba-minimal``'s code — the official Mamba paper and those two
repositories (``state-spaces/mamba``, ``johnma2006/mamba-minimal``) are used
only as reading material while writing our own version. See
``doc/Mamba_for_GPSM.docx`` for the background concepts.
"""

from __future__ import annotations

import json
from typing import Optional
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file

from src.gpsm.utils.logger import logger
from src.gpsm.utils.utils import _to_jsonable


def _validate_config(config: dict) -> dict:
    """Check a ``MotionMamba`` config and return a validated copy.

    Note on task ordering: the doc's plan assigns adding the Mamba-specific
    keys (``d_state``, ``d_conv``, ``expand``, ``dt_rank``) to Task 1.5, as
    a later step — but :class:`MambaBlock` (Task 1.3) cannot be built
    without them (e.g. ``in_proj``'s output width depends on ``expand``),
    so that small piece of Task 1.5 is pulled forward into this task
    instead of left as dead validation code nothing reads yet.

    Also derives ``d_inner`` (the block's internal working width) and
    stores it back into the returned config, the same "single source of
    truth" pattern ``sm_gpt.py``'s ``_validate_config`` uses for
    ``head_size`` — every block reads ``config["d_inner"]`` instead of
    recomputing ``expand * n_embd`` separately in more than one place.

    Args:
        config: Raw config dict, e.g. loaded from YAML/JSON.

    Returns:
        A validated copy of ``config`` (copied so the caller's own dict is
        never mutated), with ``d_inner`` added.

    Raises:
        KeyError: If a required key is missing.
    """
    required = [
        "state_dim", "n_embd", "n_layers", "dropout", "learning_rate",
        "d_state", "d_conv", "expand", "dt_rank",
    ]
    for key in required:
        if key not in config:
            raise KeyError(f"Config missing required key: '{key}'")

    config = dict(config)
    config["d_inner"] = config["expand"] * config["n_embd"]
    # Optional: how many control numbers accompany each pose frame. 0 means
    # "no control", i.e. the plain next-pose model this class started as, so
    # older configs and checkpoints keep working untouched.
    config.setdefault("control_dim", 0)
    config["input_dim"] = config["state_dim"] + config["control_dim"]
    return config


class SelectiveScan(nn.Module):
    """The core recurrence at the heart of a Mamba block.

    This is *the* selective state space model (SSM) computation the whole
    Mamba architecture is built around — everything else in a Mamba block
    (input projection, causal convolution, gating) exists just to feed this
    one recurrence the right numbers and use its output. Written here from
    first principles (Algorithm 2 in the Mamba paper), cross-checked
    against the reference implementations while writing it, but not copied
    from either — see the module docstring for why.

    Used by :class:`MambaBlock`, which supplies the ``delta``/``B``/``C``
    this class needs from its own input. Still unit-tested on its own too
    (``tests/test_selective_scan.py``), independent of the block around it.

    The maths, in words
    --------------------
    A state space model keeps a hidden state ``h`` that gets nudged by the
    input at every timestep and slowly forgotten over time — think of it
    as a running summary of everything seen so far. Concretely, for one
    timestep ``t``::

        h(t) = A_bar(t) * h(t-1) + B_bar(t) * u(t)     # update the state
        y(t) = C(t) . h(t)                              # read a prediction out of it

    ``u`` is this block's input, ``h`` is the hidden state, and ``y`` is
    the output. ``A_bar``/``B_bar`` are *discretized* versions of two
    learned parameters ``A`` and ``B`` — discretized because ``A``/``B``
    describe a *continuous*-time system (standard in classical control
    theory), and this model runs on discrete timesteps (frame 1, frame 2,
    ...), so a learned step size ``delta`` (``Δ``) converts one into the
    other:

        A_bar(t) = exp(delta(t) * A)             # exact for a linear system (zero-order hold)
        B_bar(t) = delta(t) * B(t)                # a simplified (Euler) approximation, not exact
                                                    #   -- intentional: the Mamba paper notes A
                                                    #      matters far more than B does here.

    What makes this "selective" (Mamba's actual contribution over older,
    non-selective SSMs) is *which* of these are allowed to depend on the
    input: ``A`` is a fixed, learned parameter — the same at every
    timestep, for every input. ``delta``, ``B``, and ``C`` are instead
    *computed from* the current input (outside this class, by the block
    around it) — so the model can decide, based on what it currently sees,
    how strongly to write into the state (``B``), how fast that state
    decays (``delta``, which scales ``A``'s effect), and how to read a
    prediction back out of it (``C``). A fixed-parameter SSM cannot do
    this; that selectivity is the whole reason Mamba works better than the
    S4 models it's built on.

    Implementation note
    --------------------
    The loop below is a plain, sequential Python loop over timesteps,
    built entirely from ordinary differentiable PyTorch operations
    (multiply, add, exp, sum). Because of that, **no custom backward() is
    written or needed** — PyTorch's autograd differentiates straight
    through the loop automatically, exactly like it would for any other
    ``nn.Module.forward()``. This is the simple, "obviously correct"
    version; it is not the fast version (a real training run over long
    sequences would want a parallelized scan instead — deliberately out of
    scope here, see ``doc/Mamba_Implementation_Task_Breakdown.docx``
    Phase 4, which is optional and deferred).
    """

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selective scan over a whole sequence.

        Args:
            u     : ``(batch, seq_len, d_inner)`` — the block's input at
                    every timestep.
            delta : ``(batch, seq_len, d_inner)`` — per-timestep,
                    per-channel step size (already positive, e.g. via
                    softplus — that is the caller's responsibility, not
                    this class's).
            A     : ``(d_inner, d_state)`` — fixed, learned. Not
                    input-dependent (see class docstring for why).
            B     : ``(batch, seq_len, d_state)`` — input-dependent,
                    shared across the ``d_inner`` channels.
            C     : ``(batch, seq_len, d_state)`` — input-dependent,
                    shared across the ``d_inner`` channels.
            D     : ``(d_inner,)`` — fixed, learned skip-connection weight
                    (``y`` gets ``D * u`` added directly, bypassing the
                    state entirely — a learned residual).

        Returns:
            ``y``: ``(batch, seq_len, d_inner)`` — same shape as ``u``.
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # ---- discretize: turn the continuous-time A, B into their
        # per-timestep, per-channel discrete equivalents A_bar, B_bar_u.
        # Shapes below all broadcast up to (batch, seq_len, d_inner, d_state).
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        B_bar_u = (
            delta.unsqueeze(-1)        # (batch, seq_len, d_inner, 1)
            * B.unsqueeze(2)           # (batch, seq_len, 1, d_state)
            * u.unsqueeze(-1)          # (batch, seq_len, d_inner, 1)
        )  # (batch, seq_len, d_inner, d_state)

        # ---- sequential recurrence over time: h(t) = A_bar(t) * h(t-1) + B_bar_u(t) ----
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs_per_timestep = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar_u[:, t]  # (batch, d_inner, d_state)
            # read a prediction out of the state via C(t): a dot product
            # over the state dimension, per (batch, d_inner) channel.
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            outputs_per_timestep.append(y_t)
        y = torch.stack(outputs_per_timestep, dim=1)  # (batch, seq_len, d_inner)

        # ---- learned skip connection, bypassing the state entirely ----
        y = y + u * D
        return y


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Mamba's own architecture (and the reference implementations we studied
    in Task 0.1) normalizes each block's input with RMSNorm rather than the
    more familiar ``nn.LayerNorm`` that ``StateMachineGPT`` uses. The
    difference is small: RMSNorm rescales a vector by its root-mean-square
    magnitude only — it does not also recenter it to zero mean the way
    LayerNorm does (no mean subtraction, so there is no learned bias term
    either, only a learned per-channel scale). It is simpler and slightly
    cheaper, and is the standard choice for this architecture, so
    :class:`MotionMamba` uses it too for consistency, including for its own
    final norm.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: ``(..., d_model)``. Returns: same shape as ``x``."""
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# Fixed hyperparameters for _init_dt_proj below. These are not exposed as
# model config keys on purpose (see doc/Mamba_Implementation_Task_Breakdown
# .docx's general preference for keeping the config surface small) — they
# are init-time constants, not something a training run needs to tune per
# experiment. Values match the reference repos' own defaults (Task 0.1).
_DT_MIN = 1e-3
_DT_MAX = 1e-1
_DT_INIT_FLOOR = 1e-4
_DT_SCALE = 1.0


def _init_dt_proj(dt_proj: nn.Linear, dt_rank: int) -> None:
    """Give ``dt_proj`` a targeted initialization instead of a generic one.

    ``dt_proj`` expands ``delta`` from its small ``dt_rank`` width up to the
    full ``d_inner`` width. Left at a generic N(0, 0.02)/zero-bias
    initialization (what every other Linear in this model gets — see
    ``MotionMamba._init_weights``), every channel would start with almost
    the same, near-zero ``delta`` — meaning every SSM channel would start
    out updating its state at nearly the same, very slow rate. A targeted
    init instead spreads different channels across a range of starting
    speeds, which the Mamba paper's authors and the reference
    implementations (Task 0.1) found trains noticeably better. Ported from
    ``mamba_simple.py``'s ``__init__`` — the formulas themselves are
    self-contained arithmetic, not systems engineering, so there is no
    reason to skip them.

    Two things happen here:

    1. **Weight** — filled with ``Uniform(-s, s)`` where
       ``s = dt_rank ** -0.5 * _DT_SCALE``. Scaling by ``dt_rank ** -0.5``
       keeps this layer's *output* variance roughly independent of
       ``dt_rank`` (a smaller ``dt_rank`` means fewer terms summed in the
       matrix multiply, so each individual weight needs to be
       proportionally larger to land at a similar overall output scale —
       the same reasoning behind standard init schemes like Xavier/He
       init, just applied to this specific layer).
    2. **Bias** — set so that ``softplus(bias)`` (the actual ``delta``
       value used once the block runs — see ``MambaBlock.forward()``) is
       spread *log-uniformly* across ``[_DT_MIN, _DT_MAX]`` at
       initialization: some channels start reacting quickly, others
       slowly. Since ``softplus`` cannot be inverted by simple algebra
       into "just assign this bias", we sample the *target* ``delta``
       value first, then solve for the bias that produces it:
       ``bias = softplus^-1(delta) = delta + log(1 - exp(-delta))``,
       computed via the numerically stable form below (working with
       ``exp(-delta)``, which stays safely bounded in ``(0, 1)`` for
       ``delta > 0``, avoids ever computing ``exp(delta)`` directly, which
       could overflow for a larger sampled ``delta``).

    Both the weight and bias are marked ``_no_reinit`` afterwards (same
    convention/name the official repo uses for this) so
    ``MotionMamba._init_weights``'s later generic initialization pass
    knows to leave them alone instead of overwriting this with N(0, 0.02).

    Args:
        dt_proj : The ``Linear(dt_rank, d_inner)`` layer to initialize,
                  modified in place.
        dt_rank : ``dt_proj.in_features`` — passed explicitly rather than
                  read off the layer purely for readability at the call
                  site.
    """
    dt_init_std = dt_rank**-0.5 * _DT_SCALE
    nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

    d_inner = dt_proj.out_features
    log_dt_min, log_dt_max = math.log(_DT_MIN), math.log(_DT_MAX)
    target_dt = torch.exp(
        torch.rand(d_inner) * (log_dt_max - log_dt_min) + log_dt_min
    ).clamp(min=_DT_INIT_FLOOR)

    # Numerically stable softplus^-1: softplus(x) = log(1 + exp(x)), so
    # x = log(exp(target_dt) - 1) = target_dt + log(1 - exp(-target_dt)).
    inverse_softplus_bias = target_dt + torch.log(-torch.expm1(-target_dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inverse_softplus_bias)

    dt_proj.weight._no_reinit = True
    dt_proj.bias._no_reinit = True


class MambaBlock(nn.Module):
    """The full Mamba block: projections + causal conv + gate, wrapped
    around :class:`SelectiveScan`.

    :class:`SelectiveScan` (Task 1.2) is only the core recurrence — this
    class is everything *around* it: getting the input into the right
    shape and width, computing the per-timestep ``delta``/``B``/``C`` the
    scan needs, and gating/projecting its output back down. Follows the
    block diagram in ``doc/Mamba_for_GPSM.docx`` (Figure 2), written as our
    own module — see that doc for the full plain-language walkthrough of
    each step below.

    Forward pass, in order:
        1. ``in_proj``   — one Linear splits the block's input into two
           equal-width paths: ``x`` (the "main" path, goes through the scan)
           and ``res`` (the "gate" path, used only to gate the scan's
           output at the end).
        2. ``conv1d``    — a short *causal* depthwise convolution over
           ``x``, giving every timestep a little local context (a few
           neighboring frames) cheaply, before the scan runs. "Causal"
           means it only ever looks backward in time, never forward — it
           is padded on the left and trimmed on the right so frame ``t``'s
           output never depends on frame ``t+1``.
        3. ``SiLU``       — activation after the conv.
        4. ``x_proj``    — a small Linear reads ``delta``/``B``/``C``
           directly off the (already convolved) ``x``. This is where
           "selectivity" enters — see :class:`SelectiveScan`'s docstring.
        5. :class:`SelectiveScan` — the actual recurrence (Task 1.2).
        6. **gate**       — the scan's output is multiplied, elementwise,
           by ``SiLU(res)`` — this lets the block learn to suppress or pass
           through information per-channel.
        7. ``out_proj``  — projects back down to the model's width
           (``n_embd``). The residual connection around the whole block is
           added by :class:`ResidualBlock`, not here.
    """

    def __init__(self, config: dict):
        super().__init__()
        n_embd = config["n_embd"]
        d_inner = config["d_inner"]
        d_state = config["d_state"]
        d_conv = config["d_conv"]
        dt_rank = config["dt_rank"]

        # Step 1: one projection, split in two (main path + gate path).
        self.in_proj = nn.Linear(n_embd, d_inner * 2, bias=False)

        # Step 2: causal depthwise convolution. groups=d_inner makes this
        # "depthwise" — each of the d_inner channels is convolved with its
        # own small kernel, independently of the others (cheap: d_inner
        # tiny 1-D filters, not one big mixed filter). padding=d_conv - 1
        # pads on the left only in effect, once we trim the right-hand
        # overhang below — see forward().
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Step 4: reads delta (before its softplus + dt_proj), B, and C
        # off the convolved x, all in one small Linear.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # Expands delta from a small rank up to the full d_inner width.
        # Given a targeted initialization (not the generic default every
        # other Linear in this model gets) — see _init_dt_proj() above.
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        _init_dt_proj(self.dt_proj, dt_rank)

        # A is a fixed, learned parameter (see SelectiveScan's docstring
        # for why it is not input-dependent). Initialized as
        # -1, -2, ..., -d_state, the same across every d_inner channel to
        # start — a standard, well-tested starting point for this kind of
        # state space model (borrowed from the S4 line of work Mamba is
        # built on), not something specific to GPSM. Stored as its log
        # (A_log) so that A = -exp(A_log) is guaranteed negative (a stable,
        # decaying state) for any real-valued A_log the optimizer produces.
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # D is a fixed, learned per-channel skip weight (see
        # SelectiveScan.forward()'s docstring). Initialized to 1.
        self.D = nn.Parameter(torch.ones(d_inner))

        self.scan = SelectiveScan()
        self.out_proj = nn.Linear(d_inner, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: ``(B, T, n_embd)``. Returns: ``(B, T, n_embd)``."""
        B, T, _ = x.shape

        x_and_res = self.in_proj(x)  # (B, T, 2 * d_inner)
        x, res = x_and_res.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Conv1d expects channels before time: (B, d_inner, T).
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :T]  # trim the right-hand overhang -> causal
        x = x.transpose(1, 2)  # back to (B, T, d_inner)
        x = torch.nn.functional.silu(x)

        A = -torch.exp(self.A_log)  # (d_inner, d_state) — always negative

        x_dbl = self.x_proj(x)  # (B, T, dt_rank + 2 * d_state)
        dt_rank = self.dt_proj.in_features
        d_state = A.shape[1]
        delta, ssm_B, ssm_C = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
        delta = torch.nn.functional.softplus(self.dt_proj(delta))  # (B, T, d_inner), must stay positive

        y = self.scan(x, delta, A, ssm_B, ssm_C, self.D)  # (B, T, d_inner)

        y = y * torch.nn.functional.silu(res)  # gate
        return self.out_proj(y)  # (B, T, n_embd)


class ResidualBlock(nn.Module):
    """Wraps :class:`MambaBlock` with an RMSNorm and a residual connection.

    Same role as ``sm_gpt.py``'s ``Block`` class (which wraps attention +
    feed-forward the same way), just around a single Mamba mixer instead —
    Mamba blocks are usually described as being self-contained enough that
    they do not need a separate feed-forward sub-layer the way a
    transformer block does.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config["n_embd"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: ``(B, T, n_embd)``. Returns: ``(B, T, n_embd)``."""
        return self.mixer(self.norm(x)) + x


class MotionMamba(nn.Module):
    """SSM-based next-pose model for GPSM.

    ``MotionMamba`` predicts pose ``t + 1`` from a window of past poses,
    exactly like ``StateMachineGPT`` (``sm_gpt.py``), but using a stack of
    SSM blocks as the sequence-mixing mechanism instead of causal
    self-attention.

    Input  : ``obs`` ``(B, T, state_dim)`` — window of ``T`` past poses.
    Output : ``logits`` ``(B, T, state_dim)`` — predicted pose at every
             step (use ``logits[:, -1, :]`` for the actual next-frame
             prediction).

    Two things are deliberately absent compared to ``StateMachineGPT``:

    * **No positional embedding.** An SSM's recurrence encodes step order
      implicitly — each hidden state is built from the previous one — so
      there is nothing extra to add.
    * **No ``block_size`` ceiling.** Causal attention needs a fixed-size
      mask, capping how long a sequence can be. An SSM's recurrence has no
      such cap.

    Config keys (see :func:`_validate_config`):
        state_dim      Width of one pose frame.
        n_embd         Model embedding width.
        n_layers       Number of stacked blocks.
        dropout        Dropout probability (currently unused by
                        :class:`MambaBlock`/:class:`ResidualBlock` — kept
                        for interface parity with ``StateMachineGPT``;
                        revisit if/when dropout is added to the real block).
        learning_rate  AdamW learning rate.
        d_state        SSM state width per channel (``N`` in the paper).
        d_conv         Width of the causal depthwise convolution.
        expand         Multiplier on ``n_embd`` giving the block's
                        internal working width, ``d_inner``.
        dt_rank        Rank of the small projection that produces delta,
                        before it is expanded up to ``d_inner``.
        loss_scale     Optional scalar multiplied onto the MSE loss
                        (read by the trainer, not by this class).

    See the module docstring for what is, and is not, implemented yet.
    """

    def __init__(self, config: dict):
        super().__init__()
        config = _validate_config(dict(config))
        self.config = config

        # Input is the pose plus (optionally) the control that goes with it;
        # the output stays pose-only, since control is supplied, never predicted.
        self.prev_pos_embedding = nn.Linear(config["input_dim"], config["n_embd"])
        self.blocks = nn.Sequential(*[ResidualBlock(config) for _ in range(config["n_layers"])])
        self.ln_f = RMSNorm(config["n_embd"])
        self.next_pos_head = nn.Linear(config["n_embd"], config["state_dim"])

        self.optimizer = optim.AdamW(self.parameters(), lr=config["learning_rate"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights. Same scheme as ``StateMachineGPT`` for every
        ``nn.Linear`` layer, EXCEPT a weight or bias tensor marked
        ``_no_reinit`` (currently only ``MambaBlock.dt_proj`` — see
        ``_init_dt_proj()``), which is left exactly as it was set up
        already. Without this check, this method would run *after*
        ``_init_dt_proj()`` (called from inside each ``MambaBlock``, which
        is built before ``self.apply(self._init_weights)`` runs) and
        silently overwrite its carefully targeted initialization with a
        generic N(0, 0.02) — this is precisely the "training framework
        has a post-initialization hook that zeros every Linear bias"
        problem the official Mamba repo's own troubleshooting notes warn
        about.

        Also does not touch :class:`MambaBlock`'s ``A_log``, ``D``, or its
        ``conv1d`` — those are given their own, more specific
        initializations directly in :class:`MambaBlock.__init__`, since a
        generic N(0, 0.02) would not make sense for them (e.g. ``A_log``
        needs to make ``A`` negative, not centered on zero)."""
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_no_reinit", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor, targets=None) -> torch.Tensor:
        """Predict a pose at every input timestep.

        Args:
            obs     : ``(B, T, D)`` or ``(T, D)`` — window of past frames,
                      each ``concat(pose, control)`` when the model is
                      configured with control (``D == input_dim``), or just
                      the pose when ``control_dim`` is 0.
            targets : Unused — kept only so this method has the same
                      signature as ``StateMachineGPT.forward()``, which
                      also ignores it.

        Returns:
            ``logits`` with the same leading dims as ``obs`` and last dim
            ``state_dim`` — pose only. The model never predicts control:
            control is an input describing what the motion should do next,
            supplied by the player at run time.

        Raises:
            TypeError: If ``obs`` is not a tensor.
            ValueError: If ``obs`` has the wrong number of dimensions, an
                empty sequence, or a feature width that does not match
                ``config['input_dim']``.
        """
        if not torch.is_tensor(obs):
            raise TypeError(f"`obs` must be a torch.Tensor, got {type(obs)}")

        if obs.dim() == 2:
            obs = obs.unsqueeze(0)  # (T, D) -> (1, T, D)
            squeeze_B = True
        elif obs.dim() == 3:
            squeeze_B = False
        else:
            raise ValueError(f"`obs` must be 2D (T,D) or 3D (B,T,D). Got {tuple(obs.shape)}")

        B, T, D = obs.shape

        if T <= 0:
            raise ValueError(f"Sequence length T must be > 0, got T={T}")
        if D != self.config["input_dim"]:
            raise ValueError(
                f"Input feature dim D={D} does not match config['input_dim']="
                f"{self.config['input_dim']} (state_dim {self.config['state_dim']} "
                f"+ control_dim {self.config['control_dim']})."
            )

        x = self.prev_pos_embedding(obs)  # (B, T, n_embd) — no positional embedding needed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.next_pos_head(x)  # (B, T, state_dim)

        if squeeze_B:
            logits = logits.squeeze(0)  # (T, state_dim)
        return logits

    @torch.no_grad()
    def predict_next(self, x_one: torch.Tensor, return_last_only: bool = True) -> torch.Tensor:
        """Predict the next pose frame for a single sample.

        Same signature and behaviour as ``StateMachineGPT.predict_next()``.

        Args:
            x_one            : ``(T, D)`` or ``(1, T, D)``.
            return_last_only : ``True`` -> return ``(D,)`` / ``(1, D)`` —
                the predicted next frame. ``False`` -> return every
                predicted frame, ``(T, D)`` / ``(1, T, D)``.

        Returns:
            ``(D,)`` if the input was ``(T, D)``, or ``(1, D)`` if the
            input was ``(1, T, D)`` (shape swaps accordingly when
            ``return_last_only=False`` — see above).
        """
        self.eval()

        if not torch.is_tensor(x_one):
            raise TypeError(f"x_one must be a torch.Tensor, got {type(x_one)}")

        if x_one.dim() == 2:
            x_in = x_one.unsqueeze(0)  # (1, T, D)
            squeeze_B = True
        elif x_one.dim() == 3:
            if x_one.shape[0] != 1:
                raise ValueError(f"predict_next expects B=1, got B={x_one.shape[0]}")
            x_in = x_one
            squeeze_B = False
        else:
            raise ValueError(f"x_one must be (T,D) or (1,T,D). Got {tuple(x_one.shape)}")

        x_in = x_in.to(self.device)
        logits = self.forward(x_in)  # (1, T, state_dim)

        if return_last_only:
            pred = logits[:, -1, :]  # (1, D)
            return pred.squeeze(0) if squeeze_B else pred
        return logits.squeeze(0) if squeeze_B else logits

    @torch.no_grad()
    def rollout(
        self,
        seed_frames: torch.Tensor,
        n_steps: Optional[int] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively generate future poses.

        With control configured, this is no longer something the model can
        do on its own: each new frame needs to be told where the motion is
        meant to go, exactly as a game would read the player's stick once
        per frame. So ``control`` supplies one control vector per generated
        frame, and its length decides how many frames come out.

        Without control (``control_dim == 0``) the old behaviour applies and
        ``n_steps`` is used instead — the model free-runs from the seed.

        Note on cost: this calls :meth:`forward` on the whole growing window
        every step. That is the straightforward version, not the fast one —
        an SSM can carry a small fixed-size state between steps instead of
        reprocessing the history, which is the real reason to use one for
        real-time generation. Swapping the internals for that does not
        change this signature or its output.

        Args:
            seed_frames : ``(T, input_dim)`` — the starting window, each row
                          ``concat(pose, control)`` when control is in use.
            n_steps     : How many frames to generate. Only for the
                          no-control case; ignored when ``control`` is given.
            control     : ``(n_steps, control_dim)`` — the control for each
                          frame being generated. ``control[i]`` belongs to
                          the *newly generated* frame ``i``, and is what
                          steers the frame after it.

        Returns:
            ``(n_steps, state_dim)`` — the predicted poses only (no control,
            and not including the seed).
        """
        self.eval()
        if seed_frames.dim() != 2:
            raise ValueError(f"seed_frames must be (T, D), got {tuple(seed_frames.shape)}")

        control_dim = self.config["control_dim"]
        if control_dim > 0:
            if control is None:
                raise ValueError(
                    "This model was configured with control_dim="
                    f"{control_dim}, so rollout() needs a `control` array "
                    "(one row per frame to generate) — it cannot invent the "
                    "motion's intent by itself."
                )
            control = control.to(self.device)
            if control.dim() != 2 or control.shape[1] != control_dim:
                raise ValueError(
                    f"control must be (n_steps, {control_dim}), got {tuple(control.shape)}"
                )
            n_steps = control.shape[0]
        elif n_steps is None:
            raise ValueError("Without control, rollout() needs n_steps.")

        window = seed_frames.to(self.device)
        predicted_poses = []

        for step in range(n_steps):
            next_pose = self.predict_next(window)  # (state_dim,)
            predicted_poses.append(next_pose.unsqueeze(0))

            # Extend the window with the new frame. With control, that frame
            # is the predicted pose plus the control belonging to it.
            if control_dim > 0:
                next_row = torch.cat([next_pose, control[step]], dim=0)
            else:
                next_row = next_pose
            window = torch.cat([window, next_row.unsqueeze(0)], dim=0)

        return torch.cat(predicted_poses, dim=0)  # (n_steps, state_dim)

    def rollout_batch(self, seed: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Differentiable, batched autoregressive rollout (for training).

        The training-time counterpart to :meth:`rollout` — not decorated
        with ``@torch.no_grad()`` and does not call ``self.eval()``, so
        gradients flow through every predicted step. Needed for supervised
        fine-tuning on multi-step generation (mirrors
        ``StateMachineGPT.rollout_batch()``).

        Args:
            seed    : ``(B, T0, D)`` — batch of history windows (``T0 >= 1``).
            n_steps : Number of future frames to generate.

        Returns:
            ``(B, n_steps, D)`` — the predicted frames only (the seed is
            not included).
        """
        if seed.dim() != 3:
            raise ValueError(f"seed must be (B, T0, D), got {tuple(seed.shape)}")

        window = seed.to(self.device)
        predicted_frames = []

        for _ in range(n_steps):
            logits = self.forward(window)  # (B, t, D)
            next_frame = logits[:, -1:, :]  # (B, 1, D)
            predicted_frames.append(next_frame)
            window = torch.cat([window, next_frame], dim=1)

        return torch.cat(predicted_frames, dim=1)  # (B, n_steps, D)

    def print_param_size(self) -> None:
        """Log the total and trainable parameter counts, in millions."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters      : {total / 1e6:.3f} M")
        logger.info(f"Trainable parameters  : {trainable / 1e6:.3f} M")

    def save(self, path: str) -> None:
        """Save a plain ``torch.save`` checkpoint (model + optimizer +
        config). Same format as ``StateMachineGPT.save()``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str, device=None) -> None:
        """Load a checkpoint written by :meth:`save`. Same format and
        behaviour as ``StateMachineGPT.load()``."""
        if device is None:
            device = next(self.parameters()).device
        ckpt = torch.load(path, map_location=device)
        if "config" in ckpt and ckpt["config"] != self.config:
            raise ValueError("Checkpoint config != current model config.")
        self.load_state_dict(ckpt["model_state_dict"])
        if self.optimizer is not None and ckpt.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        logger.info(f"Loaded model from {path}.")

    def save_safetensors(self, path: str, save_optimizer: bool = True) -> None:
        """Save weights as safetensors, plus a JSON metadata sidecar.

        Writes a ``"model_class": "MotionMamba"`` field into the metadata
        (read back by ``rollout_inference.py``'s checkpoint loader — Task
        2.2 — so a checkpoint on disk always says which class it belongs
        to). Otherwise identical in format to
        ``StateMachineGPT.save_safetensors()``.

        Args:
            path           : Checkpoint path (extension is ignored — the
                              same base name is reused for
                              ``.safetensors``/``.json``/``.optim.safetensors``).
            save_optimizer : Whether to also save optimizer state.
        """
        base = os.path.splitext(path)[0]
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

        save_file({k: v.detach().cpu() for k, v in self.state_dict().items()}, base + ".safetensors")

        meta = {
            "format": "safetensors_ckpt_v1",
            "model_class": "MotionMamba",
            "config": _to_jsonable(self.config),
            "has_optimizer": False,
            "optimizer_state_keys": None,
        }

        if save_optimizer and self.optimizer is not None:
            try:
                opt_sd = self.optimizer.state_dict()
                opt_tensors = {}
                opt_keys = {}
                opt_nontensor = {}

                for pidx, st in opt_sd.get("state", {}).items():
                    p = str(pidx)
                    opt_keys[p] = []
                    opt_nontensor[p] = {}
                    for sk, sv in st.items():
                        if torch.is_tensor(sv):
                            opt_tensors[f"state/{p}/{sk}"] = sv.detach().cpu()
                            opt_keys[p].append(sk)
                        else:
                            opt_nontensor[p][sk] = _to_jsonable(sv)

                save_file(opt_tensors, base + ".optim.safetensors")
                meta.update({
                    "has_optimizer": True,
                    "optimizer_state_keys": opt_keys,
                    "optimizer_param_groups": _to_jsonable(opt_sd.get("param_groups", [])),
                    "optimizer_state_nontensor": opt_nontensor,
                })
            except Exception as e:
                meta["optimizer_save_error"] = str(e)

        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_safetensors(self, path: str, device=None, load_optimizer: bool = True, strict: bool = True) -> None:
        """Load a checkpoint written by :meth:`save_safetensors`. Same
        format and behaviour as ``StateMachineGPT.load_safetensors()``."""
        if device is None:
            device = next(self.parameters()).device
        base = os.path.splitext(path)[0]

        meta = None
        meta_path = base + ".json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("config") is not None and meta["config"] != _to_jsonable(self.config):
                raise ValueError("Checkpoint config != current model config.")

        model_sd = {k: v.to(device) for k, v in load_file(base + ".safetensors").items()}
        self.load_state_dict(model_sd, strict=strict)

        optim_path = base + ".optim.safetensors"
        if (load_optimizer and self.optimizer is not None
                and meta and meta.get("has_optimizer") and os.path.exists(optim_path)):
            try:
                opt_sd = self.optimizer.state_dict()
                if "optimizer_param_groups" in meta:
                    opt_sd["param_groups"] = meta["optimizer_param_groups"]
                if "optimizer_state_nontensor" in meta:
                    for pidx, fields in meta["optimizer_state_nontensor"].items():
                        pidx_int = int(pidx)
                        opt_sd["state"].setdefault(pidx_int, {}).update(fields)
                for key, tensor in load_file(optim_path).items():
                    _, pidx, sk = key.split("/", 2)
                    opt_sd["state"].setdefault(int(pidx), {})[sk] = tensor.to(device)
                self.optimizer.load_state_dict(opt_sd)
            except Exception as e:
                logger.warning(f"Skipped optimizer restore: {e}")

        logger.info(f"Loaded safetensors checkpoint from {base}.")
