"""Pytest configuration for GPSM's unit test suite.

This makes sure the repo root is on ``sys.path`` before any test module
runs, so tests can ``import src.gpsm...`` the same way the rest of the
codebase already does (e.g. ``python -m src.gpsm.tests.rollout_inference``),
regardless of which directory ``pytest`` happens to be invoked from.

Note: this ``tests/`` folder (repo root) holds automated pytest unit tests
for model internals. It is separate from ``src/gpsm/tests/``, which holds
runnable inference/visualization tooling (GIF demos, SMPL-X rendering) —
different purpose, kept in its own place on purpose.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --- fixtures for tests/mocap_unification (the copied src/gpsm/smplx package) ---
import pytest  # noqa: E402

from src.gpsm.smplx.config import find_smplx_model  # noqa: E402


@pytest.fixture(scope="session")
def smplx_model_path():
    """Path to the SMPL-X body model, or skip the test if it isn't on disk
    (the model is registration-gated, so it is never committed)."""
    path = find_smplx_model()
    if path is None:
        pytest.skip(
            "No SMPL-X model file found (registration-gated asset; put it in "
            "model/ or set GPSM_SMPLX_MODEL to enable model-dependent tests)."
        )
    return str(path)


@pytest.fixture(scope="session")
def smplx_body(smplx_model_path):
    """One shared SMPL-X body model for every test that needs it."""
    from src.gpsm.smplx.fitting.body import SMPLXBody
    return SMPLXBody(model_path=smplx_model_path)
