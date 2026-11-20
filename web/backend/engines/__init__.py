"""Engine registry. Order here is the order of the model switch in the UI."""
from __future__ import annotations

import logging
from typing import Dict

from .. import config
from .base import EngineUnavailable, TTSEngine
from .mock_engine import MockEngine
from .mms_engine import MMSEngine
from .omnivoice_engine import OmniVoiceEngine

log = logging.getLogger("vong.engines")


def _omnivoice_runnable() -> bool:
    try:
        import torch  # noqa: F401
        import omnivoice  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the machine
        log.warning("OmniVoice not importable (%s); using the mock engine.", exc)
        return False
    from .omnivoice_engine import pick_device

    device = pick_device()
    if device == "cpu":
        log.warning("No CUDA or MPS device; using the mock engine. Set VONG_MOCK=0 to run OmniVoice on CPU anyway.")
        return False
    log.info("OmniVoice will run on %s", device)
    return True


def build_engines() -> Dict[str, TTSEngine]:
    if config.MOCK == "1":
        use_real = False
    elif config.MOCK == "0":
        use_real = True
    else:
        use_real = _omnivoice_runnable()
    omni = OmniVoiceEngine()
    engines: Dict[str, TTSEngine] = {
        "omnivoice": omni if use_real else MockEngine(omni),
        "model2": MMSEngine(),
    }
    return engines


__all__ = ["build_engines", "TTSEngine", "EngineUnavailable"]
