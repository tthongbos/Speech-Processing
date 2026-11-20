"""A stand-in engine for working on the website without a GPU.

It exposes the same parameters as the engine it imitates, takes about as many progress steps, and returns a
synthetic, speech-shaped signal (one voiced burst per syllable) so the player, waveform and caches can all
be exercised end to end. Start the server with VONG_MOCK=1 to force it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import Param, StepCallback, TTSEngine


class MockEngine(TTSEngine):
    def __init__(self, imitate: TTSEngine) -> None:
        super().__init__()
        self.id = imitate.id
        self.name = imitate.name
        self.tagline = imitate.tagline + " (chế độ mô phỏng, không dùng GPU)"
        self.version = imitate.version + "-mock"
        self.PARAMS = imitate.PARAMS
        self.FACTS = list(imitate.FACTS) + [("Chế độ", "Mô phỏng: âm thanh tổng hợp giả, chỉ để thử giao diện")]
        self._steps_key = "num_step" if any(p.key == "num_step" for p in imitate.PARAMS) else None

    def load(self) -> None:
        time.sleep(1.2)
        self.device = "cpu (mô phỏng)"

    def prepare_voice(self, wav_path: Path, ref_text: Optional[str], cache_file: Path,
                      on_stage: Callable[[str], None]) -> Tuple[Any, str, bool]:
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data, data["ref_text"], True
        on_stage("Chép lời giọng mẫu bằng Vọng" if not ref_text else "Mã hoá giọng mẫu thành acoustic token")
        time.sleep(1.5 if not ref_text else 0.6)
        seed = int(hashlib.sha1(str(wav_path).encode()).hexdigest()[:6], 16)
        data = {"pitch": 110 + seed % 120, "ref_text": ref_text or "Đây là lời giọng mẫu được chép tự động trong chế độ mô phỏng."}
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data, data["ref_text"], False

    def steps_per_batch(self, params: Dict[str, Any]) -> int:
        return int(params.get(self._steps_key, 16)) if self._steps_key else 16

    def synthesize(self, texts: List[str], voice: Any, params: Dict[str, Any],
                   on_step: StepCallback) -> List[np.ndarray]:
        steps = self.steps_per_batch(params)
        per_step = 0.02 + 0.012 * len(texts)
        for _ in range(steps):
            time.sleep(per_step)
            on_step()
        speed = float(params.get("speed", 1.0))
        return [self._speechish(t, voice["pitch"], speed, int(params.get("seed", 0))) for t in texts]

    def _speechish(self, text: str, pitch: float, speed: float, seed: int) -> np.ndarray:
        sr = self.sample_rate
        rng = np.random.default_rng(seed + len(text))
        out: List[np.ndarray] = []
        for word in text.split():
            dur = (0.12 + 0.025 * len(word)) / speed
            n = int(sr * dur)
            t = np.arange(n) / sr
            f0 = pitch * (1 + 0.08 * rng.standard_normal()) * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
            phase = 2 * np.pi * np.cumsum(f0) / sr
            voiced = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 6))
            env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
            out.append((0.25 * voiced * env).astype(np.float32))
            gap = 0.18 if word[-1:] in ",.;:!?" else 0.05
            out.append(np.zeros(int(sr * gap / speed), dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(sr // 2, dtype=np.float32)
