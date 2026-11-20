"""The contract every TTS model in the studio implements.

To add a model: subclass `TTSEngine`, fill `PARAMS`, implement `load`, `prepare_voice` and `synthesize`,
then register it in `backend/engines/__init__.py`. The job queue, caches, progress, normalization and the
web UI (which builds its controls from `PARAMS`) work for it unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

StepCallback = Callable[[], None]


@dataclass
class Param:
    key: str
    label: str
    type: str  # "int" | "float" | "bool" | "select"
    default: Any
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: List[Dict[str, Any]] = field(default_factory=list)
    group: str = "basic"  # "basic" | "advanced" | "pipeline"
    affects_audio: bool = True  # False: excluded from the segment cache key

    def coerce(self, value: Any) -> Any:
        if value is None:
            return self.default
        try:
            if self.type == "bool":
                return bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes", "on")
            if self.type == "int":
                v = int(round(float(value)))
            elif self.type == "float":
                v = float(value)
            else:
                allowed = [o["value"] for o in self.options]
                return value if value in allowed else self.default
        except (TypeError, ValueError):
            return self.default
        if self.min is not None:
            v = max(self.min, v)
        if self.max is not None:
            v = min(self.max, v)
        return type(v)(v)


# Pipeline settings shared by every engine: they run in the studio, around the model.
PIPELINE_PARAMS: List[Param] = [
    Param("normalize_vi", "Chuẩn hoá tiếng Việt", "bool", True,
          "Đọc số, %, °C, ngày tháng và chữ viết tắt thành chữ trước khi đưa vào mô hình.", group="pipeline"),
    Param("split_sentences", "Tách câu và sinh theo lô", "bool", True,
          "Chia văn bản thành từng câu, sinh nhiều câu trong một lượt và lưu cache theo từng câu.", group="pipeline"),
    Param("pause_ms", "Nghỉ giữa các câu", "int", 120, "Khoảng lặng chèn giữa hai câu, tính bằng ms.",
          min=0, max=800, step=20, group="pipeline", affects_audio=False),
    Param("trim_pauses", "Rút ngắn khoảng lặng dài", "bool", True,
          "Đo mức ồn nền của từng câu rồi cắt các khoảng lặng quá dài. Cần khi giọng mẫu thu ở nơi có tiếng ồn.",
          group="pipeline", affects_audio=False),
    Param("max_pause_ms", "Khoảng lặng dài nhất", "int", 400,
          "Khoảng lặng giữa câu dài hơn mức này bị rút về đúng mức này.",
          min=200, max=1500, step=50, group="pipeline", affects_audio=False),
    Param("batch_size", "Số câu mỗi lô", "int", 8, "Giảm nếu GPU báo hết bộ nhớ.",
          min=1, max=16, step=1, group="pipeline", affects_audio=False),
    Param("seed", "Seed", "int", 42, "Cố định để kết quả lặp lại được và cache có ý nghĩa.",
          min=0, max=999999, step=1, group="pipeline"),
]


class EngineUnavailable(Exception):
    pass


class TTSEngine:
    id: str = ""
    name: str = ""
    tagline: str = ""
    version: str = "1"
    placeholder: bool = False
    needs_voice: bool = True  # False for single-speaker models that ignore the reference clip
    PARAMS: List[Param] = []
    FACTS: List[Tuple[str, str]] = []

    def __init__(self) -> None:
        self.sample_rate = 24000
        self.state = "idle"  # idle | loading | ready | error | placeholder
        self.error: Optional[str] = None
        self.device = "cpu"

    # ----- schema -------------------------------------------------------
    def all_params(self) -> List[Param]:
        return list(self.PARAMS) + PIPELINE_PARAMS

    def resolve_params(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        raw = raw or {}
        return {p.key: p.coerce(raw.get(p.key)) for p in self.all_params()}

    def audio_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """The subset of params that changes the waveform: part of the cache key."""
        return {p.key: params[p.key] for p in self.all_params() if p.affects_audio and p.key in params}

    def info(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "tagline": self.tagline, "placeholder": self.placeholder,
            "needs_voice": self.needs_voice,
            "state": self.state, "error": self.error, "device": self.device, "sample_rate": self.sample_rate,
            "params": [asdict(p) for p in self.all_params()],
            "facts": [{"label": k, "value": v} for k, v in self.FACTS],
        }

    # ----- lifecycle ----------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def prepare_voice(self, wav_path: Path, ref_text: Optional[str], cache_file: Path,
                      on_stage: Callable[[str], None]) -> Tuple[Any, str, bool]:
        """Turn a reference clip into whatever the model conditions on.

        Returns (voice_prompt, resolved_ref_text, loaded_from_cache). Implementations should persist the
        prompt at `cache_file` so the next run skips audio tokenization and ASR entirely.
        """
        raise NotImplementedError

    def steps_per_batch(self, params: Dict[str, Any]) -> int:
        """How many times `on_step` fires per `synthesize` call, for the progress bar."""
        return 1

    def synthesize(self, texts: List[str], voice: Any, params: Dict[str, Any],
                   on_step: StepCallback) -> List[np.ndarray]:
        """Render a batch of sentences in the reference voice. Call `on_step()` as work advances;
        it raises when the user cancels, so let the exception propagate."""
        raise NotImplementedError
