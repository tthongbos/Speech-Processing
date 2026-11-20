"""OmniVoice (Zhu et al., 2026, arXiv:2604.00688): zero-shot TTS with a masked diffusion language model.

Generation unmasks acoustic tokens over `num_step` iterations. The model's forward pass runs once per
iteration for the whole batch, so a forward hook gives an exact progress signal without touching the library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .. import config
from .base import Param, StepCallback, TTSEngine

OMNIVOICE_PARAMS: List[Param] = [
    Param("num_step", "Số bước giải mã", "int", 32,
          "Số lượt mở mask acoustic token. Ít bước thì nhanh, nhiều bước thì rõ hơn; paper ghi nhận chất lượng giảm rõ ở 8 bước.",
          min=4, max=64, step=4),
    Param("speed", "Tốc độ đọc", "float", 1.0, "Lớn hơn 1 đọc nhanh hơn, nhỏ hơn 1 đọc chậm hơn.",
          min=0.7, max=1.3, step=0.05),
    Param("guidance_scale", "Classifier-free guidance", "float", 2.0,
          "Độ bám theo văn bản và giọng mẫu. Cao quá dễ gắt giọng.", min=0.0, max=4.0, step=0.1),
    Param("language", "Ngôn ngữ", "select", "vi", "Gợi ý ngôn ngữ cho mô hình; để tự nhận nếu văn bản trộn nhiều thứ tiếng.",
          options=[{"value": "vi", "label": "Tiếng Việt"}, {"value": "en", "label": "Tiếng Anh"},
                   {"value": "auto", "label": "Tự nhận"}]),
    Param("t_shift", "Dịch thời gian (t_shift)", "float", 0.1,
          "Nhỏ hơn thì dồn bước vào vùng nhiễu cao, nơi quyết định khung và alignment.",
          min=0.05, max=1.0, step=0.05, group="advanced"),
    Param("position_temperature", "Nhiệt độ vị trí", "float", 5.0,
          "Lớn hơn 0 thì chọn vị trí mở mask ngẫu nhiên kiểu Gumbel.", min=0.0, max=10.0, step=0.5, group="advanced"),
    Param("class_temperature", "Nhiệt độ token", "float", 0.0, "0 là chọn token tham lam (greedy).",
          min=0.0, max=1.5, step=0.05, group="advanced"),
    Param("denoise", "Khử nhiễu giọng mẫu", "bool", True, "Thêm token <|denoise|> khi có audio mẫu.", group="advanced"),
    Param("postprocess_output", "Cắt khoảng lặng thừa", "bool", True,
          "Bỏ khoảng lặng dài ở giữa, đầu và cuối audio sinh ra.", group="advanced"),
]

OMNIVOICE_FACTS = [
    ("Paper", "Zhu et al., 2026, arXiv:2604.00688"),
    ("Kiến trúc", "Diffusion language model, sinh song song bằng cách mở mask dần"),
    ("Đầu ra", "24 kHz, mono"),
    ("Giấy phép", "Code Apache 2.0, trọng số CC-BY-NC"),
]

def pick_device() -> str:
    """cuda first, then Apple Silicon (MPS), then CPU; VONG_DEVICE overrides."""
    import torch

    if config.DEVICE != "auto":
        return config.DEVICE
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


GEN_KEYS = ("num_step", "guidance_scale", "t_shift", "position_temperature", "class_temperature",
            "denoise", "postprocess_output")


class OmniVoiceEngine(TTSEngine):
    id = "omnivoice"
    name = "OmniVoice"
    tagline = "Zero-shot, hơn 600 ngôn ngữ, clone giọng từ 3 đến 10 giây audio"
    version = "omnivoice-0.2"
    PARAMS = OMNIVOICE_PARAMS
    FACTS = OMNIVOICE_FACTS

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self._step_cb: Optional[StepCallback] = None

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        device = pick_device()
        if config.DTYPE != "auto":
            dtype = getattr(torch, config.DTYPE)
        else:
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = OmniVoice.from_pretrained(config.OMNIVOICE_REPO, device_map=device, dtype=dtype)
        self.model.eval()
        self.sample_rate = int(getattr(self.model, "sampling_rate", None) or config.SAMPLE_RATE)
        self.device = device
        # one forward call per decoding step for the whole batch -> exact progress
        self.model.register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inputs, _output) -> None:
        cb = self._step_cb
        if cb is not None:
            cb()

    def _prompt_class(self):
        try:
            from omnivoice import VoiceClonePrompt
        except ImportError:  # older layouts
            from omnivoice.models.omnivoice import VoiceClonePrompt
        return VoiceClonePrompt

    def prepare_voice(self, wav_path: Path, ref_text: Optional[str], cache_file: Path,
                      on_stage: Callable[[str], None]) -> Tuple[Any, str, bool]:
        VoiceClonePrompt = self._prompt_class()
        if cache_file.exists():
            prompt = VoiceClonePrompt.load(str(cache_file), map_location="cpu")
            prompt.ref_audio_tokens = prompt.ref_audio_tokens.to(self.model.device)
            return prompt, prompt.ref_text, True
        on_stage("Chép lời giọng mẫu bằng Vọng" if not ref_text else "Mã hoá giọng mẫu thành acoustic token")
        prompt = self.model.create_voice_clone_prompt(ref_audio=str(wav_path), ref_text=ref_text or None)
        prompt.save(str(cache_file))
        return prompt, prompt.ref_text, False

    def steps_per_batch(self, params: Dict[str, Any]) -> int:
        return int(params["num_step"])

    def synthesize(self, texts: List[str], voice: Any, params: Dict[str, Any],
                   on_step: StepCallback) -> List[np.ndarray]:
        import torch

        torch.manual_seed(int(params["seed"]))  # seeds CPU, CUDA and MPS generators
        language = None if params["language"] == "auto" else params["language"]
        gen = {k: params[k] for k in GEN_KEYS}
        self._step_cb = on_step
        try:
            audios = self.model.generate(
                text=list(texts),
                language=language,
                voice_clone_prompt=[voice] * len(texts),
                speed=float(params["speed"]),
                **gen,
            )
        finally:
            self._step_cb = None
        return [np.asarray(a, dtype=np.float32).reshape(-1) for a in audios]
