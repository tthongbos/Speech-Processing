"""Model 2: MMS-TTS (Meta, Massively Multilingual Speech), checkpoint facebook/mms-tts-vie.

A VITS model trained per language: one built-in Vietnamese voice, no voice cloning, 16 kHz output, and a
single non-autoregressive forward pass per sentence. Ported from the group's CLI (mms_tts/cli.py in
tthongbos/Speech-Processing): same checkpoint, NFC normalization, speakable-token check, seed and CPU threads.
"""
from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import Param, StepCallback, TTSEngine

MMS_MODEL_ID = os.environ.get("VONG_MMS_MODEL", "facebook/mms-tts-vie")
MMS_DEVICE = os.environ.get("VONG_MMS_DEVICE", "auto")  # VITS is light: CPU is often fast enough


class MMSEngine(TTSEngine):
    id = "model2"
    name = "MMS-TTS"
    tagline = "VITS của Meta, một giọng tiếng Việt có sẵn, không clone giọng, 16 kHz"
    version = "mms-tts-vie"
    needs_voice = False
    PARAMS: List[Param] = [
        Param("speaking_rate", "Tốc độ đọc", "float", 1.0,
              "Lớn hơn 1 đọc nhanh hơn. VITS chia độ dài dự đoán cho hệ số này.",
              min=0.6, max=1.6, step=0.05),
        Param("noise_scale", "Độ biến thiên giọng (noise_scale)", "float", 0.667,
              "Nhiễu trong prior của flow. Cao thì ngữ điệu sống động hơn nhưng dễ méo.",
              min=0.0, max=1.2, step=0.01),
        Param("noise_scale_duration", "Độ biến thiên nhịp (noise_scale_duration)", "float", 0.8,
              "Nhiễu của bộ dự đoán độ dài âm vị (stochastic duration predictor).",
              min=0.0, max=1.5, step=0.05, group="advanced"),
    ]
    FACTS = [
        ("Paper", "Pratap et al., 2023, Scaling Speech Technology to 1,000+ Languages"),
        ("Kiến trúc", "VITS: VAE, normalizing flow và HiFi-GAN, sinh trong một lượt"),
        ("Checkpoint", "facebook/mms-tts-vie"),
        ("Đầu ra", "16 kHz, mono, một giọng cố định"),
        ("Giấy phép", "Trọng số CC-BY-NC 4.0"),
        ("Nguồn", "mms_tts/ trong repo nhóm"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = 16000
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        device = MMS_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("VONG_MMS_CPU_THREADS", "4")))
        self.tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL_ID)
        self.model = VitsModel.from_pretrained(MMS_MODEL_ID).to(device).eval()
        self.sample_rate = int(self.model.config.sampling_rate)
        self.device = device

    def prepare_voice(self, wav_path: Optional[Path], ref_text: Optional[str], cache_file: Path,
                      on_stage: Callable[[str], None]) -> Tuple[Any, str, bool]:
        return None, "", True  # single-speaker model: the reference clip is not used

    def steps_per_batch(self, params: Dict[str, Any]) -> int:
        return max(1, int(params.get("batch_size", 1)))  # one tick per sentence

    def synthesize(self, texts: List[str], voice: Any, params: Dict[str, Any],
                   on_step: StepCallback) -> List[np.ndarray]:
        import torch
        from transformers import set_seed

        set_seed(int(params["seed"]))
        self.model.speaking_rate = float(params["speaking_rate"])
        self.model.noise_scale = float(params["noise_scale"])
        self.model.noise_scale_duration = float(params["noise_scale_duration"])
        out: List[np.ndarray] = []
        for text in texts:
            text = " ".join(unicodedata.normalize("NFC", text).split())
            inputs = self.tokenizer(text, return_tensors="pt")
            ids = inputs["input_ids"]
            speakable = torch.ones_like(ids, dtype=torch.bool)
            for tid in (self.tokenizer.pad_token_id, self.tokenizer.unk_token_id):
                if tid is not None:
                    speakable &= ids != tid
            if not speakable.any().item():
                out.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.float32))
            else:
                with torch.inference_mode():
                    wav = self.model(**inputs.to(self.model.device)).waveform[0]
                out.append(wav.detach().cpu().float().numpy().reshape(-1))
            on_step()
        return out
