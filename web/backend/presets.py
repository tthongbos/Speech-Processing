"""Built-in reference voices for quick demos: real speakers from FLEURS (Google, CC BY 4.0).

Each clip ships with its transcript, so picking a preset skips the ASR step entirely. The voices are seeded
into the voice store at startup with the same content-hash id an upload of the same file would get.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from . import audio
from .store import Store

PRESET_DIR = Path(__file__).resolve().parent / "presets"


def seed_presets(store: Store) -> int:
    index = PRESET_DIR / "presets.json"
    if not index.exists():
        return 0
    spec = json.loads(index.read_text(encoding="utf-8"))
    count = 0
    for i, v in enumerate(spec["voices"]):
        wav = PRESET_DIR / f"{v['slug']}.wav"
        if not wav.exists():
            continue
        voice_id = hashlib.sha256(wav.read_bytes()).hexdigest()[:20]
        vdir = store.voice_dir(voice_id)
        if not (vdir / "meta.json").exists():
            vdir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wav, vdir / "ref.wav")
        meta = {
            "id": voice_id, "name": v["name"], "source": "preset", "preset": True, "order": i,
            "gender": v["gender"], "original": wav.name, "duration": round(audio.duration_of(vdir / "ref.wav"), 2),
            "truncated": False, "created": 0, "transcript": v["transcript"], "reused": False,
            "credit": spec.get("license", ""),
        }
        store.save_voice_meta(voice_id, meta)
        count += 1
    return count
