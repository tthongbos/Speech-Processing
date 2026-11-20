"""Disk-backed caches.

  voices/<id>/ref.wav + meta.json   uploaded or recorded samples, deduplicated by content hash
  prompts/<engine>_<key>.*          encoded voice prompts (skips ASR and audio tokenization next time)
  segments/<key>.wav                one rendered sentence, keyed by engine, voice, text and audio params
  results/<key>.wav                 one joined result, keyed by its segment keys and the pause length
  history.json                      the last finished jobs
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config


def make_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.voices = root / "voices"
        self.prompts = root / "prompts"
        self.segments = root / "segments"
        self.results = root / "results"
        for d in (self.voices, self.prompts, self.segments, self.results):
            d.mkdir(parents=True, exist_ok=True)
        self.history_file = root / "history.json"
        self._lock = threading.Lock()
        self._prompt_mem: "OrderedDict[str, Any]" = OrderedDict()
        self.hits = {"prompt_memory": 0, "prompt_disk": 0, "segment": 0, "result": 0}

    # ----- voices -------------------------------------------------------
    def voice_dir(self, voice_id: str) -> Path:
        if not voice_id or not voice_id.isalnum():
            raise KeyError(voice_id)
        return self.voices / voice_id

    def voice_meta(self, voice_id: str) -> Dict[str, Any]:
        path = self.voice_dir(voice_id) / "meta.json"
        if not path.exists():
            raise KeyError(voice_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save_voice_meta(self, voice_id: str, meta: Dict[str, Any]) -> None:
        d = self.voice_dir(voice_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_voices(self) -> List[Dict[str, Any]]:
        out = []
        for meta_file in self.voices.glob("*/meta.json"):
            try:
                out.append(json.loads(meta_file.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def delete_voice(self, voice_id: str) -> None:
        d = self.voice_dir(voice_id)
        if d.exists():
            shutil.rmtree(d)
        for f in self.prompts.glob(f"*_{voice_id}_*"):
            f.unlink(missing_ok=True)
        with self._lock:
            for k in [k for k in self._prompt_mem if f"|{voice_id}|" in k]:
                self._prompt_mem.pop(k, None)

    # ----- voice prompts --------------------------------------------------
    def prompt_file(self, engine_id: str, engine_version: str, voice_id: str, ref_text: str, suffix: str) -> Path:
        return self.prompts / f"{engine_id}_{voice_id}_{make_key(engine_version, ref_text)}{suffix}"

    def prompt_get(self, mem_key: str) -> Optional[Any]:
        with self._lock:
            if mem_key in self._prompt_mem:
                self._prompt_mem.move_to_end(mem_key)
                return self._prompt_mem[mem_key]
        return None

    def prompt_put(self, mem_key: str, value: Any) -> None:
        with self._lock:
            self._prompt_mem[mem_key] = value
            self._prompt_mem.move_to_end(mem_key)
            while len(self._prompt_mem) > config.PROMPT_MEMORY_SLOTS:
                self._prompt_mem.popitem(last=False)

    # ----- segments and results -------------------------------------------
    def segment_path(self, key: str) -> Path:
        return self.segments / f"{key}.wav"

    def result_path(self, key: str) -> Path:
        return self.results / f"{key}.wav"

    # ----- history ------------------------------------------------------
    def history(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        try:
            return json.loads(self.history_file.read_text(encoding="utf-8"))
        except ValueError:
            return []

    def add_history(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            items = [e for e in self.history() if e.get("id") != entry.get("id")]
            items.insert(0, entry)
            self.history_file.write_text(json.dumps(items[: config.HISTORY_LIMIT], ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    # ----- maintenance ----------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        def size(d: Path) -> int:
            return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        return {
            "voices": len(list(self.voices.glob("*/meta.json"))),
            "prompts": len(list(self.prompts.iterdir())),
            "segments": len(list(self.segments.glob("*.wav"))),
            "results": len(list(self.results.glob("*.wav"))),
            "bytes": size(self.root),
            "prompts_in_memory": len(self._prompt_mem),
            "hits": dict(self.hits),
        }

    def clear(self, keep_voices: bool = True) -> None:
        targets = [self.prompts, self.segments, self.results] + ([] if keep_voices else [self.voices])
        for d in targets:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._prompt_mem.clear()
        if self.history_file.exists():
            self.history_file.unlink()
        self.stamp = time.time()
