"""A single GPU worker that runs synthesis jobs one at a time and reports fine-grained progress."""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import audio
from .engines import EngineUnavailable, TTSEngine
from .pauses import plan_text
from .store import Store, make_key

log = logging.getLogger("vong.jobs")

TERMINAL = ("done", "error", "cancelled")

# The reference clip is conditioned on with its long pauses shortened. OmniVoice estimates the output
# length from (reference text / reference duration); pauses in the clip make the speaker look slow, the
# estimate comes out too long and the model fills the slack with silence. Part of every cache key.
REF_PREP = "tight-400ms"

# Progress budget: preparing the voice up to 12%, generation up to 96%, joining to 100%.
P_VOICE, P_GEN, P_JOIN = 0.12, 0.96, 1.0


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    engine_id: str
    voice_id: str
    text: str
    ref_text: str
    params: Dict[str, Any]
    status: str = "queued"  # queued | running | done | error | cancelled
    stage: str = "Đang chờ tới lượt"
    progress: float = 0.0
    detail: str = ""
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    version: int = 0
    queue_position: int = 0

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id, "engine": self.engine_id, "status": self.status, "stage": self.stage,
            "detail": self.detail, "progress": round(self.progress, 4), "error": self.error,
            "result": self.result, "created": self.created, "started": self.started,
            "finished": self.finished, "queue_position": self.queue_position, "version": self.version,
            "elapsed": round((self.finished or time.time()) - self.started, 2) if self.started else 0,
        }


class JobManager:
    def __init__(self, engines: Dict[str, TTSEngine], store: Store) -> None:
        self.engines = engines
        self.store = store
        self.jobs: Dict[str, Job] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="tts-worker", daemon=True)
        self._worker.start()

    # ----- public API -----------------------------------------------------
    def submit(self, engine_id: str, voice_id: str, text: str, ref_text: str, params: Dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], engine_id=engine_id, voice_id=voice_id, text=text,
                  ref_text=ref_text, params=params)
        with self._lock:
            self.jobs[job.id] = job
            job.queue_position = sum(1 for j in self.jobs.values() if j.status == "queued") - 1
            self._prune()
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self.jobs.get(job_id)
        if job and job.status not in TERMINAL:
            job.cancel_requested = True
            if job.status == "queued":
                self._finish(job, "cancelled", stage="Đã huỷ")
        return job

    # ----- internals ------------------------------------------------------
    def _prune(self) -> None:
        done = [j for j in self.jobs.values() if j.status in TERMINAL]
        for j in sorted(done, key=lambda j: j.created)[:-50]:
            self.jobs.pop(j.id, None)

    def _update(self, job: Job, **changes: Any) -> None:
        for k, v in changes.items():
            setattr(job, k, v)
        job.version += 1

    def _finish(self, job: Job, status: str, **changes: Any) -> None:
        self._update(job, status=status, finished=time.time(), **changes)
        with self._lock:
            queued = sorted((j for j in self.jobs.values() if j.status == "queued"), key=lambda j: j.created)
            for i, j in enumerate(queued):
                if j.queue_position != i:
                    self._update(j, queue_position=i)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status in TERMINAL:
                continue
            try:
                self._run(job)
            except Cancelled:
                self._finish(job, "cancelled", stage="Đã huỷ", detail="")
            except EngineUnavailable as exc:
                self._finish(job, "error", stage="Không chạy được", error=str(exc))
            except Exception as exc:  # report, keep the worker alive
                log.error("job %s failed: %s", job.id, traceback.format_exc())
                msg = str(exc) or exc.__class__.__name__
                if "out of memory" in msg.lower():
                    msg = "GPU hết bộ nhớ. Giảm “Số câu mỗi lô” hoặc rút ngắn văn bản rồi thử lại."
                self._finish(job, "error", stage="Lỗi khi sinh audio", error=msg)

    def _prepare_voice(self, job: Job, engine: TTSEngine):
        self._update(job, stage="Chuẩn bị giọng mẫu", progress=0.03)
        meta = self.store.voice_meta(job.voice_id)
        wav_path = self._tight_reference(job.voice_id)
        ref_text = job.ref_text.strip()
        mem_key = f"{engine.id}|{job.voice_id}|{engine.version}|{REF_PREP}|{ref_text}"
        prompt_entry = self.store.prompt_get(mem_key)
        prompt_source = "memory"
        if prompt_entry is not None:
            self.store.hits["prompt_memory"] += 1
        else:
            suffix = ".json" if engine.version.endswith("-mock") else ".pt"
            cache_file = self.store.prompt_file(engine.id, f"{engine.version}|{REF_PREP}", job.voice_id, ref_text, suffix)
            voice_obj, resolved, from_disk = engine.prepare_voice(
                wav_path, ref_text or None, cache_file,
                on_stage=lambda s: self._update(job, stage=s, progress=0.06))
            prompt_entry = (voice_obj, resolved)
            self.store.prompt_put(mem_key, prompt_entry)
            prompt_source = "disk" if from_disk else "encoded"
            if from_disk:
                self.store.hits["prompt_disk"] += 1
        voice_obj, resolved_ref = prompt_entry
        if not ref_text and resolved_ref and meta.get("transcript") != resolved_ref:
            meta["transcript"] = resolved_ref
            self.store.save_voice_meta(job.voice_id, meta)
        return voice_obj, resolved_ref, prompt_source

    def _tight_reference(self, voice_id: str):
        vdir = self.store.voice_dir(voice_id)
        tight = vdir / "ref_tight.wav"
        if not tight.exists():
            data, sr = audio.read_wav(vdir / "ref.wav")
            audio.write_wav(tight, audio.shorten_pauses(data, sr, max_pause_ms=400, edge_ms=100), sr)
        return tight

    def _check_cancel(self, job: Job) -> None:
        if job.cancel_requested:
            raise Cancelled()

    def _run(self, job: Job) -> None:
        engine = self.engines[job.engine_id]
        self._update(job, status="running", started=time.time(), stage="Đang nạp mô hình", progress=0.01,
                     queue_position=0)
        while engine.state == "loading":
            self._check_cancel(job)
            time.sleep(0.25)
        if engine.placeholder:
            raise EngineUnavailable(f"{engine.name} chưa được gắn vào.")
        if engine.state != "ready":
            raise EngineUnavailable(f"Không nạp được {engine.name}: {engine.error}")

        params = job.params
        t0 = time.time()

        # 1. text -> segments
        plan = plan_text(job.text, normalize=params["normalize_vi"], split=params["split_sentences"])
        segments = plan.segments
        text = plan.display()
        if not segments:
            raise ValueError("Văn bản trống sau khi chuẩn hoá.")

        # 2. voice prompt: memory -> disk -> encode (and transcribe if needed)
        if not engine.needs_voice:  # single-speaker model (MMS): no reference clip involved
            job.voice_id = ""
            voice_obj, resolved_ref, prompt_source = None, "", "none"
        else:
            voice_obj, resolved_ref, prompt_source = self._prepare_voice(job, engine)
        self._check_cancel(job)

        # 3. cache lookup per segment, then for the whole result
        audio_params = engine.audio_params(params)
        seg_keys = [make_key(engine.id, engine.version, REF_PREP, job.voice_id, resolved_ref, s, audio_params)
                    for s in segments]
        # post-processing runs on cached segments at join time, so it keys the result, not the segments
        post = {k: params[k] for k in ("pause_ms", "trim_pauses", "max_pause_ms") if k in params}
        post["explicit"] = [plan.before, plan.tail_ms]
        result_key = make_key(seg_keys, post)
        result_path = self.store.result_path(result_key)
        sr = engine.sample_rate

        cached_count = sum(1 for k in seg_keys if self.store.segment_path(k).exists())
        if result_path.exists():
            self.store.hits["result"] += 1
            self._complete(job, result_path, result_key, segments, len(segments), t0, text, resolved_ref,
                           prompt_source, from_result_cache=True)
            return

        parts: List[Optional[np.ndarray]] = [None] * len(segments)
        missing: List[int] = []
        for i, k in enumerate(seg_keys):
            p = self.store.segment_path(k)
            if p.exists():
                parts[i] = audio.read_wav(p)[0]
                self.store.hits["segment"] += 1
            else:
                missing.append(i)

        # 4. render the missing sentences in batches, one progress tick per decoding step
        bs = max(1, int(params["batch_size"]))
        batches = [missing[i:i + bs] for i in range(0, len(missing), bs)]
        per_batch = max(1, engine.steps_per_batch(params))
        total_steps = per_batch * len(batches)
        done_steps = 0
        gen_t0 = time.time()

        def label(batch_no: int) -> str:
            return (f"Đang sinh {len(missing)} câu" + (f", lô {batch_no + 1}/{len(batches)}" if len(batches) > 1 else "")
                    + (f" ({cached_count} câu lấy từ cache)" if cached_count else ""))

        for b, idxs in enumerate(batches):
            step_in_batch = 0

            def on_step() -> None:
                nonlocal done_steps, step_in_batch
                if job.cancel_requested:
                    raise Cancelled()
                done_steps += 1
                step_in_batch = min(step_in_batch + 1, per_batch)
                frac = min(done_steps / total_steps, 1.0)
                rate = (time.time() - gen_t0) / max(done_steps, 1)
                eta = rate * max(total_steps - done_steps, 0)
                self._update(job, stage=label(b), progress=P_VOICE + (P_GEN - P_VOICE) * frac,
                             detail=f"bước {step_in_batch}/{per_batch}, còn khoảng {eta:.0f} giây")

            self._update(job, stage=label(b), detail=f"bước 0/{per_batch}")
            outs = engine.synthesize([segments[i] for i in idxs], voice_obj, params, on_step)
            for i, wav in zip(idxs, outs):
                parts[i] = wav
                audio.write_wav(self.store.segment_path(seg_keys[i]), wav, sr)

        # 5. join
        self._update(job, stage="Ghép các câu", progress=P_GEN, detail="")
        clips = [p for p in parts if p is not None]
        if params.get("trim_pauses"):
            clips = [audio.shorten_pauses(c, sr, int(params["max_pause_ms"])) for c in clips]
        # an explicit [nghỉ …] is exact: strip the model's own silence on both sides of it
        n = len(clips)
        for i in range(n):
            lead = bool(plan.before[i])
            trail = (i + 1 < n and bool(plan.before[i + 1])) or (i == n - 1 and plan.tail_ms > 0)
            if lead or trail:
                clips[i] = audio.strip_edges(clips[i], sr, lead=lead, trail=trail)
        gaps = [g if g is not None else (int(params["pause_ms"]) if i else 0) for i, g in enumerate(plan.before)]
        joined = audio.join_segments(clips, sr, gaps, plan.tail_ms)
        audio.write_wav(result_path, joined, sr)
        self._complete(job, result_path, result_key, segments, cached_count, t0, text, resolved_ref,
                       prompt_source, from_result_cache=False)

    def _complete(self, job: Job, path, key: str, segments: List[str], cached: int, t0: float,
                  text: str, ref_text: str, prompt_source: str, from_result_cache: bool) -> None:
        duration = audio.duration_of(path)
        elapsed = time.time() - t0
        engine = self.engines[job.engine_id]
        result = {
            "url": f"/api/audio/{key}.wav",
            "duration": round(duration, 2),
            "elapsed": round(elapsed, 2),
            "rtf": round(elapsed / duration, 3) if duration else None,
            "segments": len(segments),
            "segments_cached": cached,
            "result_cached": from_result_cache,
            "prompt_source": prompt_source,
            "text_used": text,
            "ref_text": ref_text,
        }
        self._finish(job, "done", stage="Hoàn tất", progress=1.0, detail="", result=result)
        self.store.add_history({
            "id": job.id, "created": job.created, "engine": engine.id, "engine_name": engine.name,
            "voice_id": job.voice_id, "text": job.text, "params": job.params, **result,
        })
