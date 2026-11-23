"""HTTP API and static site. Run with:  python -m backend.serve  (or uvicorn backend.app:app)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import audio, config
from .engines import build_engines
from .jobs import TERMINAL, JobManager
from .pauses import plan_text
from .presets import seed_presets
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("vong")

app = FastAPI(title="Vọng, studio nhân bản giọng nói tiếng Việt")
store = Store(config.CACHE_DIR)
engines = build_engines()
jobs = JobManager(engines, store)
BOOT = time.time()


def _load_engines() -> None:
    for engine in engines.values():
        if engine.placeholder:
            continue
        engine.state = "loading"
        t0 = time.time()
        try:
            engine.load()
            engine.state = "ready"
            log.info("%s ready in %.1fs on %s", engine.name, time.time() - t0, engine.device)
        except Exception as exc:  # keep serving the UI; report the error in /api/status
            engine.state = "error"
            engine.error = str(exc)
            log.exception("failed to load %s", engine.name)


@app.on_event("startup")
def _startup() -> None:
    log.info("seeded %d preset voices", seed_presets(store))
    threading.Thread(target=_load_engines, name="engine-loader", daemon=True).start()


# ----- status and models ----------------------------------------------------------
@app.get("/api/status")
def status() -> Dict[str, Any]:
    return {
        "uptime": round(time.time() - BOOT, 1),
        "mock": any(e.version.endswith("-mock") for e in engines.values()),
        "engines": {k: {"state": e.state, "device": e.device, "error": e.error} for k, e in engines.items()},
        "cache": store.stats(),
        "queue": sum(1 for j in jobs.jobs.values() if j.status == "queued"),
        "ffmpeg": audio.ffmpeg_available(),
        "limits": {"max_text": config.MAX_TEXT_CHARS, "max_ref_seconds": config.MAX_REF_SECONDS,
                   "max_upload_mb": config.MAX_UPLOAD_MB},
    }


@app.get("/api/models")
def models() -> Dict[str, Any]:
    return {"models": [e.info() for e in engines.values()]}


# ----- voices ---------------------------------------------------------------------
@app.get("/api/voices")
def list_voices() -> Dict[str, Any]:
    return {"voices": store.list_voices()}


@app.post("/api/voices")
async def upload_voice(file: UploadFile = File(...), name: str = Form(""), source: str = Form("upload")) -> Dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(400, "File rỗng.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File lớn hơn {config.MAX_UPLOAD_MB} MB.")
    voice_id = hashlib.sha256(data).hexdigest()[:20]
    try:
        meta = store.voice_meta(voice_id)
        meta["reused"] = True
        return meta
    except KeyError:
        pass
    suffix = Path(file.filename or "audio").suffix[:8] or ".bin"
    vdir = store.voice_dir(voice_id)
    vdir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        audio.convert_to_wav(tmp_path, vdir / "ref.wav", config.SAMPLE_RATE, config.MAX_REF_SECONDS)
    except audio.AudioError as exc:
        store.delete_voice(voice_id)
        raise HTTPException(400, str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)
    duration = audio.duration_of(vdir / "ref.wav")
    if duration < config.MIN_REF_SECONDS:
        store.delete_voice(voice_id)
        raise HTTPException(400, "Audio mẫu ngắn hơn 1 giây. Hãy dùng đoạn 3 đến 10 giây.")
    meta = {
        "id": voice_id,
        "name": (name or Path(file.filename or "").stem or "Giọng mẫu").strip()[:60],
        "source": "record" if source == "record" else "upload",
        "original": file.filename,
        "duration": round(duration, 2),
        "truncated": duration >= config.MAX_REF_SECONDS - 0.05,
        "created": time.time(),
        "transcript": "",
        "reused": False,
    }
    store.save_voice_meta(voice_id, meta)
    return meta


@app.get("/api/voices/{voice_id}/audio")
def voice_audio(voice_id: str):
    try:
        path = store.voice_dir(voice_id) / "ref.wav"
    except KeyError:
        raise HTTPException(404, "Không có giọng này.")
    if not path.exists():
        raise HTTPException(404, "Không có giọng này.")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str) -> Dict[str, Any]:
    try:
        meta = store.voice_meta(voice_id)
    except KeyError:
        raise HTTPException(404, "Không có giọng này.")
    if meta.get("preset"):
        raise HTTPException(403, "Không xoá được giọng có sẵn.")
    store.delete_voice(voice_id)
    return {"ok": True}


# ----- text -----------------------------------------------------------------------
class NormalizeIn(BaseModel):
    text: str
    normalize: bool = True
    split: bool = True


@app.post("/api/normalize")
def normalize(body: NormalizeIn) -> Dict[str, Any]:
    plan = plan_text(body.text[: config.MAX_TEXT_CHARS], normalize=body.normalize, split=body.split)
    return {"text": plan.display(), "segments": plan.segments, "pauses_before": plan.before,
            "tail_ms": plan.tail_ms, "explicit_ms": plan.explicit_ms()}


# ----- jobs -----------------------------------------------------------------------
class JobIn(BaseModel):
    engine: str
    voice_id: str
    text: str
    ref_text: str = ""
    params: Dict[str, Any] = {}


@app.post("/api/jobs")
def create_job(body: JobIn) -> Dict[str, Any]:
    engine = engines.get(body.engine)
    if engine is None:
        raise HTTPException(404, "Không có mô hình này.")
    if engine.placeholder:
        raise HTTPException(409, f"{engine.name} chưa được gắn vào.")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Hãy nhập văn bản cần đọc.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(400, f"Văn bản dài hơn {config.MAX_TEXT_CHARS} ký tự.")
    if engine.needs_voice:
        try:
            store.voice_meta(body.voice_id)
        except KeyError:
            raise HTTPException(400, "Chưa có giọng mẫu. Hãy tải lên hoặc thu âm trước.")
    params = engine.resolve_params(body.params)
    job = jobs.submit(engine.id, body.voice_id, text, body.ref_text[:1000], params)
    return job.public()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job.")
    return job.public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    job = jobs.cancel(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job.")
    return job.public()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job.")

    async def stream():
        last = -1
        idle = 0.0
        while True:
            if job.version != last:
                last = job.version
                idle = 0.0
                yield f"data: {json.dumps(job.public(), ensure_ascii=False)}\n\n"
                if job.status in TERMINAL:
                    break
            elif idle > 10:
                idle = 0.0
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.1)
            idle += 0.1

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/audio/{name}")
def result_audio(name: str):
    key = name[:-4] if name.endswith(".wav") else name
    if not key.isalnum():
        raise HTTPException(404, "Không có file.")
    path = store.result_path(key)
    if not path.exists():
        raise HTTPException(404, "File đã bị xoá khỏi cache.")
    return FileResponse(path, media_type="audio/wav", filename=f"vong-{key[:8]}.wav",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/history")
def history() -> Dict[str, Any]:
    items = [h for h in store.history() if store.result_path(h["url"].rsplit("/", 1)[-1][:-4]).exists()]
    return {"items": items}


class ClearIn(BaseModel):
    keep_voices: bool = True


@app.post("/api/cache/clear")
def clear_cache(body: ClearIn) -> Dict[str, Any]:
    if any(j.status == "running" for j in jobs.jobs.values()):
        raise HTTPException(409, "Đang có job chạy, hãy đợi xong rồi xoá cache.")
    store.clear(keep_voices=body.keep_voices)
    return {"ok": True, "cache": store.stats()}


@app.exception_handler(HTTPException)
async def http_error(_request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


class SiteFiles(StaticFiles):
    """The UI changes often during the project: make browsers revalidate (ETag) instead of reusing stale CSS/JS."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if not path.startswith("media/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", SiteFiles(directory=str(config.FRONTEND_DIR), html=True), name="site")
