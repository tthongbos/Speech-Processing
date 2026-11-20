"""Audio I/O without heavy dependencies: ffmpeg for decoding any format, the stdlib wave module for WAV."""
from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class AudioError(Exception):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_to_wav(src: Path, dst: Path, sample_rate: int, max_seconds: Optional[float] = None) -> None:
    """Decode m4a, mp3, wav, ogg, flac, webm, aac... into mono 16-bit PCM WAV at `sample_rate`."""
    if not ffmpeg_available():
        raise AudioError("Máy chủ chưa cài ffmpeg nên không giải mã được file âm thanh.")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-sample_fmt", "s16"]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.2f}"]
    cmd.append(str(dst))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size < 100:
        raise AudioError("Không đọc được file âm thanh này. Hãy thử m4a, mp3, wav, ogg hoặc flac.")


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        raw = wf.readframes(n)
    if width != 2:
        raise AudioError("Chỉ hỗ trợ WAV 16-bit trong bộ nhớ đệm.")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.write_bytes(wav_bytes(audio, sample_rate))


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def duration_of(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _silence_mask(a: np.ndarray, sample_rate: int, frame_ms: int = 20):
    """Per-frame silence flags with a threshold adapted to the clip, or None when the clip has no clear
    speech/silence contrast. The threshold sits 30% of the way from the noise floor (10th percentile) to
    the speech level (95th percentile); short quiet blips inside a pause count as silence."""
    f = max(1, sample_rate * frame_ms // 1000)
    n = len(a) // f
    if n < 10:
        return None
    rms = np.sqrt((a[: n * f].reshape(n, f) ** 2).mean(axis=1) + 1e-12)
    smooth = np.convolve(rms, np.ones(5) / 5, mode="same")  # 100 ms window, ignores single clicks
    level = 20 * np.log10(np.maximum(smooth, 1e-6))
    floor, speech = np.percentile(level, 10), np.percentile(level, 95)
    if speech - floor < 10:
        return None
    silent = level < floor + 0.3 * (speech - floor)

    # a breath or a noise blip (short AND quiet) between two silent stretches is part of the pause;
    # a short but louder island is a word ("bạn" peaks around -31 dB, breaths around -37 dB) and stays
    island = max(1, 160 // frame_ms)
    blip_ceiling = floor + 0.45 * (speech - floor)
    i = 0
    while i < n:
        if silent[i]:
            i += 1
            continue
        j = i
        while j < n and not silent[j]:
            j += 1
        if 0 < i and j < n and j - i <= island and level[i:j].max() < blip_ceiling:
            silent[i:j] = True
        i = j
    return silent, f, n


def strip_edges(audio: np.ndarray, sample_rate: int, lead: bool = True, trail: bool = True,
                keep_ms: int = 20) -> np.ndarray:
    """Remove leading and/or trailing silence, so an explicit pause next to it is exactly as long as asked."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    mask = _silence_mask(a, sample_rate)
    if mask is None:
        return a
    silent, f, n = mask
    voiced = np.where(~silent)[0]
    if voiced.size == 0:
        return a
    keep = max(0, keep_ms * sample_rate // 1000)
    start = max(0, voiced[0] * f - keep) if lead else 0
    end = min(len(a), (voiced[-1] + 1) * f + keep) if trail else len(a)
    return a[start:end]


def shorten_pauses(audio: np.ndarray, sample_rate: int, max_pause_ms: int = 400, edge_ms: int = 100,
                   frame_ms: int = 20) -> np.ndarray:
    """Cut pauses longer than `max_pause_ms` down to that length, and trim edges to `edge_ms`.

    OmniVoice's own post-processing calls a frame silent only below a fixed -50 dBFS. A reference clip
    recorded in a room with background noise (around -45 dBFS) teaches the model to "speak" that noise,
    so a long pause sits at about -42 dBFS and survives. `_silence_mask` adapts the threshold per clip.
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    mask = _silence_mask(a, sample_rate, frame_ms)
    if mask is None:
        return a
    silent, f, n = mask

    keep = np.ones(n, dtype=bool)
    max_frames = max(1, max_pause_ms // frame_ms)
    edge_frames = max(0, edge_ms // frame_ms)
    i = 0
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        run = j - i
        if i == 0:  # leading silence
            keep[: max(0, run - edge_frames)] = False
        elif j == n:  # trailing silence
            keep[i + edge_frames:] = False
        elif run > max_frames:  # middle pause: keep the two ends, drop the middle
            head = max_frames // 2
            keep[i + head: j - (max_frames - head)] = False
        i = j
    if keep.all():
        return a

    # rebuild from kept frame runs, with a 10 ms crossfade at every cut
    xf = max(1, sample_rate // 100)
    pieces = []
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < n and keep[j]:
            j += 1
        end = j * f if j < n else len(a)
        pieces.append(a[i * f: end].copy())
        i = j
    out = pieces[0]
    for p in pieces[1:]:
        if len(out) > xf and len(p) > xf:
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            blended = out[-xf:] * ramp[::-1] + p[:xf] * ramp
            out = np.concatenate([out[:-xf], blended, p[xf:]])
        else:
            out = np.concatenate([out, p])
    return out


def join_segments(parts: List[np.ndarray], sample_rate: int, gaps_ms: List[int], tail_ms: int = 0,
                  fade_ms: int = 8) -> np.ndarray:
    """Concatenate sentence clips. `gaps_ms[i]` is the silence placed before clip i (0 for none), `tail_ms`
    the silence after the last one. A few ms of fade at each clip edge removes clicks."""
    if not parts:
        return np.zeros(int(sample_rate * tail_ms / 1000), dtype=np.float32)
    fade = max(1, int(sample_rate * fade_ms / 1000))
    out: List[np.ndarray] = []
    for clip_in, gap in zip(parts, gaps_ms):
        clip = np.asarray(clip_in, dtype=np.float32).reshape(-1).copy()
        if clip.size > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        if gap > 0:
            out.append(np.zeros(int(sample_rate * gap / 1000), dtype=np.float32))
        out.append(clip)
    if tail_ms > 0:
        out.append(np.zeros(int(sample_rate * tail_ms / 1000), dtype=np.float32))
    return np.concatenate(out)
