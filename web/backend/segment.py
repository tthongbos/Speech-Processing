"""Split long text into sentence-sized segments.

Short segments do three jobs: they keep every item under OmniVoice's 30 s chunking threshold, they let one
batched call render many sentences at once, and they make the per-sentence cache useful (edit one sentence,
regenerate only that one).
"""
from __future__ import annotations

import re
from typing import List

_SENTENCE_END = re.compile(r"(?<=[.!?…;])\s+|\n+")
_SOFT_BREAK = re.compile(r"(?<=[,:])\s+")


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_segments(text: str, max_chars: int = 220, min_chars: int = 40, enabled: bool = True) -> List[str]:
    text = _clean(text)
    if not text:
        return []
    if not enabled:
        return [text]
    pieces: List[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = _clean(sentence)
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        # a very long sentence: break at commas, then hard-wrap on spaces as a last resort
        buf = ""
        for part in _SOFT_BREAK.split(sentence):
            if buf and len(buf) + 1 + len(part) > max_chars:
                pieces.append(buf)
                buf = part
            else:
                buf = f"{buf} {part}".strip()
            while len(buf) > max_chars:
                cut = buf.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(buf[:cut].strip())
                buf = buf[cut:].strip()
        if buf:
            pieces.append(buf)

    # merge fragments that are too short to carry natural prosody on their own
    merged: List[str] = []
    for piece in pieces:
        if merged and (len(merged[-1]) < min_chars or len(piece) < min_chars // 2) \
                and len(merged[-1]) + 1 + len(piece) <= max_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged
