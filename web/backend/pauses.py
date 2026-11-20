"""Explicit pauses written in the text: [nghỉ 1.5s], [nghỉ 800ms], [nghỉ 2 giây], [nghỉ] (1 s), [pause 1s].

The text is cut at every marker before normalization (so "1.5s" is never read aloud), each piece is split
into sentences as usual, and the marker becomes an exact stretch of silence at join time. Sentences never
merge across a marker, so the cache still works per sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .segment import split_segments
from .textnorm_vi import normalize_vi

PAUSE_RE = re.compile(
    r"\[\s*(?:nghỉ|nghi|ngừng|dừng|pause)\s*(?:(\d+(?:[.,]\d+)?)\s*(ms|s|giây|giay|sec)?)?\s*\]",
    re.IGNORECASE,
)
DEFAULT_MS = 1000
MIN_MS, MAX_MS = 50, 10000


def marker_ms(m: re.Match) -> int:
    num, unit = m.group(1), (m.group(2) or "s").lower()
    if num is None:
        return DEFAULT_MS
    value = float(num.replace(",", "."))
    ms = value if unit == "ms" else value * 1000
    return int(max(MIN_MS, min(MAX_MS, round(ms))))


@dataclass
class TextPlan:
    segments: List[str] = field(default_factory=list)
    # explicit silence before segment i in ms; None means "use the normal pause between sentences"
    before: List[Optional[int]] = field(default_factory=list)
    tail_ms: int = 0  # explicit silence after the last segment

    def display(self) -> str:
        out = []
        for seg, gap in zip(self.segments, self.before):
            if gap:
                out.append(f"[nghỉ {gap / 1000:g}s]")
            out.append(seg)
        if self.tail_ms:
            out.append(f"[nghỉ {self.tail_ms / 1000:g}s]")
        return " ".join(out)

    def explicit_ms(self) -> int:
        return sum(g for g in self.before if g) + self.tail_ms


def plan_text(text: str, normalize: bool = True, split: bool = True) -> TextPlan:
    plan = TextPlan()
    pending: Optional[int] = None

    def add_block(chunk: str) -> None:
        nonlocal pending
        chunk = normalize_vi(chunk) if normalize else chunk
        for seg in split_segments(chunk, enabled=split):
            plan.segments.append(seg)
            plan.before.append(pending)
            pending = None

    pos = 0
    for m in PAUSE_RE.finditer(text):
        add_block(text[pos:m.start()])
        pending = (pending or 0) + marker_ms(m)
        pos = m.end()
    add_block(text[pos:])
    plan.tail_ms = pending or 0
    return plan
