"""Vietnamese text normalization for TTS.

OmniVoice's own `normalize_text=True` uses WeTextProcessing, which targets Chinese and English, so numbers,
units and abbreviations in Vietnamese text are read badly (the notebook's experiment 5 shows it). This module
rewrites them as spoken Vietnamese before synthesis. It is rule based and deliberately conservative: anything
it does not recognise is left untouched.
"""
from __future__ import annotations

import re
from typing import Callable, List, Tuple

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
SCALES = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]


def _read_two(n: int, full: bool) -> str:
    """0..99. `full` means a hundreds digit precedes, so 5 reads 'linh năm'."""
    tens, unit = divmod(n, 10)
    if tens == 0:
        if unit == 0:
            return ""
        return ("linh " if full else "") + DIGITS[unit]
    if tens == 1:
        word = "mười"
        if unit == 5:
            return word + " lăm"
        return word + ("" if unit == 0 else " " + DIGITS[unit])
    word = DIGITS[tens] + " mươi"
    if unit == 0:
        return word
    if unit == 1:
        return word + " mốt"
    if unit == 4:
        return word + " tư"
    if unit == 5:
        return word + " lăm"
    return word + " " + DIGITS[unit]


def _read_three(n: int, full: bool) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0 and not full:
        return _read_two(rest, False)
    head = DIGITS[hundreds] + " trăm"
    tail = _read_two(rest, True)
    return head + (" " + tail if tail else "")


def number_to_words(n: int) -> str:
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_words(-n)
    groups: List[int] = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)
    if len(groups) > len(SCALES):
        return " ".join(DIGITS[int(d)] for d in str(n))
    words: List[str] = []
    top = len(groups) - 1
    for idx in range(top, -1, -1):
        g = groups[idx]
        if g == 0:
            continue
        text = _read_three(g, full=idx != top)
        words.append(text + (" " + SCALES[idx] if SCALES[idx] else ""))
    return " ".join(words)


def read_digits(s: str) -> str:
    return " ".join(DIGITS[int(c)] for c in s if c.isdigit())


def _int_words(s: str) -> str:
    s = s.replace(".", "").replace(",", "")
    if len(s) > 1 and s.startswith("0"):
        return read_digits(s)  # phone numbers, codes
    return number_to_words(int(s))


def _decimal_words(whole: str, frac: str) -> str:
    frac_words = read_digits(frac) if frac.startswith("0") or len(frac) > 2 else number_to_words(int(frac))
    return f"{_int_words(whole)} phẩy {frac_words}"


ABBREVIATIONS: List[Tuple[str, str]] = [
    (r"\bTP\.?\s?HCM\b|\bTPHCM\b|\bTp\.?\s?HCM\b", "thành phố Hồ Chí Minh"),
    (r"\bTP\.(?=\s)", "thành phố"),
    (r"\bPGS\.\s?TS\.", "phó giáo sư tiến sĩ"),
    (r"\bGS\.\s?TS\.", "giáo sư tiến sĩ"),
    (r"\bPGS\.", "phó giáo sư"),
    (r"\bGS\.", "giáo sư"),
    (r"\bThS\.", "thạc sĩ"),
    (r"\bTS\.", "tiến sĩ"),
    (r"\bĐH\b", "đại học"),
    (r"\bKHMT\b", "khoa học máy tính"),
    (r"\bv\.v\.?", "vân vân"),
    (r"\bet al\.", "và cộng sự"),
    (r"\bvs\.", "so với"),
    (r"\be\.g\.", "ví dụ"),
]

UNITS = {
    "%": "phần trăm", "°C": "độ xê", "°F": "độ ép", "°": "độ",
    "km/h": "ki lô mét trên giờ", "km": "ki lô mét", "cm": "xen ti mét", "mm": "mi li mét", "m2": "mét vuông",
    "m²": "mét vuông", "m": "mét", "kg": "ki lô gam", "g": "gam", "ml": "mi li lít", "l": "lít",
    "GB": "ghi ga bai", "MB": "mê ga bai", "kHz": "ki lô héc", "Hz": "héc", "ms": "mi li giây", "s": "giây",
    "k": "nghìn", "K": "nghìn", "tr": "triệu", "đ": "đồng", "VND": "đồng", "VNĐ": "đồng", "USD": "đô la Mỹ", "$": "đô la",
}
_UNIT_PATTERN = "|".join(sorted((re.escape(u) for u in UNITS), key=len, reverse=True))
_NUM = r"\d{1,3}(?:\.\d{3})+(?!\d)|\d+(?:[.,]\d+)?"


def _num_words(token: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):  # 1.000.000 thousands separators
        return _int_words(token)
    m = re.fullmatch(r"(\d+)[.,](\d+)", token)
    if m:
        return _decimal_words(m.group(1), m.group(2))
    return _int_words(token)


def _sub(pattern: str, fn: Callable[[re.Match], str], text: str, flags: int = 0) -> str:
    return re.sub(pattern, fn, text, flags=flags)


def normalize_vi(text: str) -> str:
    if not text:
        return text
    t = text.replace(" ", " ")

    for pat, rep in ABBREVIATIONS:
        t = re.sub(pat, rep, t)

    # dates: 15/11/2026, 11/2026, 15/11
    t = _sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
             lambda m: f"ngày {_int_words(m.group(1))} tháng {_int_words(m.group(2))} năm {number_to_words(int(m.group(3)))}", t)
    t = _sub(r"\b(\d{1,2})/(\d{4})\b",
             lambda m: f"tháng {_int_words(m.group(1))} năm {number_to_words(int(m.group(2)))}", t)
    # times: 10:30, 10h30, 10h
    t = _sub(r"\b(\d{1,2})[:h](\d{2})\b",
             lambda m: f"{number_to_words(int(m.group(1)))} giờ {number_to_words(int(m.group(2)))} phút", t)
    t = _sub(r"\b(\d{1,2})h\b", lambda m: f"{number_to_words(int(m.group(1)))} giờ", t)
    # ranges: 3-10, 3–10 giây
    t = _sub(rf"\b({_NUM})\s?[-–]\s?({_NUM})\b", lambda m: f"{_num_words(m.group(1))} đến {_num_words(m.group(2))}", t)
    # money prefix: $5
    t = _sub(rf"\$\s?({_NUM})", lambda m: f"{_num_words(m.group(1))} đô la", t)
    # number + unit
    t = _sub(rf"({_NUM})\s?({_UNIT_PATTERN})(?![\wÀ-ỹ])", lambda m: f"{_num_words(m.group(1))} {UNITS[m.group(2)]}", t)
    # ordinals: thứ 2 -> thứ hai (thứ 4 -> thứ tư)
    t = _sub(r"\bthứ (\d)\b", lambda m: "thứ tư" if m.group(1) == "4" else f"thứ {_int_words(m.group(1))}", t, re.IGNORECASE)
    # plain numbers
    t = _sub(rf"(?<![\w])(?:{_NUM})(?![\w])", lambda m: _num_words(m.group(0)), t)
    # symbols
    t = t.replace("&", " và ").replace("+", " cộng ").replace("=", " bằng ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ([,.;:!?])", r"\1", t)
    return t.strip()


if __name__ == "__main__":
    samples = [
        "Nhiệt độ hôm nay là 30°C, độ ẩm 80%, giá vé 15k tại TP.HCM.",
        "WER giảm từ 2.84 xuống 2.41, SIM-o tăng từ 0.62 lên 0.66 (Chen et al., 2025).",
        "Deadline nộp bài: 15/11/2026 lúc 10h30, mỗi nhóm 2-3 thành viên.",
        "Dân số khoảng 1.000.005 người, năm 2021 có 105 trường.",
    ]
    for s in samples:
        print(s, "\n  ->", normalize_vi(s))
