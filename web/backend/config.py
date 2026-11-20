"""Runtime settings, read from environment variables so the same code runs on a laptop and on Kaggle."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
CACHE_DIR = Path(os.environ.get("VONG_CACHE_DIR", str(ROOT / "cache")))

# "1" = always use the mock engine (no GPU needed, for UI work),
# "0" = always load OmniVoice (fail loudly if it cannot),
# "auto" = OmniVoice when torch + omnivoice import and a GPU (CUDA or Apple MPS) is present, mock otherwise.
MOCK = os.environ.get("VONG_MOCK", "auto").strip().lower()

# "auto" picks cuda:0, then Apple Silicon "mps", then "cpu". Or set e.g. VONG_DEVICE=cuda:1.
DEVICE = os.environ.get("VONG_DEVICE", "auto")
# "auto" = float16 on CUDA, float32 on MPS and CPU. Or "float16" / "float32" / "bfloat16".
DTYPE = os.environ.get("VONG_DTYPE", "auto")
OMNIVOICE_REPO = os.environ.get("VONG_OMNIVOICE_REPO", "k2-fsa/OmniVoice")

SAMPLE_RATE = 24000
MAX_UPLOAD_MB = int(os.environ.get("VONG_MAX_UPLOAD_MB", "25"))
MAX_REF_SECONDS = float(os.environ.get("VONG_MAX_REF_SECONDS", "30"))
MIN_REF_SECONDS = 1.0
MAX_TEXT_CHARS = int(os.environ.get("VONG_MAX_TEXT_CHARS", "6000"))
HISTORY_LIMIT = 100
PROMPT_MEMORY_SLOTS = 16
