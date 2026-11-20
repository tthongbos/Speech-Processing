"""Entry point:  python -m backend.serve [--host 0.0.0.0] [--port 8000] [--mock]"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Vọng TTS studio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--mock", action="store_true", help="dùng engine mô phỏng, không cần GPU")
    args = parser.parse_args()
    if args.mock:
        os.environ["VONG_MOCK"] = "1"

    import uvicorn

    uvicorn.run("backend.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
