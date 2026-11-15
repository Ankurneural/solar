"""Run the Kaggle CLI with every secret masked in its output.

  python tools/kaggle_cli.py competitions submissions -c filament-segmentation-2026
  python tools/kaggle_cli.py competitions submit -c filament-segmentation-2026 -f runs/X/submission.csv -m "msg"

Why: an auth error once echoed the raw access token inside a traceback. The CLI
lives in .venv-kaggle (kaggle>=1.8 needs Python 3.11+, the training venv is 3.10)
and reads ~/.kaggle/access_token (plain ASCII, no BOM/newline).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, ".venv-kaggle", "Scripts", "kaggle.exe")
TOKEN_FILE = os.path.expanduser("~/.kaggle/access_token")


def redact(text: str, token: str) -> str:
    if token:
        text = text.replace(token, "<token>")
    text = re.sub(r"Bearer\s+\S+", "Bearer <redacted>", text)
    text = re.sub(r"KGAT_[A-Za-z0-9]+", "KGAT_<redacted>", text)
    # tokens re-encoded as UTF-16 (b'K\x00G\x00...') in reprs
    return re.sub(r"(\\x00[A-Za-z0-9_]){8,}", "<redacted>", text)


def main() -> int:
    token = open(TOKEN_FILE, encoding="ascii", errors="ignore").read().strip() \
        if os.path.exists(TOKEN_FILE) else ""
    res = subprocess.run([EXE, *sys.argv[1:]], capture_output=True, text=True, errors="replace")
    sys.stdout.write(redact(res.stdout, token))
    sys.stderr.write(redact(res.stderr, token))
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
