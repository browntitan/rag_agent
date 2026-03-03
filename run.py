import sys
from pathlib import Path

# Ensure `src/` is on PYTHONPATH when running from the repo root.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_chatbot.cli import app

if __name__ == "__main__":
    app()
