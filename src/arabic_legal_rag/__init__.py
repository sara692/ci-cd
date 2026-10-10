import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("RAG_ROOT") or Path(__file__).resolve().parents[2])
load_dotenv(ROOT / ".env", override=True)
