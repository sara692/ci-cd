import os
from pathlib import Path

import yaml

ROOT = Path(os.getenv("RAG_ROOT", Path(__file__).resolve().parents[2]))


def load_params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
