"""Download the embedding model into backend/models/ so deployments don't fetch it at runtime.

Run as the build step on Vercel (see vercel.json), or locally:  python scripts/fetch_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer  # noqa: E402

from app.config import get_settings  # noqa: E402

name = get_settings().embedding_model
target = ROOT / "models" / name.rsplit("/", 1)[-1]
SentenceTransformer(name, device="cpu").save(str(target))
print(f"Saved {name} to {target}")
