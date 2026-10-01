import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def _path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.strip().lower().encode()).hexdigest()[:16]
    return CACHE_DIR / f"{namespace}_{digest}.json"


def get(namespace: str, key: str):
    path = _path(namespace, key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def put(namespace: str, key: str, value) -> None:
    _path(namespace, key).write_text(
        json.dumps(value, ensure_ascii=False), encoding="utf-8"
    )