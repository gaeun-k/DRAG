import json
from pathlib import Path

from .config import SETTINGS


def load_samples(path=None):
    path = Path(path) if path else SETTINGS.data_dir / "public" / "resume_matching.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("id") or not isinstance(row.get("jobpost"), str):
                raise ValueError("공개 데이터 형식이 올바르지 않습니다. 준비 스크립트를 실행하세요.")
            items.append(row)
    return items
