"""공개 합성 사례만 다운로드. 개인 문서 DB에 넣지 않습니다."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from career_twin.config import SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--role", default="", help="공고 본문의 키워드. 예: LLM. 기본은 모든 직무")
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--revision", help="재현용 Hugging Face dataset commit SHA")
    args = parser.parse_args()
    if args.limit < 1 or args.max_scan < args.limit:
        parser.error("limit은 양수이며 max-scan 이하여야 합니다.")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = args.revision or HfApi().dataset_info(SETTINGS.dataset_id, timeout=30).sha
    stream = load_dataset(SETTINGS.dataset_id, split="train", revision=revision, streaming=True)
    selected, scanned = [], 0
    for row in stream.take(args.max_scan):
        scanned += 1
        if args.role and args.role.casefold() not in str(row.get("jobpost", "")).casefold():
            continue
        if not all(isinstance(row.get(key), str) for key in ("jobpost", "resume", "selfintro")):
            continue
        stable = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
        selected.append({"id": stable, "source": SETTINGS.dataset_id, "synthetic": True,
                         **{key: row.get(key) for key in ("jobpost", "resume", "selfintro",
                              "evaluation", "total_score", "resume_score", "selfintro_score")}})
        if len(selected) >= args.limit:
            break
    if not selected:
        raise SystemExit("검색 조건에 맞는 사례가 없습니다. --role을 바꾸거나 --max-scan을 늘리세요.")
    folder = SETTINGS.data_dir / "public"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "resume_matching.jsonl"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    temporary.replace(target)
    manifest = {"dataset": SETTINGS.dataset_id, "revision": revision, "split": "train",
        "selection": "first matching records, streaming; no random sampling", "role": args.role,
        "scanned": scanned, "count": len(selected), "synthetic": True,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "note": "교육용 합성 사례. 실제 공고·합격자·합격 확률이 아님."}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {target}\n실제 행 수: {len(selected)}, 스캔: {scanned}, revision: {revision}")
    if len(selected) < args.limit:
        print("요청 수보다 적습니다. 다음 실행에서 max-scan을 늘릴 수 있습니다.")


if __name__ == "__main__":
    main()
