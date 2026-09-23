"""파일 제한과 무료 실행 정책. API 키가 있어도 유료 호출은 허용하지 않는다."""
FREE_ONLY = True
MAX_FILE_MB = 30
MAX_TOTAL_MB = 1024
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_TOTAL_BYTES = MAX_TOTAL_MB * 1024 * 1024
MAX_TEXT_CHARS = 20_000_000
MAX_CHUNKS = 100_000
INDEX_BATCH_SIZE = 256


def require_allowed_mode(mode: str) -> None:
    if FREE_ONLY and mode not in {"demo", "local"}:
        raise ValueError("무료 전용 설정입니다. 비용 방지를 위해 OpenAI API 호출을 차단했습니다.")


def validate_upload_sizes(sizes) -> None:
    total = 0
    for size in sizes:
        if size > MAX_FILE_BYTES:
            raise ValueError(f"파일 하나는 {MAX_FILE_MB}MB 이하여야 합니다.")
        total += size
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"문서 전체 크기는 {MAX_TOTAL_MB}MB 이하여야 합니다.")
