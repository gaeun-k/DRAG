from io import BytesIO
from pathlib import Path
import re

from .config import SETTINGS


def split_text(text, page=None, section="", size=750, overlap=100):
    """페이지·섹션 경계를 넘지 않는 문자 청크. 원문 인용을 위해 문자열을 보존합니다."""
    if size <= overlap or overlap < 0:
        raise ValueError("청크 크기는 겹침보다 커야 합니다.")
    result = []
    for start in range(0, len(text), size - overlap):
        chunk = text[start:start + size].strip()
        if chunk:
            result.append({"text": chunk, "page": page, "section": section})
        if start + size >= len(text):
            break
    return result


def extract(filename, content):
    if len(content) > SETTINGS.max_file_bytes:
        raise ValueError("파일은 20MB 이하로 올려주세요.")
    suffix = Path(filename).suffix.lower()
    warnings, chunks = [], []
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("암호가 걸린 PDF는 지원하지 않습니다.")
        if len(reader.pages) > SETTINGS.max_pdf_pages:
            raise ValueError("PDF는 150쪽 이하로 나눠 올려주세요.")
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                warnings.append(f"{number}쪽: 텍스트 없음 (스캔/OCR 미지원)")
            chunks += split_text(text, page=number)
    elif suffix in {".md", ".txt"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("TXT/MD는 UTF-8 인코딩으로 저장하세요.") from exc
        section, block = "본문", []
        for line in text.splitlines(keepends=True):
            if suffix == ".md" and re.match(r"^#{1,6}\s+", line):
                chunks += split_text("".join(block), section=section)
                section, block = line.strip().lstrip("#").strip(), [line]
            else:
                block.append(line)
        chunks += split_text("".join(block), section=section)
    else:
        raise ValueError("PDF, Markdown, TXT 파일만 지원합니다.")
    if not chunks:
        raise ValueError("추출된 텍스트가 없습니다. 스캔 PDF는 텍스트 자료를 함께 올려주세요.")
    return chunks, warnings


def render_page(path, page):
    """PDF의 벡터 도형까지 보존하기 위해 그림 추출 대신 페이지를 렌더링합니다."""
    import fitz
    with fitz.open(path) as doc:
        if not 1 <= page <= len(doc):
            raise ValueError("페이지 번호가 올바르지 않습니다.")
        current = doc[page - 1]
        scale = min(2.0, 1280 / max(current.rect.width, current.rect.height))
        return current.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
