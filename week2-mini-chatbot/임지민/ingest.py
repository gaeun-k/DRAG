"""Local documents, bounded public web fetch, PDF tables and optional vision."""
import csv
import hashlib
import io
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from langchain_core.documents import Document

from rag import load_document
from runtime_policy import MAX_FILE_BYTES

MEDIA = Path(__file__).resolve().parent / ".local-ai/media"


def validate_url(url):
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("공개된 http/https 주소를 입력하세요.")
    if parts.port not in {None, 80, 443}:
        raise ValueError("일반 웹 포트만 지원합니다.")
    addresses = socket.getaddrinfo(parts.hostname, parts.port or 443)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("내부 네트워크 주소는 수집할 수 없습니다.")


def load_url(url):
    current = url.strip()
    with requests.Session() as session:
        session.trust_env = False
        for _ in range(6):
            validate_url(current)
            with session.get(current, stream=True, allow_redirects=False, timeout=(5, 25),
                             headers={"User-Agent": "RAGStudy/1.0"}) as response:
                if response.is_redirect:
                    current = urljoin(current, response.headers["Location"])
                    continue
                response.raise_for_status()
                data = bytearray()
                for part in response.iter_content(65536):
                    data.extend(part)
                    if len(data) > min(MAX_FILE_BYTES, 5 * 1024 * 1024):
                        raise ValueError("웹 문서는 5MB 이하만 수집합니다. 큰 자료는 파일로 올려주세요.")
                content_type = response.headers.get("Content-Type", "").lower()
                if "html" not in content_type and "text/plain" not in content_type:
                    raise ValueError("웹 주소는 HTML/TXT 문서만 지원합니다. PDF는 다운로드 후 올려주세요.")
                encoding = response.encoding if response.encoding and response.encoding.lower() != "iso-8859-1" else "utf-8"
                text = data.decode(encoding, errors="replace")
                docs = load_document("web.html" if "html" in content_type else "web.txt", text.encode())
                for doc in docs:
                    doc.metadata.update(source=current, url=current)
                return docs
    raise ValueError("주소 이동이 너무 많습니다.")


def load_file(name, data, vision=False, visual_pages=3, describe=None):
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("파일 하나는 30MB 이하여야 합니다.")
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        return [Document(page_content="\n".join(f"{k}: {v}" for k, v in row.items()),
                         metadata={"source": Path(name).name, "row": i, "kind": "table"})
                for i, row in enumerate(rows, 1)]
    if suffix == ".py":
        return load_document(Path(name).stem + ".txt", data)
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        if not vision or describe is None:
            raise ValueError("이미지는 로컬 AI 모드에서 이미지·표 분석을 켜주세요.")
        from PIL import Image
        image = Image.open(io.BytesIO(data))
        image.thumbnail((1400, 1400))
        MEDIA.mkdir(parents=True, exist_ok=True)
        path = MEDIA / (hashlib.sha256(data).hexdigest() + ".png")
        image.convert("RGB").save(path)
        return [Document(page_content=describe(path), metadata={"source": Path(name).name,
                         "kind": "image", "image_path": str(path), "is_description": True})]
    if suffix != ".pdf":
        return load_document(name, data)
    import pymupdf
    docs = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("암호화된 PDF는 암호를 해제해주세요.")
        for i, page in enumerate(pdf):
            meta = {"source": Path(name).name, "page": i + 1, "kind": "text"}
            text = page.get_text(sort=True).strip()
            if text:
                docs.append(Document(page_content=text, metadata=meta))
            tables = page.find_tables()
            for table in tables.tables:
                rows = table.extract()
                body = "\n".join(" | ".join(str(value or "") for value in row) for row in rows)
                if body.strip():
                    docs.append(Document(page_content=body, metadata={**meta, "kind": "table"}))
            if vision and i < visual_pages:
                MEDIA.mkdir(parents=True, exist_ok=True)
                path = MEDIA / (hashlib.sha256(data).hexdigest() + f"_p{i+1}.png")
                if not path.exists():
                    scale = min(1.5, 1400 / max(page.rect.width, page.rect.height))
                    page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).save(path)
                docs.append(Document(page_content=describe(path), metadata={**meta,
                    "kind": "image", "image_path": str(path), "is_description": True}))
    if not docs:
        raise ValueError("텍스트를 읽을 수 없습니다. 로컬 AI의 이미지·표 분석을 켜서 스캔 페이지를 읽어보세요.")
    return docs
