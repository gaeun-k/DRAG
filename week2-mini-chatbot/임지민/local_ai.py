"""Only loopback Ollama and explicitly installed local models; never a paid API."""
import base64
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import requests
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen3.5:2b"
EMBED_MODEL = "embeddinggemma:latest"
ALLOWED_MODELS = {CHAT_MODEL, EMBED_MODEL}
for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
    os.environ[key] = "false"


def call_api(endpoint, payload=None, timeout=600):
    if endpoint not in {"tags", "show", "chat", "embed"}:
        raise ValueError("허용되지 않은 로컬 API입니다.")
    if payload and payload.get("model") not in ALLOWED_MODELS:
        raise ValueError("무료 로컬 모델만 사용할 수 있습니다.")
    session = requests.Session()
    session.trust_env = False
    try:
        response = (session.get(f"{BASE_URL}/api/{endpoint}", timeout=(3, timeout)) if payload is None
                    else session.post(f"{BASE_URL}/api/{endpoint}", json=payload, timeout=(3, timeout)))
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ValueError("로컬 AI에 연결하지 못했거나 처리가 끝나지 않았습니다. run_local_ai.bat를 실행하고 다시 시도하세요.") from exc
    finally:
        session.close()


def model_status():
    try:
        available = {m["name"]: m for m in call_api("tags", timeout=3).get("models", [])}
        return all(name in available for name in ALLOWED_MODELS)
    except ValueError:
        return False


def local_chat(messages, max_tokens=900):
    # Reject any installed model that redirects to a remote/cloud model.
    detail = call_api("show", {"model": CHAT_MODEL}, timeout=10)
    if detail.get("remote_host") or detail.get("remote_model"):
        raise ValueError("클라우드 모델은 사용할 수 없습니다.")
    result = call_api("chat", {"model": CHAT_MODEL, "messages": messages,
        "stream": False, "think": False, "keep_alive": "10m",
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": max_tokens}})
    answer = result.get("message", {}).get("content", "").strip()
    if not answer:
        raise ValueError("로컬 모델의 답변이 비어 있습니다. 질문을 짧게 바꿔주세요.")
    return answer


def as_ollama_messages(prompt):
    messages = []
    for msg in prompt.to_messages():
        role = {"human": "user", "ai": "assistant", "system": "system"}[msg.type]
        content = msg.content
        item = {"role": role, "content": content if isinstance(content, str) else ""}
        if isinstance(content, list):
            for block in content:
                if block["type"] == "text":
                    item["content"] += block["text"]
                elif block["type"] == "image_url":
                    item.setdefault("images", []).append(block["image_url"]["url"].split(",", 1)[1])
        messages.append(item)
    return messages


def describe_image(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return local_chat([{"role": "user", "content":
        "이 문서 이미지를 한국어로 설명하세요. 읽을 수 있는 제목, 표의 행/열과 수치, 차트의 축과 추세를 정확히 기록하세요. 읽을 수 없는 글자는 추측하지 마세요.",
        "images": [encoded]}], max_tokens=650)


class LocalEmbeddings(Embeddings):
    def __init__(self, cache_path=None):
        info = call_api("show", {"model": EMBED_MODEL}, timeout=10)
        if info.get("remote_host") or info.get("remote_model"):
            raise ValueError("클라우드 임베딩은 차단합니다.")
        tags = call_api("tags", timeout=10)["models"]
        self.namespace = next(m["digest"] for m in tags if m["name"] == EMBED_MODEL)
        self.cache_path = Path(cache_path or ROOT / ".local-ai/embeddings.sqlite")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value TEXT)")

    def embed_documents(self, texts):
        keys = [hashlib.sha256((self.namespace + t).encode()).hexdigest() for t in texts]
        result = [None] * len(texts)
        with sqlite3.connect(self.cache_path) as db:
            missing = []
            for i, key in enumerate(keys):
                row = db.execute("SELECT value FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    result[i] = json.loads(row[0])
                else:
                    missing.append(i)
            for start in range(0, len(missing), 16):
                positions = missing[start:start + 16]
                vectors = call_api("embed", {"model": EMBED_MODEL,
                    "input": [texts[i] for i in positions], "truncate": False})["embeddings"]
                if len(vectors) != len(positions):
                    raise ValueError("임베딩 결과 수가 다릅니다.")
                for i, vector in zip(positions, vectors):
                    result[i] = vector
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)", (keys[i], json.dumps(vector)))
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]
