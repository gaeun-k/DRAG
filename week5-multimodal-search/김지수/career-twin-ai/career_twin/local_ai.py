"""Ollama HTTP API. 개인 자료는 설정한 서버로 전달됩니다."""
import base64
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .config import SETTINGS


class ModelError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url=SETTINGS.ollama_url):
        self.url = url.rstrip("/")

    def request(self, route, payload=None):
        req = Request(self.url + route, data=None if payload is None else
                      json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=180) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            raise ModelError(f"Ollama HTTP {exc.code}. 모델 설치와 서버 설정을 확인하세요.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ModelError("Ollama에 연결할 수 없습니다. ollama serve와 모델 설치를 확인하세요.") from exc

    def models(self):
        return [m["name"] for m in self.request("/api/tags").get("models", [])]

    def embed(self, texts):
        vectors = self.request("/api/embed", {"model": SETTINGS.embed_model, "input": texts,
                                "truncate": False, "keep_alive": "0s"}).get("embeddings", [])
        if len(vectors) != len(texts):
            raise ModelError("임베딩 개수가 입력과 다릅니다.")
        return vectors

    def chat(self, system, prompt, schema=None, image=None):
        message = {"role": "user", "content": prompt}
        if image is not None:
            message["images"] = [base64.b64encode(image).decode()]
        payload = {"model": SETTINGS.vision_model if image is not None else SETTINGS.chat_model,
                   "messages": [{"role": "system", "content": system}, message],
                   "stream": False, "keep_alive": "0s",
                   "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1400}}
        if schema:
            payload["format"] = schema
        result = self.request("/api/chat", payload)
        if result.get("done_reason") == "length":
            raise ModelError("답변 길이 한도에 도달했습니다. 질문이나 요구사항 수를 줄여주세요.")
        content = result.get("message", {}).get("content", "")
        if not content.strip():
            raise ModelError("모델에서 답변을 받지 못했습니다.")
        if schema:
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise ModelError("모델이 올바른 JSON을 반환하지 않았습니다. 다시 실행하세요.") from exc
        return content
