"""
현재 이 API 키로 실제 사용 가능한 임베딩 모델 목록을 확인하는 스크립트.
"""
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

print("=== embedContent를 지원하는 모델 목록 ===")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "embedContent" in actions:
        print(m.name)
