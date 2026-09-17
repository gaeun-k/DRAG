# 미니 RAG 챗봇 — TAVE 18기 DRAG 스터디 2주차

## 개요

LangChain으로 구현한 자체 문서 기반 RAG(Retrieval-Augmented Generation) 챗봇입니다.
teddynote(langchain-kr) `12-RAG/01-RAG-Basic-Webloader.ipynb`를 베이스로 만들었습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `rag_chatbot.py` | 터미널 버전. teddynote 원본 구조를 최대한 그대로 유지한 기본 버전 |
| `app.py` | Streamlit 웹 UI 버전. 대화 기억 · 여러 문서 동시 학습 · 일반 지식 폴백 기능 추가 |
| `requirements.txt` | 필요 패키지 목록 |
| `.env.example` | API 키 설정 예시 |

## 원본 대비 변경 사항 (`rag_chatbot.py`)

1. 네이버 뉴스 고정 URL → 원하는 위키/블로그 URL로 교체 가능하게 변경
2. 하드코딩된 질문 4개를 순서대로 스트림 출력 → 터미널에서 계속 질문 가능한 챗봇 루프로 변경
3. LangSmith 로깅(`langchain_teddynote.logging`)은 필수가 아니라 제외
4. OpenAI LLM → Google Gemini(`gemini-3.6-flash`, 무료 API)로 교체
5. 임베딩: Google 임베딩 API가 계정별 할당량 문제로 불안정해서, 로컬에서 동작하는
   HuggingFace 한국어 임베딩 모델(`jhgan/ko-sroberta-multitask`)로 교체 — 할당량/과금 걱정 없이 실행 가능

## `app.py` 추가 기능

- **대화 기억**: 후속 질문을 이전 대화 맥락으로 재해석한 뒤 문서를 검색
- **멀티 문서**: 여러 URL을 한 번에 입력하면 하나의 벡터스토어로 통합해서 학습
- **일반 지식 폴백**: 문서에 없는 질문은 AI의 일반 지식으로도 답변 (사이드바에서 켜고 끌 수 있음, 켜져 있을 땐 "문서에 없는 내용"임을 먼저 밝히고 답함)

## 실행 방법

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# .env.example을 .env로 복사한 뒤 GOOGLE_API_KEY 입력

python rag_chatbot.py        # 터미널 버전
streamlit run app.py         # 웹 UI 버전 (추가 기능 포함)
```

## 참고 자료

- Base 코드: [teddynote/langchain-kr, 12-RAG](https://github.com/teddylee777/langchain-kr)
