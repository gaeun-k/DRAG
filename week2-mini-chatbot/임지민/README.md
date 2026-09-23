# 2주차 무료 로컬 RAG 챗봇

**[쉬운 실행 안내](START_HERE.md)**부터 읽으세요. `run_chatbot.bat`를 더블클릭하면 됩니다.

이 버전은 Ollama의 로컬 **Qwen3.5 2B**로 답변·이미지 분석, **EmbeddingGemma**로 의미 임베딩을 수행합니다. API 키나 결제는 필요 없고, OpenAI 및 클라우드 모델 호출은 차단합니다.

## 현재 기능

| 기능 | 구현 |
| --- | --- |
| 답변 가독성 | 짧은 답변과 핵심 항목, 접힌 출처별 원문, 긴 파일명 축약 |
| 문서 입력 | PDF/TXT/MD/HTML/CSV/Python, data 하위 폴더, 공개 웹 주소, 이미지 |
| 청킹 | 재귀·구분자·의미 분할 비교 |
| 의미 검색 | 로컬 임베딩 + FAISS 또는 Chroma |
| 검색 비교 | similarity, BM25, 혼합 순위 결합, MMR, threshold, MultiQuery |
| 대화 기억 | RunnableWithMessageHistory, 최근 3회 대화, 검색 질의 재작성 |
| 멀티모달 | 표 추출, 페이지/이미지 설명 인덱싱, 원본 이미지와 연결한 답변 |
| RAPTOR | UMAP/GMM/BIC를 사용한 제한 규모 계층 요약 및 원문 연결 |
| Chain of Density | 본문에서 빠진 정보를 보완하는 5회 요약 |
| 학습 기록 | 단계별 노트북, 질문·답변·근거 JSON 다운로드 |
| 캐시 | 모델 digest와 본문 해시를 키로 로컬 SQLite 임베딩 캐시 |

## 코드 읽는 순서

1. `step_by_step.py`: 원래 기본 RAG 흐름을 단계별로 실행합니다. `DEMO=False`도 무료 로컬 모델입니다.
2. `local_ai.py`: 고정된 loopback 주소와 로컬 모델만 호출합니다.
3. `ingest.py`: 파일·웹 페이지·표·이미지 로딩입니다.
4. `advanced_rag.py`: 청킹·검색·기억·답변·심화 알고리즘입니다.
5. `app.py`: 입력 화면과 읽기 쉬운 답변/출처 표시입니다.
6. `runtime_policy.py`: 파일당 30MB, 합계 1GB 및 무료 호출 정책입니다.

`rag.py`의 OpenAI 코드와 `references`의 원본은 비교 학습 자료로 보존했습니다. 현재 앱은 `StudyRag`를 사용합니다. 원본 노트북의 모든 패키지·모델을 똑같이 복제한 것은 아니며 비용이 드는 부분은 로컬 대체 구현으로 연결했습니다. 첨부 코드 설명은 `docs/테디노트_코드_상세해설.md`, 현재 구현 차이는 `docs/현재_구현_범위.md`를 읽으세요.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe verify_local.py
```

마지막 명령은 실제 로컬 모델과 공개 예제 웹 주소를 사용합니다. 인터넷은 웹 수집에만 필요하며 AI 요청은 내 컴퓨터에서 처리합니다. 결과는 `outputs/local-verification.json`에 저장합니다.

작은 로컬 모델의 답변·인용·이미지 숫자 인식은 완벽하지 않습니다. 테스트 통과가 모든 실제 문서의 정확성을 보장하지는 않습니다. 인덱스와 대화는 세션 메모리이며 디스크의 캐시는 임베딩과 이미지입니다.

## 공식 참고

- 기반 예제: [테디노트 12-RAG](https://github.com/teddylee777/langchain-kr/tree/main/12-RAG)
- 로컬 채팅/이미지: [Ollama chat API](https://docs.ollama.com/api/chat)
- 로컬 임베딩: [Ollama embed API](https://docs.ollama.com/api/embed)
- 클라우드 차단: [Ollama FAQ](https://docs.ollama.com/faq)
