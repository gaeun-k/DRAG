# 2주차 무료 로컬 RAG 챗봇

**처음 받는 조원은 아래 설치 절차부터 진행하세요.** 설치 후에는 `run_chatbot.bat`만 실행하면 됩니다.

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

`rag.py`의 OpenAI 코드은 비교 학습 자료로 보존했습니다. 현재 앱은 `StudyRag`를 사용합니다. 원본 노트북의 모든 패키지·모델을 똑같이 복제한 것은 아니며 비용이 드는 부분은 로컬 대체 구현으로 연결했습니다. 첨부 코드 설명은 `docs/테디노트_코드_상세해설.md`, 현재 구현 차이는 `docs/현재_구현_범위.md`를 읽으세요.

## 검증

```powershell
.\.venv\Scripts\python.exe verify_setup.py
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


## 조원 PC에서 처음 설치하기

지원 설치 환경은 **Windows 10/11 x64 + Python 3.12 64비트**입니다. macOS/Linux 자동 설치 스크립트는 포함하지 않았습니다. CPU에서도 실행할 수 있지만 작은 모델이어도 속도와 메모리 사용량은 PC에 따라 다릅니다. 개발 검증 환경은 RAM 32GB였으며, 다른 사양의 최저 메모리 요구량은 검증하지 않았습니다.

1. [Python 공식 다운로드](https://www.python.org/downloads/windows/)에서 Python 3.12 64비트를 설치하고 Python launcher를 포함합니다.
2. 이 저장소 위쪽 **Code → Download ZIP**을 선택하고 ZIP을 압축 해제합니다. Git을 쓰면 `git clone https://github.com/gaeun-k/DRAG.git`도 가능합니다.
3. 압축을 푼 폴더의 `week2-mini-chatbot/임지민`으로 이동합니다. 저장소 루트에서 실행하지 마세요.
4. **setup_windows.bat**를 더블클릭합니다. 가상환경, Python 패키지, Ollama, 두 로컬 모델을 설치합니다. 최초 설치에는 인터넷과 수 GB 이상의 여유 공간이 필요합니다.
5. 완료되면 **run_chatbot.bat**를 실행합니다. 실행 창을 닫지 마세요.
6. 브라우저가 자동으로 열리지 않으면 실행 창에 표시된 URL을 엽니다. 기본값은 `http://127.0.0.1:8501`입니다.
7. **실습용 자체 위키 사용 → 문서 인덱스 만들기 → 질문 입력** 순서로 사용합니다.

API 키나 결제는 필요 없습니다. 패키지와 모델의 최초 다운로드를 마친 뒤 로컬 파일 QA는 인터넷 없이 사용할 수 있습니다. 웹 문서 수집에는 인터넷이 필요합니다. 서버가 종료되면 다른 사람에게 localhost 주소를 보내도 접속할 수 없습니다. 조원마다 자신의 PC에서 실행합니다.

### 수동 설치

현재 `임지민` 폴더에서 PowerShell을 열고 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe verify_setup.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_local_ai.ps1
.\.venv\Scripts\python.exe verify_setup.py --models
.\run_chatbot.bat
```

`ExecutionPolicy Bypass`는 해당 설치 프로세스에만 적용됩니다. 기존 Ollama 서버가 11434 포트에서 실행 중이면 그 서버를 사용하고 그 서버의 모델 저장소에 모델을 다운로드합니다. 새 서버는 `.local-ai/models`를 사용합니다.

### 자체 문서와 실습

- 샘플은 가상의 스터디 위키입니다. 자체 문서를 쓰려면 샘플 체크를 해제하고 파일을 올립니다.
- 파일당 30MB, 합계 1GB입니다. 실제 1GB 처리 성능은 검증하지 않았습니다.
- `data` 폴더 체크로 로컬 문서를 읽을 수도 있습니다. 개인 문서는 GitHub에 추가하지 마세요.
- 이미지·표 분석은 해당 체크를 켭니다. PDF 시각 분석은 기본 앞 3페이지이며 범위를 조절할 수 있습니다.
- `run_notebook.bat`로 JupyterLab을 열고 01, 02 노트북을 실행합니다.
- `docs/나의_구현_과정.md`, `docs/실험_기록.md`에 자신이 바꾼 내용과 결과를 남깁니다.

### 설치 오류 확인

- `py` 또는 Python 3.12를 찾지 못함: Python launcher를 포함해 64비트 Python 3.12를 설치하세요.
- 모델 다운로드 실패: 인터넷과 남은 디스크 용량을 확인하고 `setup_windows.bat`를 다시 실행하세요.
- 로컬 AI 연결 오류: `run_local_ai.bat`를 실행하고 다시 시도하세요. 이미 서버가 있으면 포트 사용 중 안내가 나올 수 있습니다.
- 8501 포트 사용 중: 기존 챗봇 실행 창을 닫거나 Streamlit 실행에 `--server.port 8502`를 추가하세요.
- 첫 답변이 느림: 모델을 처음 읽는 데 시간이 필요합니다. 우선 짧은 샘플 문서로 확인하세요.

### 배포 범위와 재현 확인

Python 소스, 버전 목록, 실행 스크립트, 출력이 제거된 노트북, 샘플 문서와 설명 파일을 제공합니다. `.venv`, `.local-ai`, `.env`, 개인 PDF와 대화 로그는 제외했습니다. `verify_setup.py`는 모델 없이 로딩·검색·무료 정책을 점검하고 `--models`는 모델 설치 여부까지 확인합니다. `verify_local.py`와 `verify_advanced_local.py`는 실제 로컬 모델을 사용한 추가 실험입니다.
