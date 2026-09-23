# 김가은의 week2-mini-chatbot 실습

LangChain으로 구현한 자체 문서 기반 RAG(Retrieval-Augmented Generation) 챗봇입니다.
teddynote(langchain-kr) `12-RAG/01-RAG-Basic-Webloader.ipynb`를 베이스로 만들었습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `week2_chat.py` | 고정 문서 버전. 위키백과 등 지정한 URL 1개를 로드해서, 그 문서로 exit 전까지 계속 질문 가능 |
| `week2_wiki-chat.py` | 나무위키 검색 버전. 검색할 "단어"를 입력받아 해당 나무위키 문서(`namu.wiki/w/단어`)를 그때그때 크롤링해서 답변. 같은 단어로는 계속 질문 가능, `새단어` 입력 시 다른 단어로 전환 |

## 원본(01-RAG-Basic-Webloader.ipynb) 대비 변경 사항

1. 네이버 뉴스 고정 URL → 원하는 위키/블로그 URL로 교체 가능하게 변경 (`week2_chat.py`는 `SOURCE_URL` 변수, `week2_wiki-chat.py`는 검색어 입력으로 나무위키 URL을 그때그때 조립)
2. 하드코딩된 질문 4개를 순서대로 스트림 출력 → 터미널에서 exit 전까지 계속 질문 가능한 챗봇 루프로 변경
3. LangSmith 로깅(`langchain_teddynote.logging`)은 필수가 아니라 제외
4. 임베딩·LLM 모두 유료 API(OpenAI)가 막혀서 무료 조합으로 교체
   - 임베딩: `OpenAIEmbeddings` → `HuggingFaceEmbeddings(BAAI/bge-m3)` (로컬 계산, 완전 무료. 1주차 CH08에서 다룬 모델을 그대로 재사용)
   - LLM: `ChatOpenAI` → `ChatGoogleGenerativeAI(gemini-3.8-flash)` (무료 API 키. 구버전 모델은 신규 사용자에게 제공되지 않아 최신 모델로 교체)

문서 분할 / 벡터스토어(FAISS) 생성 / 검색기 / 프롬프트 / 체인 구성은 원본 구조를 그대로 따릅니다.

## `week2_wiki-chat.py` 동작 방식

```
검색할 단어: 아이유
['아이유' 문서 로드 완료. 이제 이 문서에 대해 계속 질문하세요]

질문: 데뷔 연도는?
→ 아이유의 데뷔 연도는 2008년입니다...

질문: 소속사는 어디야?
→ EDAM엔터테인먼트입니다...

질문: 새단어        # 다른 단어로 전환
검색할 단어: 손흥민
...

질문: exit          # 완전 종료
```

## 실행 방법

```bash
pip install -r requirements.txt   # dotenv, langchain, langchain-community, langchain-google-genai, langchain-huggingface, sentence-transformers, faiss-cpu, beautifulsoup4

# .env에 GOOGLE_API_KEY 입력 (https://aistudio.google.com/apikey 에서 무료 발급)

python week2_chat.py        # 고정 문서 버전
python week2_wiki-chat.py   # 나무위키 검색 버전
```

⚠️ `gemini-3.8-flash` 무료 티어는 **하루 20회 요청 제한**이 있습니다. 한도를 넘기면 `429 RESOURCE_EXHAUSTED` 에러가 발생하니, 데모/테스트 시 참고하세요.

## 디버깅하며 겪은 문제

- `OPENAI_API_KEY` 크레딧 소진(429) → 임베딩을 로컬 모델로 교체
- `GOOGLE_API_KEY` 프로젝트가 403 "denied access" 상태 → 새 프로젝트로 키 재발급해서 해결 (기존 키가 노출된 적이 있어서였던 것으로 추정)
- `gemini-2.5-flash` 등 구버전 모델이 신규 사용자에게 404 → `gemini-3.8-flash`로 교체
- 나무위키는 별도 JS 렌더링 없이 `requests` 기반 `WebBaseLoader`로도 본문이 정상적으로 크롤링됨을 확인

## 참고 자료

- Base 코드: [teddynote/langchain-kr, 12-RAG](https://github.com/teddylee777/langchain-kr)
