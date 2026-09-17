# 김가은의 week1-rag-basics 실습

테디노트 랭체인 한국어 튜토리얼([teddylee777/langchain-kr](https://github.com/teddylee777/langchain-kr)) `08-Embeddings`, `09-VectorStore`, `10-Retriever` 실습 코드 정리.

## 폴더 구조

```
김가은/
├── 08-Embeddings/
│   └── 03-HuggingFaceEmbeddings.ipynb     # HuggingFace 임베딩 모델 사용법
├── 09-VectorStore/
│   ├── 01-Chroma.ipynb                    # Chroma 벡터DB 기본 사용법
│   ├── 02-FAISS.ipynb                     # FAISS 벡터DB 기본 사용법
│   └── data/
│       ├── finance-keywords.txt
│       └── nlp-keywords.txt
├── 10-Retriever/
│   ├── 01-VectorStoreRetriever.ipynb      # VectorStore 기반 Dense Retrieval
│   ├── 10-Kiwi-BM25Retriever.ipynb        # Kiwi 형태소 분석 + BM25 Sparse Retrieval
│   └── data/
│       └── appendix-keywords.txt
└── README.md
```

## 실습 내용

### CH08 — HuggingFace Embeddings (`03-HuggingFaceEmbeddings.ipynb`)
- HuggingFace Endpoint / HuggingFace Hub 임베딩 모델을 LangChain에 연결
- 벡터 내적을 이용한 유사도 계산 원리 확인
- `intfloat/multilingual-e5-large-instruct` 등 다국어 모델 실습
- BGE-M3 임베딩: Dense, Sparse(Lexical Weight), Multi-Vector(ColBERT) 3가지 방식 비교

### CH09 — Chroma (`09-VectorStore/01-Chroma.ipynb`)
- `from_documents` / `from_texts`로 벡터 저장소 생성
- 유사도 검색, 문서 추가/삭제, 컬렉션 초기화(`reset_collection`)
- 벡터 저장소를 Retriever로 변환
- 멀티모달(이미지) 임베딩을 활용한 이미지 검색 실습

### CH09 — FAISS (`09-VectorStore/02-FAISS.ipynb`)
- `from_documents` / `from_texts`로 FAISS 벡터 저장소 생성, 유사도 검색
- 문서/텍스트 추가(`add_documents`, `add_texts`), 문서 삭제
- 로컬 저장·로드, 서로 다른 두 FAISS 객체 병합(`merge_from`)
- Retriever로 변환(`as_retriever`)

### CH10 — VectorStoreRetriever (`10-Retriever/01-VectorStoreRetriever.ipynb`)
- `as_retriever()`로 VectorStore 기반 Dense Retrieval 초기화
- Max Marginal Relevance(MMR)로 다양성 있는 검색 결과 확보
- 유사도 점수 임계값(`similarity_score_threshold`) 검색, `top_k` 설정
- 질문/문서 임베딩 모델이 분리된 경우(예: Upstage)의 동적 설정(Configurable)

### CH10 — Kiwi-BM25Retriever (`10-Retriever/10-Kiwi-BM25Retriever.ipynb`)
- 한국어 형태소 분석기 Kiwi로 토크나이징 후 BM25 Sparse Retrieval 적용
- 다양한 문장으로 검색기 튜닝 결과 테스트
- Kiwi, 공백 기준 분리 등 여러 검색기의 검색 결과 비교 실험
- Konlpy 형태소 분석기와의 비교

→ 임베딩 모델 개념, 벡터DB(FAISS vs Chroma) 비교, Dense Retrieval vs Sparse Retrieval(BM25) 원리까지 1주차 학습 목표를 모두 다룹니다.

## 참고자료

- [테디노트 위키독스 「랭체인 노트」](https://wikidocs.net/book/14314)
- [teddylee777/langchain-kr (GitHub)](https://github.com/teddylee777/langchain-kr)
