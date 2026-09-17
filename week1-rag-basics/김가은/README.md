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

- **CH08 — HuggingFace Embeddings**: HuggingFace Hub/Endpoint 임베딩 모델을 LangChain에 연결해 문서 임베딩 생성
- **CH09 — Chroma**: 문서 저장, 유사도 검색, 메타데이터 필터링, 문서 추가/삭제, 저장/로드
- **CH09 — FAISS**: 문서 저장, 유사도 검색, 인덱스 병합(`merge_from`), 저장/로드
- **CH10 — VectorStoreRetriever**: 벡터DB 기반 Dense Retrieval (`similarity`, `mmr` 검색)
- **CH10 — Kiwi-BM25Retriever**: 한국어 형태소 분석기(Kiwi) 토크나이징 + BM25 기반 Sparse Retrieval

→ 임베딩 모델 개념, 벡터DB(FAISS vs Chroma) 비교, Dense Retrieval vs Sparse Retrieval(BM25) 원리까지 1주차 학습 목표를 모두 다룹니다.

## 참고자료

- [테디노트 위키독스 「랭체인 노트」](https://wikidocs.net/book/14314)
- [teddylee777/langchain-kr (GitHub)](https://github.com/teddylee777/langchain-kr)
