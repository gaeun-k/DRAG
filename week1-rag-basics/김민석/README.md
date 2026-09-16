# 김민석의 week1-rag-basics 실습

# 김민석의 week1-rag-basics 실습

테디노트 랭체인 한국어 튜토리얼 `08-Embeddings`, `09-VectorStore`, `10-Retriever` 실습 코드와 결과 정리.

- 실행 환경: Google Colab (CPU)
- 임베딩 기본 모델: `jhgan/ko-sroberta-multitask` (한국어, 768차원, 무료)
- 원본은 OpenAI 유료 API 기준이라, 대부분 허깅페이스 무료 모델로 바꿔서 진행함

## 폴더 구조

```
김민석/
├── 08-Embeddings/
│   └── 08_Embedding.ipynb      # 임베딩 모델 7종 비교
├── 09-VectorStore/
│   └── 09_VectorStore.ipynb    # Chroma / FAISS / Pinecone
└── README.md
```

---

## CH08 — 임베딩: 모델 7종 직접 비교

같은 문장 세트를 모델 7개에 전부 태워서 비교함.

| 모델 | 차원 | 비용 | 특이사항 |
|---|---|---|---|
| `ko-sroberta-multitask` | 768 | 무료 | 접두어 없이 바로 정상 동작. 기본으로 채택 |
| `multilingual-e5-large-instruct` | 1024 | 무료 | `Instruct: ...\nQuery: ...` 형식 필요. 접두어 없이 돌리면 무관한 문장이 더 높은 점수를 받는 걸 실제로 확인함 |
| Solar (Upstage) | 4096 | API 키 | 질문/문서 모델이 아예 분리되어 있어 가장 안정적인 순위가 나옴 |
| `nomic-embed-text` (Ollama) | 768 | 무료(로컬) | `search_query:`/`search_document:` 접두어 필요 + 영어 전용이라 한국어 질의에서 한계 확인 |
| `nomic-embed-text` (llama.cpp) | 768 | 무료(로컬) | 개념은 Ollama와 동일. `llama-cpp-python` 설치(컴파일)에 시간이 오래 걸림 |
| GPT4All (MiniLM) | 384 | 무료 | 키/설정 없이 즉시 동작. 차원이 가장 작음 |

**실행 결과 — 접두어 유무에 따른 차이 (nomic-embed-text)**

```
접두어 없음:
[0] 0.926 | LangChain은 초거대 언어모델로...
[3] 0.706 | 안녕, 만나서 반가워.        ← 무관한 문장인데 3위
[4] 0.424 | Retrieval-Augmented Generation...  ← 관련 있는데 꼴찌

접두어(search_query:/search_document:) 적용 후:
순위는 동일했으나 점수 격차는 좁혀짐 (0.706→0.604, 0.424→0.515)
→ 원인은 접두어가 아니라 nomic-embed-text 자체가 영어 전용 모델이기 때문
```

**배운 것**: 모델 카드에 명시된 입력 형식(접두어)을 지키지 않으면 검색 순위가 실제로 왜곡된다. 또한 지원 언어가 안 맞으면 접두어를 고쳐도 근본적인 한계는 남는다.

---

## CH09 — 벡터DB: Chroma / FAISS / Pinecone

### Chroma, FAISS 기본 사용
- 문서 저장 → 유사도 검색 → 메타데이터 필터 → 문서 추가/삭제 → 저장/로드까지 원본 예제 그대로 실행, 에러 없이 완료
- FAISS는 `merge_from()`으로 서로 다른 두 벡터DB를 합칠 수 있음을 확인

**직접 발견한 버그**: 원본 문서가 짧으면(413자) `chunk_size=600` 설정과 무관하게 파일 전체가 청크 1개로 묶여버림. `chunk_size`를 100으로 낮춰 용어별로 재분할한 뒤 정상화함. → 2주차 청킹 전략이 왜 필요한지 실제 사례로 확인.

### Pinecone 하이브리드 검색 (Dense+Sparse, alpha 가중합)

Dense(임베딩)와 Sparse(Kiwi+BM25)를 `alpha` 값으로 가중합하는 방식. 같은 질문("TF IDF 에 대하여 알려줘")으로 alpha만 바꿔서 비교함.

```
=== alpha=1 (Dense만) ===
점수 42.281 | TF-IDF: 단어의 빈도와 여러 문서에 걸친 희소성을...
점수 35.610 | 토큰화: 텍스트를 문장, 단어, 서브워드 등...
점수 25.850 | 임베딩: 텍스트, 이미지 등 데이터를...

=== alpha=0 (Sparse만) ===
점수 0.202  | TF-IDF: 단어의 빈도와 여러 문서에 걸친 희소성을...
점수 0.000  | BERT: 문장 전체의 양방향 문맥을...
점수 0.000  | 변동성: 자산 가격이 일정 기간 동안...
```

**배운 것**: Dense는 "뜻이 비슷한" 문서를 폭넓게 가져오고(0점 없음), Sparse는 질문 단어가 전혀 안 겹치면 가차없이 0점을 준다. `alpha`로 이 비중을 직접 조절할 수 있다 

---


## 참고자료

- 테디노트 위키독스 「랭체인 노트」 — https://wikidocs.net/book/14314
- 깃허브 [teddylee777/langchain-kr](https://github.com/teddylee777/langchain-kr)
