# 임지민의 week1-rag-basics 실습

여기에 이번 주 실습 코드와 결과를 정리해주세요.

## 1. RAG란?

**RAG = Retrieval-Augmented Generation**

쉽게 말하면,

> **LLM이 바로 답하지 않고 → 먼저 외부 문서에서 필요한 정보를 검색하고 → 검색된 내용을 근거로 답하게 하는 구조**
> 

LLM 단독 사용에는 최신 정보, 사내·개인 문서, 출처 추적, hallucination 등의 한계가 있고, RAG는 외부 문서를 검색해서 이런 한계를 보완한다.

전체 과정은 

**Indexing → Retrieval → Generation**

> 문서를 미리 준비한다
> 
> 
> → 질문과 관련된 문서를 찾는다
> 
> → 질문 + 찾은 문서를 LLM에게 준다
> 

---

## 2. Indexing

Indexing은 사용자가 질문하기 **전에 미리 해놓는 준비 단계**야.

```
문서
 ↓
Chunking
 ↓
Embedding
 ↓
Vector Store 저장
```

예를 들어 100페이지짜리 사업보고서가 있으면 그대로 저장하는 게 아니라,

```
Chunk 1 : 매출과 영업이익
Chunk 2 : AI CAPEX
Chunk 3 : 환율 위험
Chunk 4 : 향후 전망
```

처럼 작은 단위로 쪼갠다.

왜냐하면 문서 전체를 하나로 임베딩하면 여러 주제가 한 벡터 안에 섞이고, 질문과 관련 없는 내용까지 검색될 수 있기 때문이야. Chunk가 너무 작아도 문맥이 끊기고 너무 크면 검색 노이즈가 많아지므로 `chunk_size`도 RAG 성능에 영향을 준다.

### 핵심

**Indexing은 질문할 때마다 하는 게 아니다.**

문서가 업데이트됐을 때 다시 해주는 사전 준비 작업임

---

# 3. Embedding

Embedding은

> **텍스트의 의미를 숫자로 된 벡터로 표현하는 것**
> 

이야.

예를 들어,

```
"강아지가 뛰어논다"

→ [0.12, -0.44, 0.88, ...]
```

처럼 표현한다.

여기서 중요한 건 `0.12`가 강아지를 의미한다든가 하는 식으로 **각 숫자를 해석하는 것이 아니다.**

핵심은

> **의미가 비슷한 텍스트 → 벡터 공간에서도 가깝다**
> 

는 거야.

예를 들어

```
강아지 ── 개 ── puppy

주식 ── 채권 ── 금융자산
```

처럼 비슷한 개념끼리 가까운 곳에 위치하도록 학습된다.

이때 벡터 사이의 유사도를 판단할 때 대표적으로 **Cosine Similarity**를 사용

---

# 4. Vector Store

그럼 임베딩된 벡터를 어디에 저장할까?

→ **Vector Store**

Vector Store에는 보통

```
Embedding Vector
+
원래 Text Chunk
+
Metadata
```

를 같이 연결해둔다.

Metadata에는 예를 들어

```
source = "2026_report.pdf"
page = 23
company = "Samsung"
date = "2026-03-01"
```

같은 정보가 들어갈 수 있다.

사용자의 질문도 벡터로 바꾼 뒤, Vector Store에서 **질문 벡터와 가까운 문서 벡터를 찾는다.**

---

# 5. FAISS vs Chroma

둘 다 RAG에서 Vector Store처럼 사용할 수 있지만 역할 범위가 다름.

**FAISS**

> 빠른 벡터 similarity search에 집중한 라이브러리
> 

**Chroma**

> 벡터 + 원문 + Metadata + Collection 등을 관리하는 DB 성격의 도구
> 

비교

> **벡터 검색 자체를 빠르게 실험 → FAISS**
> 
> 
> **RAG 데이터까지 편하게 관리 → Chroma**
> 

둘의 우열이라기보다 abstraction 범위가 다르다

---

# 6. Retrieval

문서는 이미 저장되어 있음

이제 사용자가 질문했다.

```
"금리가 올라가면 주식은 왜 떨어질까?"
```

그러면 수많은 Chunk 중에서

> **어떤 Chunk를 LLM에게 줘야 할까?**
> 

를 결정해야함

그 역할이 **Retriever**

---

# 7. Dense Retrieval

Dense Retrieval은 **Embedding을 이용한 의미 검색**

```
질문
 ↓
Embedding
 ↓
Query Vector
 ↓
문서 Vector들과 비교
 ↓
가까운 문서 Top-k 선택
```

예를 들어

```
Query:
"자동차를 싸게 빌리는 방법"

Document:
"렌터카 할인 프로모션 안내"
```

단어 자체는 많이 다르지만 뜻은 비슷함

Dense Retrieval은 이를 벡터의 의미 유사도로 찾는다.

### Dense가 잘하는 것

```
동의어
패러프레이즈
표현이 달라도 의미가 같은 문장
```

### Dense가 약할 수 있는 것

```
AB-1937
제12조 3항
특정 이름
정확한 숫자
```

처럼 exact match가 중요한 검색.

---

# 8. Sparse Retrieval / BM25

BM25는 Embedding이 없어도 됨

핵심은

> **질문에 나온 단어가 문서에 얼마나 잘 등장하는가?**
> 

야.

예를 들어 질문이

```
AI 반도체 수출 규제
```

라면

```
AI       → D1, D2, D5
반도체    → D2, D5
수출      → D2, D3
규제      → D2, D4
```

이고 D2에 네 단어가 전부 있으면 높은 후보가 되는 식

하지만 BM25는 단순히 단어 개수만 세는 건 아니다.

### TF

Term Frequency.

> 그 문서에 검색 단어가 얼마나 자주 등장하는가?
> 

하지만 똑같은 단어를 무한히 반복한다고 점수가 계속 똑같이 상승하지는 않는다.

### IDF

Inverse Document Frequency.

> 전체 문서 중 얼마나 희귀한 단어인가?
> 

예를 들어

```
"정보"
```

보다

```
"HBM3E"
```

같은 희귀한 단어가 훨씬 중요한 검색 신호가 된다.

### Document Length

너무 긴 문서가 단순히 단어가 많이 포함되었다는 이유만으로 유리하지 않도록 길이도 보정한다.

---

# 9. Dense vs Sparse

|  | Dense | Sparse / BM25 |
| --- | --- | --- |
| 보는 것 | **의미** | **단어** |
| Embedding | 필요 | 불필요 |
| 강점 | 동의어, 다른 표현 | exact keyword |
| 예 | 싼 렌터카 ↔ 차량 할인 대여 | AB-1937 ↔ AB-1937 |
| 약점 | 정확 문자열 | 표현 변화 |

> **Dense는 의미가 비슷한지를 보고, Sparse는 실제 단어가 얼마나 잘 겹치는지를 봅니다.**
> 

---

# 10. Hybrid Retrieval

```
                 ┌→ Dense Retrieval ──┐
Query ───────────┤                    ├→ 결과 결합 → Top-k
                 └→ BM25 Retrieval ───┘
```

예를 들어

```
"삼성전자 HBM3E 공급 전망"
```

이라면,

BM25는

```
삼성전자
HBM3E
```

같은 정확한 키워드를 잘 찾고,

Dense는

```
"고대역폭 메모리 공급 확대"
```

처럼 표현은 다르지만 의미가 비슷한 문서를 찾을 수 있음

---

# 11. Generation

검색이 끝나면 최종적으로

```
사용자의 Question
+
검색된 Context
+
Prompt
```

를 LLM에게 전달한다.

즉,

```
Question:
금리 인상이 성장주에 왜 악재인가?

Context:
[검색된 Chunk 1]
[검색된 Chunk 2]
[검색된 Chunk 3]

Instruction:
위 Context만 이용해서 답하세요.
```

처럼 만들고 LLM이 최종 Answer를 만든다.

---

# 12. 최종 구조

이걸 한 번에 보면:

```
            [ OFFLINE ]

PDF / 문서
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Store
    │
    │
──────────────
    │
            [ ONLINE ]

사용자 질문
    ↓
Retrieval
 ┌──┴───┐
Dense  BM25
 └──┬───┘
   Hybrid
     ↓
   Top-k
     ↓
Question + Context
     ↓
    LLM
     ↓
   Answer
```
