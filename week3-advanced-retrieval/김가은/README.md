# 김가은의 week3-advanced-retrieval 실습

## RAG 챗봇 + Reranking + SigLIP 이미지 검색

`week3_rag_chatbot_compact.ipynb`

2주차 챗봇(Dense 검색 + LLM)에 **Reranking**을 붙여 켰을 때와 껐을 때를 비교하고, 4주차 **SigLIP**을 붙여 질문하면 관련 이미지도 같이 찾아 주는 웹 챗봇입니다. 기존 노트북에서 기능을 줄이고 쓰기 편하게 정리했습니다.

### 검색 모드
| 모드 | 동작 |
|---|---|
| 기본 검색 | bge-m3 Dense 검색 상위 4개 → LLM 답변 (2주차와 동일) |
| Reranking | Dense 상위 20개 → Cross-Encoder(`bge-reranker-v2-m3`)로 다시 채점 → 상위 4개 → LLM 답변 |
| 이미지 검색 (SigLIP) | 질문을 SigLIP 텍스트 벡터로 바꿔 미리 계산한 이미지 벡터와 비교, 짝일 확률 `sigmoid(t·cos + b)` 표시 |

### 흐름
1. 모델 로드: LLM(`Qwen3-4B-Instruct-2507`), 임베딩(bge-m3), Reranker, SigLIP(다국어)
2. 문서 로더: URL은 본문만 추출(위키는 `#mw-content-text`), PDF는 PyMuPDF로 페이지별 추출. 청크마다 출처·제목·페이지 기록
3. 지식베이스: bge-m3 임베딩 + FAISS 내적(= 코사인 유사도)
4. 이미지 저장소: 올린 이미지는 SigLIP 벡터를 미리 계산해 두고, 질문 벡터만 그때그때 계산
5. 답변 생성: 이전 대화 몇 턴 + 번호 붙은 문맥을 넣고, 문장 끝에 `[1]`처럼 근거 번호 표시
6. (선택) 평가: 문서별로 질문·정답을 생성하고 RAGAS 4대 지표(Faithfulness, Answer Relevancy, Context Precision, Context Recall)로 기본 검색 vs Reranking 비교
7. Gradio 웹 챗봇

### 웹 챗봇 기능
- **💬 채팅**: 답변 스트리밍, 이전 대화 기억, `📎 근거 문서`를 펼치면 어떤 문서의 몇 페이지를 썼는지 표시
- **사이드바**: 문서 추가(URL / PDF), 검색할 문서 선택, 이미지 추가, Reranking·이미지 검색 켜고 끄기
- **🖼️ 이미지 검색**: 문장으로 이미지만 검색 (5주차 텍스트→이미지 검색 미리보기)
- **⚖️ 비교**: 같은 질문을 기본 검색 / Reranking으로 동시에 돌려 나란히 비교 (순위 변화 포함)

### 기존 노트북에서 바꾼 점
- 기능을 검색 모드 2개 + 이미지 검색으로 줄이고, 설정은 사이드바로 모음
- 답변을 한 글자씩 스트리밍하고, 근거 문서는 접히는 메시지로 분리
- 이미지 결과 메시지는 LLM 대화 기록에서 제외해 Q&A 문맥이 밀려나지 않게 함

### 알려진 한계
- 같은 파일 이름으로 이미지를 다시 올리면 파일은 바뀌지만 기존 벡터가 남음
