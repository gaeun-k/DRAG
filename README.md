# DRAG

TAVE 스터디 **DRAG**의 실습 코드 저장소입니다. 텍스트 기반 RAG의 핵심 파이프라인을 먼저 이해한 뒤, SigLIP 등 최신 비전-언어 임베딩 모델을 활용해 이미지와 텍스트를 함께 다루는 멀티모달 검색·RAG 시스템으로 확장합니다.

## 팀 정보

| 팀장 | 팀원 |
|---|---|
| 김가은 | 김민석, 김지수, 서태영, 유연, 임지민 |

## 6주 로드맵

| 주차 | 폴더 | 내용 |
|---|---|---|
| 1주차 | [`week1-rag-basics`](./week1-rag-basics) | RAG 기본 개념, 임베딩, 벡터DB, Dense/Sparse 검색 |
| 2주차 | [`week2-mini-chatbot`](./week2-mini-chatbot) | LangChain으로 미니 RAG 챗봇 구현 |
| 3주차 | [`week3-advanced-retrieval`](./week3-advanced-retrieval) | Reranking, RAGAS 기반 평가 |
| 4주차 | [`week4-siglip`](./week4-siglip) | SigLIP 비전-언어 임베딩 학습 |
| 5주차 | [`week5-multimodal-search`](./week5-multimodal-search) | 텍스트→이미지 멀티모달 검색 구현 |
| 6주차(선택) | [`week6-integration`](./week6-integration) | 통합 미니 프로젝트 (3인 2팀) |

각 주차 폴더의 README에 그 주의 학습 목표·내용·과제·참고자료가 정리되어 있습니다.

## 폴더 구조

```
DRAG/
├── week1-rag-basics/
│   ├── README.md
│   ├── 김가은/
│   ├── 김민석/
│   ├── 김지수/
│   ├── 서태영/
│   ├── 유연/
│   └── 임지민/
├── week2-mini-chatbot/   (동일 구조)
├── week3-advanced-retrieval/   (동일 구조)
├── week4-siglip/   (동일 구조)
├── week5-multimodal-search/   (동일 구조)
└── week6-integration/
    ├── README.md
    ├── team1/
    └── team2/
```

1~5주차는 멤버별 폴더에 각자 실습 코드를 올리고, 6주차는 3인씩 구성된 team1 / team2 폴더에 팀 단위로 결과물을 올립니다.

## 사용 자료 및 환경

- **교재**: 테디노트 랭체인 한국어 튜토리얼, DeepLearning.AI 단기 코스(LangChain, Advanced RAG), Hugging Face Vision-Language Models 자료
- **사용 툴 및 언어**: Python, LangChain, Hugging Face Transformers(SigLIP), FAISS/Chroma, OpenAI API
- **참고자료(심화)**: LangChain 공식 문서, SigLIP 논문, Self-RAG/HyDE 논문, RAGAS 공식 문서
  (※ 심화 학습용이며 필수로 읽지 않아도 됩니다 — 주요 학습은 위 교재의 강의·튜토리얼로 진행)

## 진행 방식

- 이론 주차(1·3·4주): 전원 발표자료·실습 코드 준비 → 당일 사다리타기로 발표자 선정 → 나머지는 질문·코드 비교
- 구현 주차(2·5주): 역할 분담 없이 전원 각자 구현 → 세션에서 모여 디버깅·이슈 공유
- 통합 주차(6주, 선택): 3인 2팀으로 나누어 팀별 통합 데모 완성 후 비교·발표
