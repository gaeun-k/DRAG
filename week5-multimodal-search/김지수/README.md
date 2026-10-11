# 김지수의 week5-multimodal-search 실습

# 5주차: 멀티모달 검색 시스템 구축


## 구현 기능

### 1. 텍스트 → 이미지 검색

- 캡션이 포함된 Flickr8k 이미지 300장을 불러옵니다.
- 다국어 SigLIP으로 이미지 임베딩을 생성하고 L2 정규화합니다.
- 정규화된 벡터를 FAISS `IndexFlatIP`에 저장하여 코사인 유사도 기준으로 검색합니다.
- 영어·한국어 검색어를 임베딩하고 유사한 이미지 상위 K장과 점수를 출력합니다.
- 기존 사진 및 웹에서 추가한 사진도 검색할 수 있습니다.
- 영어·한국어 검색 결과를 PNG와 JSON으로 저장합니다.
- 선택적으로 캡션 검색의 Recall@1, Recall@5, Recall@10을 계산합니다.

```text
이미지 → SigLIP 이미지 임베딩 → 정규화 → FAISS 저장
텍스트 질의 → SigLIP 텍스트 임베딩 → FAISS 검색 → 상위 K장 출력
```

### 2. 문서 기반 RAG 챗봇

처음 준비할 때 다음 공식 문서 6개를 내려받아 로컬에 저장합니다. 질문할 때마다 웹을 검색하지 않고 저장된 문서 안에서 검색합니다.

| 문서 출처 | 내용 |
| --- | --- |
| Hugging Face Transformers | SigLIP 모델 설명 |
| Hugging Face Transformers | Pipeline 사용 안내 |
| Hugging Face Transformers | RAG 모델 설명 |
| FAISS | 시작하기 |
| FAISS | 거리 지표와 코사인 유사도 |
| FAISS | 인덱스 저장과 불러오기 |

문서를 토큰 기준으로 작은 청크로 나누고, 다국어 MiniLM으로 임베딩해 문서용 FAISS 인덱스를 생성합니다. 질문과 관련된 후보를 최대 8개 검색한 뒤, 근거 3개를 Qwen 모델에 전달해 짧은 한국어 답변을 생성합니다. 참고 문서 내용과 원문 링크도 함께 표시합니다.

```text
문서 다운로드 → 청킹 → MiniLM 임베딩 → FAISS 저장
질문 → 관련 문서 검색 → 근거 3개 → Qwen 답변 생성 → 출처 표시
```

대화는 화면에 누적되지만, 검색에는 현재 질문만 사용합니다. 이전 대화를 다음 질문의 검색 문맥으로 자동 사용하지는 않습니다.

### 3. 3주차 개선 기법 적용 범위

**bge Cross-Encoder 리랭킹을 선택 옵션**으로 제공합니다. 옵션을 켜면 검색 후보를 재정렬한 뒤 근거를 선택합니다. 첫 사용에는 추가 모델 다운로드가 필요하며 CPU에서 실행하므로 기본값은 꺼짐입니다.

Colab 실행 부담을 줄이기 위해 HyDE, Self-RAG, RAG-Fusion, RAGAS는 기본 파이프라인에 포함하지 않았습니다. 이번 구현은 이미지 검색과 문서 챗봇의 통합에 초점을 맞췄습니다.

## 사용 모델과 도구

| 용도 | 모델 / 도구 |
| --- | --- |
| 이미지·텍스트 임베딩 | `google/siglip-base-patch16-256-multilingual` |
| 문서 임베딩 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 답변 생성 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 선택 리랭킹 | `BAAI/bge-reranker-v2-m3` |
| 벡터 검색 | FAISS |
| 웹 화면 | Gradio |

## Colab 실행 방법

1. `5주차_이미지검색_문서챗봇.ipynb`를 Colab에 업로드합니다.
2. **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
3. 설치 셀을 실행합니다.
4. **1. 실행 코드와 설정**의 코드 저장 셀과 바로 아래 불러오기 셀을 순서대로 실행합니다.
5. 이미지 준비 → 문서 준비 → 웹 실행 셀을 실행합니다.
6. 출력되는 `gradio.live` 링크를 열어 이미지 검색 / 문서 검색을 사용합니다.

별도 Python 파일 업로드와 API 키는 필요 없습니다. 첫 문서 질문 때 답변 모델을 불러옵니다. 질문 전에 준비하고 싶으면 웹 실행 전에 `w5.load_llm()`을 실행합니다.

### 설정과 사진 추가

실행 코드 셀에서 모델, 이미지 수, 제목, 색상 등을 직접 수정할 수 있습니다. 수정 후에는 코드 저장 셀과 불러오기 셀을 다시 실행하고, 해당 준비 셀도 실행합니다.

- 기본 사진: `/content/images/`에 저장
- 로고: `/content/logo.png`에 저장
- 4주차 노트북에서 이어서 실행: 기존 `image_paths`를 자동으로 사용
- 웹에서 사진 추가: 사진 선택 후 **이미지 준비** 버튼 클릭

기존 `SigLIP_웹검색_Colab.py` 파일은 수정하지 않았습니다.

## 실행 부담을 줄인 방법

- 이미지 데이터 수를 300장으로 제한하고 스트리밍으로 읽습니다.
- 이미지 임베딩을 배치 8장씩 처리합니다.
- 이미지·문서 FAISS 인덱스를 캐시하여 같은 런타임에서 재실행할 때 재사용합니다.
- 문서 임베딩과 선택 리랭킹은 CPU에서 실행합니다.
- 답변 모델은 첫 문서 질문 때 불러와 이미지 검색 화면을 먼저 엽니다.
- 답변 생성은 근거 3개와 최대 220개 새 토큰으로 제한합니다.
- 모델을 사용하는 웹 요청은 공통 큐에서 순차 처리합니다.

최초 다운로드 시간은 네트워크 환경에 따라 달라집니다. Colab 런타임이 삭제되면 `/content`의 캐시도 사라집니다. 지속 보관하려면 Google Drive를 마운트하고 데이터 준비 전에 `w5.CACHE`를 Drive 폴더로 변경해야 합니다.

## 결과 저장과 평가

웹 실행 이후 결과 캡처 셀에서 같은 뜻의 영어·한국어 질의를 비교합니다.

- 영어: `a dog running in the snow`
- 한국어: `눈 위에서 달리는 강아지`

결과는 `week5-multimodal-search/지수/results/`에 저장됩니다.

| 파일 | 내용 |
| --- | --- |
| `english_search.png` | 영어 질의의 상위 5장과 점수 |
| `korean_search.png` | 한국어 질의의 상위 5장과 점수 |
| `english_search.json`, `korean_search.json` | 질의, 이미지 정보, 점수 |
| `recall.json` | 선택 평가를 실행했을 때 생성되는 Recall@K 결과 |

Recall@K는 앞 50장의 첫 영어 캡션을 질의로 사용하고, 정답 이미지 ID가 상위 K개에 포함됐는지 계산합니다. 기본 300장과 추가 사진으로 구성된 후보 내 평가이므로 전체 Flickr8k 성능과 직접 비교할 수 없습니다.


### 캐시 유지의 제약

처음 실행할 때는 모델 다운로드와 이미지 임베딩 준비가 필요합니다. 데이터 수 제한, 배치 처리, FAISS 캐시, 답변 모델의 지연 로딩으로 부담을 줄였습니다. 다만 Colab 런타임이 삭제되면 로컬 캐시가 사라져 다시 준비해야 하는 제약이 남아 있습니다.


- [smol-vision: Image Search with MetaCLIP2](https://github.com/merveenoyan/smol-vision/blob/main/Image_Search_with_MetaCLIP2.ipynb)
- [다국어 SigLIP 모델](https://huggingface.co/google/siglip-base-patch16-256-multilingual)
- [Flickr8k 데이터셋](https://huggingface.co/datasets/jxie/flickr8k)
- [FAISS 공식 문서](https://github.com/facebookresearch/faiss/wiki)
