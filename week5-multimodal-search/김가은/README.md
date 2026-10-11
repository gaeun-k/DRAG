# 김가은의 week5-multimodal-search 실습

## 텍스트 → 이미지 검색 (SigLIP + FAISS)

`week5_text_to_image_search.ipynb`

smol-vision [Image Search with MetaCLIP2](https://github.com/merveenoyan/smol-vision/blob/main/Image_Search_with_MetaCLIP2.ipynb) 노트북을 베이스로, 모델을 SigLIP으로 바꿔 텍스트로 이미지를 찾는 검색 시스템을 구현했습니다.

### 흐름
1. 이미지 데이터셋 불러오기: `jxie/flickr8k` test split에서 500장 (스트리밍)
2. 오프라인 인덱싱: SigLIP 이미지 임베딩 → `normalize_L2` → FAISS `IndexFlatIP`
3. 온라인 검색: 텍스트 질의 임베딩 → `index.search()` → 상위 k장 + 코사인 유사도 + SigLIP 짝일 확률
4. 인덱스 저장/불러오기 (`faiss.write_index` / `read_index`)
5. Recall@K: 캡션으로 검색했을 때 원래 이미지가 상위 K장에 드는 비율
6. 영어 vs 한국어 질의 비교, 영어 전용 모델 vs 다국어 모델 비교
7. Gradio 검색 데모

### 원본에서 바꾼 점
| 항목 | 원본 (MetaCLIP2) | 이 노트북 (SigLIP) |
|---|---|---|
| 모델 | `facebook/metaclip-2-worldwide-giant` | `google/siglip-base-patch16-256-multilingual` (한국어 질의용) |
| 데이터 | `merve/food` (16장) | `jxie/flickr8k` 500장 (캡션 5개씩) |
| 텍스트 입력 | `processor.tokenizer([prompt])` | `padding="max_length"` 필수 |
| 임베딩 | 결과가 바로 벡터 | transformers 5.x는 `.pooler_output` |
| 인덱스 | `IndexFlatL2(1280)` | `IndexFlatIP(768)` (정규화 후 내적 = 코사인) |
| 점수 | L2 거리 | 코사인 + `sigmoid(t·cos + b)` 짝일 확률 |

### 결과

**실행 환경**: Colab T4 · 모델 `siglip-base-patch16-256-multilingual` (t = 111.00, b = −12.89) · `jxie/flickr8k` 500장 · 인덱싱 10초

**검색 예시**

| 질의 | 1위 결과 (캡션) | 코사인 | 짝일 확률 |
|---|---|---|---|
| a dog running on the grass | A brown dog running | 0.126 | 74.7% |
| 잔디밭에서 뛰는 강아지 | A brown dog running (영어 질의와 같은 사진) | 0.129 | 81.2% |
| 바다에서 노는 아이들 | A boy and a girl playing on the beach . | 0.114 | 43.9% |
| 빨간 옷을 입은 사람 | An Asian girl performs a routine with two red flags ... | 0.102 | 17.6% |
| a cat sitting on a sofa | (데이터에 고양이 없음) 3장 모두 확률 0.0% → 관련 없음으로 걸러짐 | 0.046 | 0.0% |

**Recall@K** (500장, 각 이미지의 첫 번째 영어 캡션을 질의로 사용)

| 질의 형식 | R@1 | R@5 | R@10 |
|---|---|---|---|
| 캡션 그대로 | 77.0% | 96.4% | 99.0% |
| `This is a photo of {캡션}.` | 77.4% | 95.0% | 98.8% |

**영어 vs 한국어 질의 / 영어 전용 vs 다국어 모델** (노트북 6-2, 6-3)

<!-- 6-2, 6-3 셀을 실행한 뒤 model_df 표의 숫자를 아래에 옮겨 주세요 -->

| 모델 | R@1 | R@5 | R@10 | 한↔영 1위 같음 | 한↔영 top-5 겹침 |
|---|---|---|---|---|---|
| 다국어 (`siglip-base-patch16-256-multilingual`) | 77.0% | 96.4% | 99.0% |  |  |
| 영어 전용 (`siglip-base-patch16-224`) |  |  |  |  |  |

### 알게 된 점
- **한국어 질의도 잘 됨**: 다국어 모델로 "잔디밭에서 뛰는 강아지"를 검색하면 영어 질의와 같은 사진이 1위로 나왔다.
- **질의 형식은 순위에 거의 영향 없음**: `This is a photo of …`로 감싸도 Recall@K 차이가 1%p 안팎. 4주차에서 pipeline과 직접 계산의 점수가 달랐던 건 확률 값 차이였고, 순위는 거의 그대로였다.
- **확률은 '거르기' 용도로 유용**: 데이터에 없는 내용(고양이)을 검색하면 상위 결과도 확률이 0.0%로 나와서, 관련 없는 결과를 걸러낼 수 있다. 원본 노트북에서 k개를 요청하면 무관한 이미지가 섞이던 문제를 확률로 해결할 수 있다.
- **확률 값 자체는 낮게 나오기도 함**: "빨간 옷을 입은 사람"은 결과가 맞는데도 1위 확률이 17.6%였다. 4주차에서 본 것처럼 SigLIP 확률은 평평해서, 기준값(`MIN_PROB`)을 너무 높게 잡으면 맞는 결과도 숨겨진다.

### 막힌 점
<!-- 직접 겪은 점으로 고쳐 주세요. 아래는 실행 로그에서 보인 것들입니다 -->
- `pip install` 시 `cudf-cu13`과 `pyarrow` 버전 충돌 경고가 나왔지만 실행에는 문제 없었다.
- HF_TOKEN 없이 실행하면 Hugging Face 요청 경고가 나온다 (다운로드는 정상).
- 확률 기준값(`MIN_PROB`)을 몇으로 잡아야 할지 애매하다. 질의마다 정답 확률이 17~81%로 차이가 커서 하나의 기준으로 거르기 어렵다.
