# 김가은의 week5-multimodal-search 실습

## 텍스트 → 이미지 검색 (SigLIP + FAISS)

`week5_text_to_image_search.ipynb`

smol-vision [Image Search with MetaCLIP2](https://github.com/merveenoyan/smol-vision/blob/main/Image_Search_with_MetaCLIP2.ipynb) 노트북을 베이스로, 모델을 SigLIP으로 바꿔 텍스트로 이미지를 찾는 검색 시스템을 구현했습니다.

### 흐름
1. 이미지 데이터셋 불러오기 (Flickr8k 등에서 500장, 안 되면 원본 노트북의 `merve/food`)
2. 오프라인 인덱싱: SigLIP 이미지 임베딩 → `normalize_L2` → FAISS `IndexFlatIP`
3. 온라인 검색: 텍스트 질의 임베딩 → `index.search()` → 상위 k장 + 코사인 유사도 + SigLIP 짝일 확률
4. 인덱스 저장/불러오기 (`faiss.write_index` / `read_index`)
5. (선택) Recall@K: 캡션으로 검색했을 때 원래 이미지가 상위 K장에 드는 비율
6. (선택) Gradio 검색 데모

### 원본에서 바꾼 점
| 항목 | 원본 (MetaCLIP2) | 이 노트북 (SigLIP) |
|---|---|---|
| 모델 | `facebook/metaclip-2-worldwide-giant` | `google/siglip-base-patch16-256-multilingual` (한국어 질의용) |
| 텍스트 입력 | `processor.tokenizer([prompt])` | `padding="max_length"` 필수 |
| 임베딩 | 결과가 바로 벡터 | transformers 5.x는 `.pooler_output` |
| 인덱스 | `IndexFlatL2(1280)` | `IndexFlatIP(768)` (정규화 후 내적 = 코사인) |
| 점수 | L2 거리 | 코사인 + `sigmoid(t·cos + b)` 짝일 확률 |

### 결과
<!-- Colab에서 실행한 뒤 검색 결과 캡처와 Recall@K 숫자를 여기에 붙여 주세요 -->

### 막힌 점
<!-- 세션에서 공유할 막힌 점 1~2개 -->
