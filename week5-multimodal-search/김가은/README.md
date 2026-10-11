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

**영어 전용 vs 다국어 모델** (노트북 6-3)

| 모델 | R@1 | R@5 | R@10 | 한↔영 1위 같음 | 한↔영 top-5 겹침 |
|---|---|---|---|---|---|
| 다국어 (`siglip-base-patch16-256-multilingual`) | 77.0% | 96.4% | 99.0% | 33% | 62% |
| 영어 전용 (`siglip-base-patch16-224`) | 77.4% | 95.8% | 98.4% | 33% | 58% |

**한국어 질의 12개의 1위 결과가 맞는지** (노트북 6-2·6-3의 1위 캡션을 보고 판단, ✓ 맞음 / △ 애매 / ✗ 틀림)

| 한국어 질의 | 다국어 모델 1위 | | 영어 전용 모델 1위 | |
|---|---|---|---|---|
| 잔디밭에서 뛰는 강아지 | A brown dog running | ✓ | A beagle walks through the grass . | △ |
| 해변에서 노는 아이들 | A boy and a girl playing on the beach . | ✓ | A boy and a girl playing on the beach . | ✓ |
| 눈밭에서 노는 개 | A dog in the snow | ✓ | A dog in the snow | ✓ |
| 자전거를 타는 남자 | A biker races . | ✓ | A male and female on a one seat bike ride ... | ✓ |
| 암벽을 오르는 사람 | A man dressed in grey climbing a large brown rock | ✓ | A man fiddles with a shopping bag ... | ✗ |
| 수영장으로 뛰어드는 소년 | A boy in a swimsuit jumping into water . | ✓ | A toddler is making a splash inside a blue paddling pool | △ |
| 그네를 타는 여자아이 | A lady is swinging and smiling . | ✓ | A girl with a black swimsuit plays in the sprinkler | ✗ |
| 도시 거리를 걷는 사람들 | A crowd of people walk down a busy sidewalk . | ✓ | A crowd of people walk down a busy sidewalk . | ✓ |
| 묘기를 부리는 스케이트보더 | A guy riding something down steps . | △ | A man wearing a red helmet jumps up while riding ... | △ |
| 축구를 하는 사람들 | A bunch of people playing soccer . | ✓ | A bunch of people playing soccer . | ✓ |
| 빨간 셔츠를 입은 남자 | A closeup of an elderly man . | ✗ | A closeup of an elderly man . | ✗ |
| 싸우는 개 두 마리 | Two dogs jumping on one another on a grassy area | ✓ | Two Dobermans fighting or playing together ... | ✓ |
| **합계** | | **✓ 10 · △ 1 · ✗ 1** | | **✓ 6 · △ 3 · ✗ 3** |

※ 캡션만 보고 판단한 것이라, 발표 전에 노트북의 사진으로 한 번 더 확인하기

### 알게 된 점
- **영어 실력은 두 모델이 같다**: 영어 캡션 Recall@K는 다국어 77.0% vs 영어 전용 77.4%로 거의 같다.
- **한국어는 다국어 모델이 확실히 낫다**: 한국어 질의 12개의 1위 결과를 보면 다국어 모델은 10개가 맞았고, 영어 전용 모델은 6개만 맞았다 (암벽, 그네 등에서 엉뚱한 사진). 한국어 질문을 받을 6주차 챗봇에는 다국어 모델을 쓰는 게 맞다.
- **영어 전용 모델도 한국어를 조금은 알아듣는다**: 해변, 눈, 축구, 거리처럼 일부 질의는 영어 전용 모델도 맞혔다. 완전히 못 알아듣는 건 아니지만 절반 정도는 틀린다.
- **지표를 잘 골라야 한다**: '한↔영 1위 같음'은 두 모델 모두 33%로 차이가 없게 나왔다. 정답 사진이 여러 장이라 한국어 질의가 영어와 '다른 맞는 사진'을 찾아도 불일치로 세기 때문이다. 실제 품질 차이는 1위 사진이 맞는지 직접 보고서야 드러났다 (3주차 RAGAS처럼 지표만 믿지 말고 결과를 같이 볼 것).
- **때로는 한국어 질의가 영어보다 낫다**: "a little girl on a swing"은 엉뚱한 사진(놀고 있는 남자아이)이 1위였는데, "그네를 타는 여자아이"는 그네 타는 사진을 찾았다.
- **질의 형식은 순위에 거의 영향 없음**: `This is a photo of …`로 감싸도 Recall@K 차이가 1%p 안팎.
- **확률은 '거르기' 용도로 유용하지만 기준 잡기가 어렵다**: 데이터에 없는 고양이를 검색하면 확률 0.0%로 걸러진다. 반면 맞는 결과의 1위 확률도 8%(수영장)~97%(거리)로 질의마다 차이가 크다.
- **색·옷차림 같은 세부 조건에 약하다**: "빨간 셔츠를 입은 남자"는 영어·한국어, 두 모델 모두 빨간 셔츠가 없는 노인 얼굴 사진을 1위로 찾았다.

### 막힌 점
<!-- 직접 겪은 점으로 고쳐 주세요. 아래는 실행 로그에서 보인 것들입니다 -->
- `pip install` 시 `cudf-cu13`과 `pyarrow` 버전 충돌 경고가 나왔지만 실행에는 문제 없었다.
- HF_TOKEN 없이 실행하면 Hugging Face 요청 경고가 나온다 (다운로드는 정상).
- 확률 기준값(`MIN_PROB`)을 몇으로 잡아야 할지 애매하다. 맞는 결과의 확률도 질의마다 8~97%로 차이가 커서 하나의 기준으로 거르기 어렵다.
- 영어/한국어 비교 지표('1위 같음', 'top-5 겹침')가 실제 품질 차이를 보여주지 못했다. 한국어 정답 데이터가 없어서 결국 1위 결과를 직접 보고 판단했다.
