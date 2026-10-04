# 4주차 발표자료 내용 — SigLIP 중심 비전-언어 임베딩

> 슬라이드별로 **화면에 들어갈 내용**과 **발표 멘트(스피커 노트)** 를 나눠 정리했습니다.
> 구성: 도입(1~3) → 배경: CLIP과 대조학습(4~7) → SigLIP 핵심(8~13) → 검색에 적합한 이유(14~15) → 실습(16~18) → 마무리(19~20)
> 예상 분량: 20장, 발표 25~30분

---

## Slide 1. 표지

**화면**
- 제목: **SigLIP — Sigmoid Loss로 학습하는 비전-언어 임베딩**
- 부제: DRAG 4주차 · 눈으로 찾는 RAG
- 발표자 이름 / 날짜

---

## Slide 2. 지난 주차 복습 & 이번 주 위치

**화면**
- 1~3주차: **텍스트** RAG — 임베딩 → 벡터DB → 검색 → (Reranking) → 생성
- 4주차: 임베딩 모델을 **이미지까지 다루는 모델**로 교체하려면? → **SigLIP**
- 5주차: SigLIP 임베딩 + 벡터DB로 **텍스트→이미지 검색** 구현

**도식 아이디어**
```
[문서] ─ 텍스트 임베딩 ─┐                 [이미지] ─ SigLIP 이미지 임베딩 ─┐
                       ├→ 벡터DB → 검색      ⇒                              ├→ 벡터DB → 검색
[질문] ─ 텍스트 임베딩 ─┘                 [질문]  ─ SigLIP 텍스트 임베딩 ─┘
```

**발표 멘트**
> 지금까지 RAG 파이프라인에서 "임베딩 모델" 칸은 텍스트 전용이었습니다. 이번 주는 그 칸에 이미지와 텍스트를 **같은 벡터 공간**에 넣어주는 모델, SigLIP을 끼워 넣는 준비 단계입니다.

---

## Slide 3. 오늘의 목표

**화면**
1. 비전-언어 임베딩(공동 임베딩 공간)이 무엇인지 이해한다
2. CLIP의 **Softmax 대조학습**과 SigLIP의 **Sigmoid 대조학습**의 차이를 설명할 수 있다
3. SigLIP이 **이미지-텍스트 검색**에 적합한 이유를 말할 수 있다
4. Hugging Face로 SigLIP **이미지 임베딩을 추출**할 수 있다

---

## Slide 4. 비전-언어 모델(VLM)의 큰 그림

**화면** (HF 블로그 "Vision Language Models Explained" 기반)
- VLM = 이미지 + 텍스트를 입력으로 받는 모델
- 크게 두 부류
  | 구분 | 예시 | 출력 | 용도 |
  |---|---|---|---|
  | **임베딩(듀얼 인코더)형** | CLIP, **SigLIP** | 벡터 | 검색, 분류(zero-shot), 유사도 |
  | **생성형 VLM** | LLaVA, PaliGemma, SmolVLM | 텍스트 | 이미지 QA, 캡셔닝 |
- 포인트: 생성형 VLM 상당수가 **비전 인코더로 SigLIP을 사용** (PaliGemma, Idefics3/SmolVLM 등)

**발표 멘트**
> SigLIP은 그 자체로 검색용 임베딩 모델이면서, 요즘 생성형 VLM의 "눈" 역할을 하는 표준 부품이기도 합니다. 그래서 SigLIP을 이해하면 6주차 통합 프로젝트에서 생성형 VLM까지 확장할 때도 도움이 됩니다.

---

## Slide 5. 공동 임베딩 공간 (Joint Embedding Space)

**화면**
- 이미지와 텍스트를 **같은 차원의 벡터 공간**에 매핑
- 의미가 같으면 가깝게, 다르면 멀게
- 유사도 = 두 벡터의 **코사인 유사도**(정규화 후 내적)

**도식 아이디어**
```
       🐶 사진  ●───● "a photo of a dog"
                    
  🐱 사진 ●───● "a cute cat"
                              ● "a red car"   ● 🚗 사진
```

**발표 멘트**
> 1주차 텍스트 임베딩에서 "의미가 비슷한 문장은 가까운 벡터"였던 걸 떠올려 보세요. 여기서는 문장과 **사진**이 서로 가까워질 수 있다는 점만 다릅니다. 그래서 텍스트 질문으로 이미지를 검색할 수 있게 됩니다.

---

## Slide 6. 대조학습 (Contrastive Learning)

**화면**
- 학습 데이터: 웹에서 수집한 (이미지, 캡션) **쌍** 수억~수십억 개
- 배치 안에서
  - **Positive**: 짝이 맞는 (이미지 i, 텍스트 i) → 유사도 ↑
  - **Negative**: 짝이 아닌 (이미지 i, 텍스트 j≠i) → 유사도 ↓
- 별도 라벨링 없이 "짝 맞추기"만으로 학습 → 대규모 확장 가능

**도식: 배치 유사도 행렬 (B=4 예시)**
```
            T1    T2    T3    T4
   I1     [ ✅ ]  ❌    ❌    ❌
   I2       ❌  [ ✅ ]  ❌    ❌
   I3       ❌    ❌  [ ✅ ]  ❌
   I4       ❌    ❌    ❌  [ ✅ ]
  → 대각선 = positive (B개), 나머지 = negative (B²−B개)
```

---

## Slide 7. CLIP의 방식: Softmax(InfoNCE) 손실

**화면**
- CLIP (OpenAI, 2021): 이미지 인코더 + 텍스트 인코더, 4억 쌍으로 학습
- 손실: 유사도 행렬의 **각 행/열에 Softmax** → "B개 후보 중 정답 1개 고르기" (B-way 분류)
- 수식
  $$\mathcal{L} = -\frac{1}{2B}\sum_{i=1}^{B}\left(\log\frac{e^{t\,x_i\cdot y_i}}{\sum_{j} e^{t\,x_i\cdot y_j}} + \log\frac{e^{t\,x_i\cdot y_i}}{\sum_{j} e^{t\,x_j\cdot y_i}}\right)$$
  - 이미지→텍스트, 텍스트→이미지 **두 방향**으로 각각 정규화
- **문제점**
  1. 분모(정규화)에 **배치 전체**가 필요 → 여러 GPU/TPU에 흩어진 임베딩을 모두 모아(all-gather) **B×B 전체 행렬**을 계산해야 함 → 메모리·통신 부담
  2. 좋은 성능을 내려면 **매우 큰 배치**(CLIP은 32k)가 필요
  3. 수치 안정성을 위해 추가 연산(최댓값 빼기 등) 필요

**발표 멘트**
> Softmax는 "이 이미지의 정답 캡션은 배치 안의 32,000개 중 몇 번?"을 맞히는 문제입니다. 한 칸의 확률을 계산하려 해도 같은 행의 나머지 모든 칸이 필요하다는 게 핵심 약점입니다.

---

## Slide 8. SigLIP 소개

**화면**
- 논문: *Sigmoid Loss for Language Image Pre-Training* (Zhai et al., Google DeepMind, ICCV 2023, arXiv:2303.15343)
- 핵심 아이디어 한 줄: **Softmax 대신 Sigmoid** — 모든 (이미지, 텍스트) 쌍을 **독립적인 이진 분류**로 학습
- 구조는 CLIP과 거의 동일(듀얼 인코더), **손실 함수만 바꿈** → 그런데 효율과 성능이 좋아짐

---

## Slide 9. SigLIP 구조

**화면**
```
 이미지 ──▶ [ViT 이미지 인코더] ──▶ 풀링 + 선형 투영 ──▶ L2 정규화 ──▶ x_i ─┐
                                                                         ├─▶ 유사도 행렬 t·x·y + b ─▶ Sigmoid Loss
 텍스트 ──▶ [Transformer 텍스트 인코더] ─▶ 풀링 + 선형 투영 ─▶ L2 정규화 ─▶ y_j ─┘
```
- **듀얼 인코더(Two-Tower)**: 이미지·텍스트 인코더가 **완전히 분리**되어 따로 계산 가능
- 이미지 인코더: Vision Transformer (패치 단위로 자름, 예: 16×16 패치)
  - 이미지 → 패치 → 토큰 시퀀스 → Transformer
  - 풀링: MAP(Multihead Attention Pooling) head로 하나의 벡터로 요약
- 텍스트 인코더: Transformer, 최대 길이 64 토큰
- 학습 가능한 파라미터: **온도 t**, **바이어스 b** (Slide 12)
- 모델 이름 읽는 법: `google/siglip-base-patch16-224`
  - base = 모델 크기 / patch16 = 패치 크기 16 / 224 = 입력 해상도 224×224
  - `so400m` = 계산 최적 형태로 설계한 약 4억 파라미터 ViT (Shape-Optimized)

---

## Slide 10. Sigmoid Loss 수식

**화면**
$$\mathcal{L} = -\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|}\log\sigma\big(z_{ij}\,(t\,x_i\cdot y_j + b)\big)$$
- $x_i$: i번째 이미지 임베딩, $y_j$: j번째 텍스트 임베딩 (L2 정규화)
- $z_{ij} = +1$ (i=j, 짝 맞음) / $-1$ (i≠j, 짝 아님)
- $\sigma$: 시그모이드 함수 → 각 칸의 "짝일 확률"
- 각 칸이 **독립된 로지스틱 회귀(이진 분류)** 문제
- **배치 전체 합으로 나누는 정규화 항이 없음** ← 이게 모든 장점의 출발점

**발표 멘트**
> CLIP은 행 단위로 "누가 정답인가"를 고르고, SigLIP은 칸 하나하나에 "이 둘이 짝인가 아닌가"를 예/아니오로 답합니다. 그래서 칸 하나의 손실을 계산할 때 다른 칸을 볼 필요가 없습니다.

---

## Slide 11. Softmax vs Sigmoid 직관 비교 (숫자 예시)

**화면** — 이미지 I1에 대한 유사도 점수(로짓)가 `[T1: 5.0, T2: 4.8, T3: -2.0]` 일 때
| | T1 (정답) | T2 | T3 |
|---|---|---|---|
| **Softmax** (행 정규화) | 0.55 | 0.45 | 0.00 |
| **Sigmoid** (칸별 독립) | 0.99 | 0.99 | 0.12 |

- Softmax: 확률의 합이 1 → **상대적** 점수. "T1이 T2보다 조금 낫다"만 알려줌
- Sigmoid: 각 칸이 **절대적** 점수. T2가 실제로 비슷한 캡션이면 둘 다 높을 수 있음
- → Sigmoid 학습은 T2도 낮추도록(음성) 직접 신호를 줌 / 배치 구성에 덜 의존

> (위 Sigmoid 값은 바이어스 b=0 기준 단순 예시. 실제 SigLIP은 b가 음수로 학습되어 짝이 아닌 쌍은 확률이 0 근처로 내려감)

---

## Slide 12. 온도(t)와 바이어스(b) — 왜 b가 필요한가

**화면**
- 온도 $t = \exp(t')$, 초기값 $t' = \log 10$ → 점수의 **스케일** 조절 (CLIP에도 있음)
- 바이어스 $b$, 초기값 **−10** ← SigLIP에서 새로 추가
- 이유: **양성/음성 불균형**
  - 배치 크기 B일 때 positive B개 vs negative **B²−B개**
  - 예) B=32k → 짝 1개당 비(非)짝 약 32,000개
  - 학습 초기에 모든 쌍을 "짝 아님"으로 보는 쪽에서 시작하게 해서, 초반에 거대한 음성 손실 때문에 학습이 흔들리는 것을 방지
- 결과: 학습이 안정적이고, 출력 확률이 **실제 "짝일 확률"처럼 보정(calibrated)** 됨

---

## Slide 13. SigLIP의 장점 ① — 효율성 (메모리 & 분산 학습)

**화면**
- Softmax: 모든 장치의 임베딩을 모아 **B×B 전체 행렬**로 정규화 필요
- Sigmoid: 칸끼리 독립 → **"청크(chunked)" 구현** 가능
  1. 각 장치는 자기 이미지/텍스트 블록으로 손실 계산
  2. 텍스트 임베딩을 옆 장치로 **돌려가며(permute)** 다음 블록 계산
  3. 결과 손실만 더하면 끝
- 장치당 메모리: $O(B^2)$ → $O(b^2)$ (b = 장치당 배치 크기)
- 논문 사례
  - **SigLiT** (이미지 인코더 고정, 텍스트만 학습): TPUv4 **4개 칩, 2일** 만에 ImageNet zero-shot **84.5%**
  - 배치 크기를 **100만(1M)** 까지 늘려서 실험 가능했음

**도식 아이디어**: 장치 3개가 3×3 블록 행렬을 대각선부터 한 칸씩 돌아가며 채우는 그림

---

## Slide 14. SigLIP의 장점 ② — 성능 & 배치 크기 실험

**화면**
- **작은 배치에서 Softmax보다 확실히 우수** (예: 4k~16k 배치)
- 배치를 키우면 둘의 차이는 줄어들고, **약 32k에서 성능이 포화**
  - 1M까지 키워도 추가 이득 거의 없음 → "무조건 큰 배치"가 정답은 아니다
- **노이즈에 강함**: 웹 데이터 특성상 엉뚱한 캡션이 많은데, 노이즈를 인위적으로 넣은 실험에서 Softmax보다 성능 저하가 적음
- 정리 표
  | | CLIP (Softmax) | SigLIP (Sigmoid) |
  |---|---|---|
  | 학습 문제 | B-way 다중 분류 | B² 개 이진 분류 |
  | 정규화 | 행/열 전체 필요 | 불필요 |
  | 방향 | 이미지→텍스트, 텍스트→이미지 2번 | 대칭이라 1번 |
  | 메모리 | 전체 B×B 행렬 | 장치별 블록 |
  | 작은 배치 성능 | 낮음 | 높음 |
  | 출력 해석 | 상대 확률 (합=1) | 쌍별 절대 확률 |
  | 추가 파라미터 | 온도 t | 온도 t + 바이어스 b |

---

## Slide 15. SigLIP이 이미지-텍스트 검색에 적합한 이유

**화면**
1. **듀얼 인코더 → 미리 인덱싱 가능**
   - 이미지 임베딩은 오프라인에서 한 번만 계산 → FAISS/Chroma에 저장
   - 질의 시에는 **텍스트 1개만 인코딩** → 빠른 근사 최근접 탐색(ANN)
   - (질의·이미지를 같이 넣어야 하는 cross-encoder 구조는 매번 전체를 다시 계산해야 해서 대규모 검색에 부적합)
2. **공동 임베딩 공간** → 텍스트→이미지, 이미지→텍스트, 이미지→이미지 검색 모두 같은 인덱스로 가능
3. **쌍별 독립 점수** → "이 이미지가 질의와 관련 있을 확률"로 해석 가능 → **임계값(threshold) 필터링**이 쉬움 (Softmax는 후보 집합에 따라 값이 바뀜)
4. **작은 배치·노이즈에도 강하게 학습된 고품질 임베딩** → 공개 체크포인트 성능이 좋음
5. **다양한 공개 모델**: base부터 so400m, 다국어 버전까지 HF에서 바로 사용 가능

**발표 멘트**
> 3주차 Reranking에서 bi-encoder로 후보를 뽑고 cross-encoder로 재정렬했던 것 기억나시죠? SigLIP은 그중 **bi-encoder 역할**입니다. 그래서 5주차에 대량의 이미지를 미리 벡터DB에 넣어두고 텍스트로 빠르게 찾는 구조가 가능합니다.

---

## Slide 16. 실습 ① — 모델 로드 & 이미지 임베딩 추출

**화면**
```python
import torch
from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor

ckpt = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(ckpt).eval()
processor = AutoProcessor.from_pretrained(ckpt)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_emb = model.get_image_features(**inputs)        # (1, 768)

image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)  # L2 정규화
print(image_emb.shape)
```
- `get_image_features`: 이미지 → 임베딩 벡터 (base 기준 768차원)
- 검색에 쓸 때는 **L2 정규화** 후 저장 → 내적 = 코사인 유사도

---

## Slide 17. 실습 ② — 텍스트 임베딩 & 이미지-텍스트 매칭

**화면**
```python
texts = ["a photo of 2 cats", "a photo of a dog", "a photo of a car"]

inputs = processor(text=texts, images=image,
                   padding="max_length",   # ★ SigLIP은 학습 때와 동일하게 max_length 패딩 필수
                   return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

probs = torch.sigmoid(outputs.logits_per_image)  # ★ softmax가 아니라 sigmoid
for t, p in zip(texts, probs[0]):
    print(f"{p:.1%}  {t}")
```
- 주의 포인트 (CLIP 코드 그대로 쓰면 틀리는 부분)
  1. `padding="max_length"` (max 64 토큰) — 안 하면 결과가 이상하게 나옴
  2. 확률은 `torch.sigmoid` — 각 캡션 확률의 합이 1이 아님 (정상)
  3. 프롬프트는 `"This is a photo of ..."` / `"a photo of ..."` 형태가 잘 동작

---

## Slide 18. 실습 ③ — 여러 이미지 배치 임베딩 (5주차 준비)

**화면**
```python
from pathlib import Path
import numpy as np

paths = sorted(Path("images").glob("*.jpg"))
embs = []
for i in range(0, len(paths), 32):
    batch = [Image.open(p).convert("RGB") for p in paths[i:i+32]]
    inputs = processor(images=batch, return_tensors="pt")
    with torch.no_grad():
        e = model.get_image_features(**inputs)
    embs.append(torch.nn.functional.normalize(e, dim=-1).numpy())

image_embs = np.concatenate(embs)          # (N, 768) → 5주차에 FAISS/Chroma로 인덱싱
np.save("image_embs.npy", image_embs)

# 텍스트 질의로 간단 검색
q = processor(text=["a cat sleeping on a sofa"], padding="max_length", return_tensors="pt")
with torch.no_grad():
    q_emb = torch.nn.functional.normalize(model.get_text_features(**q), dim=-1).numpy()
top5 = (image_embs @ q_emb.T).squeeze().argsort()[::-1][:5]
print([paths[i].name for i in top5])
```
- GPU 사용 시 `model.to("cuda")`, 입력도 `.to("cuda")`
- 이 결과(`image_embs.npy`)가 **5주차 벡터DB 인덱싱의 입력**

---

## Slide 19. 더 알아보기 — 모델 선택 & SigLIP 2

**화면**
| 체크포인트 | 특징 |
|---|---|
| `google/siglip-base-patch16-224` | 가볍고 빠름, 실습용 |
| `google/siglip-so400m-patch14-384` | 고성능, 고해상도(384), 생성형 VLM 비전 인코더로 많이 사용 |
| `google/siglip-base-patch16-256-multilingual` | 다국어 지원 (한국어 질의 실험용) |
| `google/siglip2-base-patch16-224` 등 | **SigLIP 2** (2025) |

- **SigLIP 2** 주요 변화
  - Sigmoid loss는 유지 + 캡셔닝 디코더 손실(LocCa), 자기 증류, 마스크 예측 등 추가 → **위치 정보·밀집(dense) 특징 향상**
  - 다국어 데이터로 학습 → 다국어 검색 성능 향상
  - **NaFlex** 변형: 원본 종횡비를 유지하며 다양한 해상도 입력
- 한국어 질의가 필요하면 multilingual / SigLIP 2 모델을 비교해볼 것 (5주차 실험 거리)

---

## Slide 20. 정리 & 토론 질문

**화면 — 핵심 3줄 요약**
1. SigLIP = CLIP 구조 + **Softmax 대신 Sigmoid 손실** (쌍별 이진 분류)
2. 배치 전체 정규화가 사라져 **메모리 효율 ↑, 작은 배치 성능 ↑, 노이즈에 강함**
3. 듀얼 인코더 + 공동 임베딩 공간 → **미리 인덱싱하는 대규모 이미지 검색에 적합**

**토론 질문 (발표 후 질의응답용)**
- Sigmoid 확률을 그대로 검색 결과 필터링 임계값으로 써도 될까? 몇으로 잡아야 할까?
- 텍스트 RAG의 Reranking(cross-encoder)처럼, 이미지 검색에도 재정렬 단계를 붙인다면 무엇을 쓸 수 있을까? (예: 생성형 VLM으로 재평가)
- 한국어 질의를 영어로 번역해서 넣는 것 vs 다국어 모델을 쓰는 것, 어느 쪽이 나을까?
- 배치가 32k에서 포화된다는 결과는 우리가 파인튜닝할 때 어떤 의미가 있을까?

**참고자료**
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023 — https://arxiv.org/abs/2303.15343
- Tschannen et al., *SigLIP 2*, 2025 — https://arxiv.org/abs/2502.14786
- Radford et al., *CLIP*, 2021 — https://arxiv.org/abs/2103.00020
- Hugging Face Blog, Vision Language Models Explained — https://huggingface.co/blog/vlms
- Hugging Face SigLIP 문서 — https://huggingface.co/docs/transformers/model_doc/siglip

---

## 부록. 예상 질문 & 답변 (발표자 대비용)

**Q1. SigLIP은 CLIP과 구조가 같은데 왜 성능이 더 좋나요?**
A. 손실 함수 차이입니다. Softmax는 배치 안 상대 비교라 배치가 작으면 negative가 부족해 학습 신호가 약하고, Sigmoid는 모든 쌍에 독립적인 신호를 줘서 작은 배치에서도 잘 학습됩니다. 큰 배치(32k 근처)에서는 차이가 줄어듭니다.

**Q2. negative가 압도적으로 많은데 불균형 문제는 없나요?**
A. 그래서 바이어스 b(초기값 −10)를 둡니다. 초기에 "대부분 짝이 아니다"에서 출발하도록 해서 불균형으로 인한 큰 초기 손실을 막습니다.

**Q3. 추론(검색)할 때도 Sigmoid를 써야 하나요?**
A. 순위만 필요하면 정규화된 임베딩의 내적(코사인 유사도)으로 충분합니다. Sigmoid는 단조 함수라 순위가 바뀌지 않습니다. 확률값(예: 필터링 임계값)이 필요할 때만 `sigmoid(t·sim + b)`를 적용하면 됩니다.

**Q4. 왜 `padding="max_length"`가 필수인가요?**
A. 학습 시 텍스트를 항상 64 토큰으로 패딩한 상태로 학습했기 때문입니다. 다른 방식으로 패딩하면 텍스트 임베딩 분포가 달라져 성능이 떨어집니다.

**Q5. 텍스트 임베딩 모델(1주차)로 이미지 캡션을 만들어 검색하는 것과 뭐가 다른가요?**
A. 캡션 방식은 캡션 생성 단계에서 정보가 손실되고 비용이 큽니다. SigLIP은 이미지 자체를 임베딩하므로 캡션에 안 적힌 시각 정보(색, 구도, 분위기)도 검색에 반영됩니다. 6주차에서 두 방식을 결합하는 것도 가능합니다.
