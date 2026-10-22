# 김가은의 week4-siglip 실습

## SigLIP 발표 자료 + 실습

| 파일 | 내용 |
|---|---|
| `SigLIP_4주차_발표.pptx` | 발표 슬라이드 24장 (개념 16장 + 실습 6장 + 마무리), 슬라이드 노트에 대본 포함 |
| `SigLIP_4주차_발표_대본.md` | 슬라이드별 발표 대본 (약 19분) |
| `SigLIP_발표자료_내용.md` | 초기 슬라이드 구성안 (최종본은 pptx) |

### 발표 구성
1. **배경**: 공동 임베딩 공간, 대조학습, CLIP(= 객관식, Softmax)과 그 약점
2. **SigLIP 핵심**: O/X 퀴즈(Sigmoid Loss), 듀얼 인코더 구조, ViT 패치, 수식과 바이어스 b(초기값 −10), 효율, CLIP vs SigLIP 비교
3. **검색에 적합한 이유**: 이미지는 미리 인덱싱하고 질문만 계산, 점수를 확률로 읽기
4. **실습**: [Inference with multilingual SigLIP](https://github.com/merveenoyan/smol-vision) 노트북 실행 결과
5. **마무리**: 핵심 3줄 요약, 참고자료

### 실습 결과
| 실습 | 모델 | 결과 |
|---|---|---|
| 입력 확인 | `siglip-base-patch16-224` | `input_ids` 3×64 (`padding="max_length"` 필수), `pixel_values` 1×3×224×224 |
| Sigmoid로 짝 확인 (COCO 고양이 사진) | `siglip-base-patch16-224` | 고양이 두 마리 32.0%, 햄버거·강아지 0% |
| 다국어 캡션 (프랑스어·네덜란드어·독일어) | `siglip-base-patch16-256-multilingual` | 프랑스어 '고양이 두 마리 사진' 19.2%, 나머지 0% (`pixel_values` 256×256) |
| Pipeline API | `siglip-base-patch16-224` | '2 cats' 0.1983, 리모컨·비행기 0 |

### 알게 된 점
- SigLIP 확률은 쌍마다 따로 해석하고 CLIP보다 값이 평평하게 나온다 → 절대값보다 다른 후보와 비교해서 읽기
- 모델마다 학습된 t, b가 달라서 확률 크기를 모델끼리 비교하면 안 된다 (같은 문장이 영어 모델 29.5%, 다국어 모델 19.2%)
- Pipeline은 라벨을 `This is a photo of {label}.`로 바꿔 넣어서 직접 계산한 값과 다르다 → 검색에서도 질문 형식을 맞추는 게 좋다
- 사진에 리모컨이 있는데도 0이 나와, 작은 물체나 주변 물체에는 약하다
