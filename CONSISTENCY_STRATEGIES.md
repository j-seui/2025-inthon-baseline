# Expression Consistency 향상 전략

이 문서는 수식 계산 모델의 Expression Consistency를 향상시키기 위해 구현된 전략들을 설명합니다.

## 1. Consistency Loss (일관성 손실)

### 개념
같은 값을 가지는 서로 다른 표현식(예: `2+3`과 `3+2`)이 동일한 출력 분포를 생성하도록 강제합니다.

### 구현
```python
consistency_lambda = 0.1  # TrainConfig에서 설정
```

- **KL Divergence 사용**: 원본 표현식과 증강된 표현식의 출력 확률 분포 차이를 최소화
- **작동 방식**: 
  - 원본 표현식 `2+3` → logits1
  - 증강 표현식 `3+2` → logits2
  - Loss = KL(softmax(logits2) || softmax(logits1))

### 효과
- 모델이 표현식의 형태가 아닌 의미에 집중하도록 유도
- 같은 값의 다양한 표현에 대해 일관된 예측 생성

---

## 2. Hard Negative Mining (어려운 부정 샘플 학습)

### 개념
비슷하게 생긴 표현식이지만 다른 값을 가지는 경우를 구별하도록 학습합니다.

### 구현
```python
use_hard_negatives = True
hard_neg_lambda = 0.05
```

- **선택 기준**:
  - 같은 배치 내에서 입력 길이가 비슷한 쌍
  - 하지만 출력 값이 다른 경우
  - 예: `12+34` vs `12+35` (길이 같지만 결과 다름)

- **학습 방법**:
  - 두 표현식의 출력 logits를 코사인 유사도로 비교
  - 다른 값이므로 유사도가 낮아지도록 margin-based loss 적용
  - Loss = ReLU(cosine_sim + 0.2)

### 효과
- 세밀한 차이를 구별하는 능력 향상
- 비슷한 패턴에 속지 않고 정확한 계산 수행

---

## 3. Depth Distribution Control (깊이 분포 조정)

### 개념
너무 단순한(depth 0-1) 데이터가 과도하게 많으면 복잡한 표현식 처리 능력이 저하됩니다.

### 구현
```python
train_dataset = ArithmeticDataset(
    depth_weights={0: 0.15, 1: 0.15, 2: 0.3, 3: 0.25, 4: 0.15}
)
```

- **균형 잡힌 분포**: 각 깊이별로 적절한 비율 할당
- **복잡도 증가**: depth 2-3에 더 많은 가중치 부여
- **단순 샘플 감소**: depth 0-1은 15%로 제한

### 효과
- 다양한 복잡도의 표현식 처리 능력 향상
- 단순한 패턴 암기가 아닌 실제 계산 학습

---

## 4. Enhanced Commutativity Augmentation (교환법칙 증강 강화)

### 개념
교환법칙(a+b = b+a, a×b = b×a)을 더 적극적으로 학습합니다.

### 구현
```python
# augmentation.py
apply_commutativity(ast, rng)  # 95% 확률
if rng.random() < 0.5:
    apply_commutativity(ast, rng)  # 추가 적용
# ... 다른 변환들
if rng.random() < 0.7:
    apply_commutativity(ast, rng)  # 마지막에 한 번 더
```

### 효과
- 피연산자 순서에 불변한 표현 학습
- 다양한 형태의 동등 표현식 생성

---

## 5. Canonical Form (정규화 형태) - 선택적

### 개념
입력 표현식을 일관된 정규 형태로 변환하여 학습합니다.

### 구현
```python
canonicalize_input = False  # 기본적으로 비활성화
```

- **정규화 규칙**:
  - 교환 가능한 피연산자는 사전순으로 정렬
  - 예: `3+2` → `2+3`, `5*2+3` → `3+2*5`

### 장단점
- **장점**: 입력 형태가 통일되어 학습이 쉬움
- **단점**: 실제 환경의 다양한 입력 형태에 대한 robustness 감소
- **권장**: 초기 학습 단계에서만 사용하거나 비활성화

---

## 전체 Loss 함수

```python
total_loss = (
    ce_loss +                              # 기본 cross-entropy
    law_lambda * law_loss +                # augmentation MSE loss
    consistency_lambda * consistency_loss + # KL divergence
    hard_neg_lambda * hard_neg_loss        # margin-based contrastive
)
```

## 권장 하이퍼파라미터

```python
TrainConfig(
    law_lambda=0.15,           # 증강 데이터 학습 가중치
    consistency_lambda=0.1,    # 일관성 손실 가중치
    hard_neg_lambda=0.05,      # 어려운 부정 샘플 가중치
    law_num_variants=2,        # 각 표현식당 변형 개수
    use_hard_negatives=True,   # hard negative mining 활성화
)
```

## 사용 예시

```python
# train.py를 실행하면 자동으로 모든 전략이 적용됩니다
python train.py

# 특정 전략만 조절하고 싶다면 TrainConfig 수정
train_config = TrainConfig(
    consistency_lambda=0.2,    # 더 강한 일관성 강제
    hard_neg_lambda=0.0,       # hard negative 비활성화
)
```
