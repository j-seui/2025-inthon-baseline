from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TokenizerConfig:
    """토크나이저 관련 설정"""
    input_chars: Optional[list] = None  # 입력 문자 집합 (None이면 기본값 사용)
    output_chars: Optional[list] = None  # 출력 문자 집합 (None이면 기본값 사용)
    add_special: bool = True  # 특수 토큰(PAD, BOS, EOS) 추가 여부


@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정 (Transformer용)"""
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """학습 관련 설정"""
    max_train_steps: Optional[int] = None
    lr: float = 3e-4          # Transformer는 보통 GRU보다 조금 작은 lr가 안정적
    valid_every: int = 200
    max_gen_len: int = 32
    show_valid_samples: int = 5
    num_epochs: int = 10      # 처음엔 10 epoch 정도 돌려보면서 모양 보자
    save_best_path: Optional[str] = None
    law_lambda: float = 0.1
    law_num_variants: int = 1
    law_max_pairs_per_batch: int = 32
    law_seed: int = 0
    curriculum_enabled: bool = False
    curriculum_num_stages: int = 1
    curriculum_stage_steps: Optional[Tuple[int, ...]] = None
    curriculum_stage_epochs: Optional[Tuple[int, ...]] = None
