from __future__ import annotations

from typing import List, Dict, Any

from dataclasses import dataclass

import torch
import torch.nn as nn

from do_not_edit.model_template import BaseModel

# ========================
# Tokenizer (통합 버전)
# ========================

# 특수 토큰 정의
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

# 규정에 맞는 입력/출력 문자 집합
# INPUT_CHARS: 수식 입력에 사용 가능한 모든 문자 (숫자, 연산자, 괄호, 공백)
# OUTPUT_CHARS: 모델이 출력할 수 있는 문자 (숫자만)
INPUT_CHARS = list("0123456789+-*/() ")
OUTPUT_CHARS = list("0123456789")


class CharTokenizer:
    """
    문자 단위 토크나이저
    
    문자열을 문자 단위로 분해하여 정수 인덱스로 변환하는 토크나이저입니다.
    Seq2Seq 모델의 입력/출력을 처리하기 위해 사용됩니다.
    """

    def __init__(self, chars: List[str], add_special: bool):
        vocab = list(chars)
        self.pad = PAD if add_special else None
        self.bos = BOS if add_special else None
        self.eos = EOS if add_special else None

        if add_special:
            vocab = [PAD, BOS, EOS] + vocab

        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str, add_bos_eos: bool) -> List[int]:
        ids: List[int] = []
        if add_bos_eos and self.bos is not None:
            ids.append(self.stoi[self.bos])

        for ch in s:
            idx = self.stoi.get(ch)
            if idx is None:
                raise ValueError(f"Unknown char '{ch}' for tokenizer.")
            ids.append(idx)

        if add_bos_eos and self.eos is not None:
            ids.append(self.stoi[self.eos])
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        s = "".join(self.itos[i] for i in ids if i in self.itos)
        if strip_special and self.bos:
            s = s.replace(self.bos, "")
        if strip_special and self.eos:
            s = s.replace(self.eos, "")
        if strip_special and self.pad:
            s = s.replace(self.pad, "")
        return s

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi.get(PAD, 0)

    @property
    def bos_id(self) -> int:
        return self.stoi.get(BOS, 0)

    @property
    def eos_id(self) -> int:
        return self.stoi.get(EOS, 0)


@dataclass
class BatchTensors:
    """
    배치 처리용 텐서 컨테이너
    
    Attributes:
        src: 토큰 인덱스 텐서 [batch_size, seq_len]
        tgt: 고정 길이 숫자 시퀀스 [batch_size, max_output_len]
    """

    src: torch.Tensor
    tgt: torch.Tensor


def _pad(seqs: List[List[int]], pad_id: int, fixed_len: int | None = None) -> torch.Tensor:
    if fixed_len is not None:
        L = fixed_len
    else:
        L = max(len(s) for s in seqs) if seqs else 1

    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            length = min(len(s), L)
            out[i, :length] = torch.tensor(s[:length], dtype=torch.long)
    return out


def digits_to_string(digits: List[int]) -> str:
    """0~9 리스트를 문자열로 변환 (선행 0 제거, 전부 0이면 '0')."""
    chars = "".join(str(max(0, min(9, int(d)))) for d in digits)
    stripped = chars.lstrip("0")
    return stripped if stripped else "0"


def tokenize_batch(
    batch: Dict[str, List[str]],
    input_tokenizer: CharTokenizer,
    output_tokenizer: CharTokenizer,
    *,
    max_input_length: int,
    max_output_length: int,
) -> BatchTensors:
    """
    배치 데이터를 토크나이징하여 모델 입력용 텐서로 변환.
    
    출력 숫자는 왼쪽을 0으로 패딩하여 길이를 고정합니다.
    """
    del output_tokenizer  # 인터페이스 유지용 (digits 직접 처리)

    src_ids: List[List[int]] = []
    for raw in batch["input_text"]:
        ids = input_tokenizer.encode(raw, add_bos_eos=False)
        if not ids:
            raise ValueError(f"Empty tokenized source for input '{raw}'")
        if len(ids) > max_input_length:
            ids = ids[:max_input_length]
        src_ids.append(ids)

    src = _pad(src_ids, input_tokenizer.pad_id, fixed_len=max_input_length)

    tgt_ids: List[List[int]] = []
    for raw_target in batch["target_text"]:
        target = raw_target.strip()
        if not target.isdigit():
            raise ValueError(f"Target '{raw_target}' is not a non-negative integer string.")
        if len(target) > max_output_length:
            raise ValueError(
                f"Target '{raw_target}' exceeds max_output_length={max_output_length}."
            )
        padded = target.rjust(max_output_length, "0")
        tgt_ids.append([int(ch) for ch in padded])

    tgt = torch.tensor(tgt_ids, dtype=torch.long)
    return BatchTensors(src=src, tgt=tgt)


# ========================
# Dilated 1D CNN Encoder
# ========================

class DilatedConvBlock(nn.Module):
    """
    단일 Dilated 1D Conv 블록 (Pre-LN + Residual)
    입력/출력 shape: [B, L, d_model]
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Pre-LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # Conv1d: 입력 [B, d_model, L] → 출력 [B, d_model, L]
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        """
        residual = x

        # Pre-norm
        x = self.norm(x)

        # [B, L, d_model] → [B, d_model, L]
        x = x.transpose(1, 2)

        # Conv1d
        x = self.conv(x)

        # [B, d_model, L] → [B, L, d_model]
        x = x.transpose(1, 2)

        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class TinyCNNSeq2Seq(nn.Module):
    """
    Dilated 1D CNN 기반 Encoder + 자리별 digit 분류 모델.

    - 입력: 토큰 인덱스 시퀀스 [B, T]
    - 출력: 각 자릿수 0~9 분포 [B, output_length, num_digit_classes]
    """

    def __init__(
        self,
        in_vocab: int,
        out_vocab: int,  # 사용은 안 하지만 인터페이스 유지용
        pad_id: int | None = None,
        **kwargs,
    ):
        super().__init__()

        d_model = kwargs.get("d_model", 256)
        dropout = kwargs.get("dropout", 0.1)
        self.output_length = kwargs.get("output_length", 8)

        # dilations: [1, 2, 4, 8] 같은 리스트를 config로 받되, 없으면 기본값
        dilations = kwargs.get("dilations", [1, 2, 4, 8])
        kernel_size = kwargs.get("kernel_size", 3)

        # 임베딩: padding_idx를 설정해두면 PAD 토큰은 자동으로 0벡터 초기화됨
        if pad_id is not None:
            self.embed_in = nn.Embedding(in_vocab, d_model, padding_idx=pad_id)
        else:
            self.embed_in = nn.Embedding(in_vocab, d_model)

        # Dilated CNN 스택
        self.layers = nn.ModuleList(
            [
                DilatedConvBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        # 전역 표현 후 자릿수별로 position embedding 더해주기
        self.num_digit_classes = len(OUTPUT_CHARS)
        self.pos_emb = nn.Embedding(self.output_length, d_model)
        self.classifier = nn.Linear(d_model, self.num_digit_classes)

    def forward(self, src: torch.Tensor, src_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        src: [B, T] (토큰 인덱스)
        src_pad_mask: [B, T], pad 위치가 True인 bool 텐서
        return: [B, output_length, num_digit_classes]
        """
        # [B, T] → [B, T, d_model]
        x = self.embed_in(src)

        # Dilated CNN 인코더 통과
        for layer in self.layers:
            x = layer(x)  # [B, T, d_model]

        # PAD 마스크를 고려한 masked mean pooling으로 전역 벡터 계산
        if src_pad_mask is not None:
            # non-pad 위치: True
            nonpad = ~src_pad_mask  # [B, T]
            # [B, T, 1]로 브로드캐스트 맞추기
            nonpad_f = nonpad.unsqueeze(-1).float()
            # PAD 위치는 0으로 날려버림
            x_masked = x * nonpad_f
            # 각 배치별 유효 길이
            lengths = nonpad_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
            h_global = x_masked.sum(dim=1) / lengths      # [B, d_model]
        else:
            # 간단 mean pooling
            h_global = x.mean(dim=1)  # [B, d_model]

        B = src.size(0)

        # 자리별 query: h_global + position embedding
        positions = torch.arange(self.output_length, device=src.device)
        positions = positions.unsqueeze(0).expand(B, -1)  # [B, output_length]
        pos_vec = self.pos_emb(positions)                 # [B, output_length, d_model]

        # [B, 1, d_model] + [B, output_length, d_model]
        h = h_global.unsqueeze(1) + pos_vec               # [B, output_length, d_model]

        # 각 자릿수마다 0~9 분류
        logits = self.classifier(h)                       # [B, output_length, num_digit_classes]
        return logits



# ========================
# InThon 규정용 Model
# ========================


class Model(BaseModel):
    """
    InThon Datathon 제출용 Model 클래스
    """

    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("best_model.pt", map_location=self.device)

        tokenizer_config_dict = checkpoint.get("tokenizer_config")
        if tokenizer_config_dict is None:
            raise ValueError("체크포인트에 'tokenizer_config'가 없습니다.")

        input_chars = tokenizer_config_dict.get("input_chars", INPUT_CHARS)
        output_chars = tokenizer_config_dict.get("output_chars", OUTPUT_CHARS)
        add_special = tokenizer_config_dict.get("add_special", True)
        self.max_input_length = tokenizer_config_dict.get("max_input_length", 64)
        self.max_output_length = tokenizer_config_dict.get("max_output_length", 8)

        self.input_tokenizer = CharTokenizer(
            input_chars if input_chars is not None else INPUT_CHARS,
            add_special=add_special,
        )
        self.output_tokenizer = CharTokenizer(
            output_chars if output_chars is not None else OUTPUT_CHARS,
            add_special=add_special,
        )

        model_config_dict = checkpoint.get("model_config")
        if model_config_dict is None:
            raise ValueError("체크포인트에 'model_config'가 없습니다.")

        model_config_dict = dict(model_config_dict)
        model_config_dict.setdefault("output_length", self.max_output_length)

        self.model = TinyCNNSeq2Seq(
            in_vocab=self.input_tokenizer.vocab_size,
            out_vocab=self.output_tokenizer.vocab_size,
            pad_id=self.input_tokenizer.pad_id,
            **model_config_dict,
        ).to(self.device)

        model_state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def predict(self, input_text: str) -> str:
        if not isinstance(input_text, str):
            input_text = str(input_text)

        batch = {"input_text": [input_text], "target_text": ["0"]}
        batch_tensors = tokenize_batch(
            batch,
            self.input_tokenizer,
            self.output_tokenizer,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
        )

        src = batch_tensors.src.to(self.device)
        pad_mask = src.eq(self.input_tokenizer.pad_id)

        with torch.no_grad():
            logits = self.model(src, pad_mask)
            preds = torch.argmax(logits, dim=-1)

        pred_str = digits_to_string(preds[0].tolist())
        return pred_str if pred_str.isdigit() else ""
