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
INPUT_CHARS = list("0123456789+-*/() =")
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
# Transformer Encoder
# ========================


class RelativePositionBias(nn.Module):
    """거리 기반 상대 위치 bias."""

    def __init__(self, num_heads: int, max_distance: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, q_len: int, k_len: int) -> torch.Tensor:
        device = self.bias.weight.device
        context = torch.arange(q_len, device=device).unsqueeze(1)
        memory = torch.arange(k_len, device=device).unsqueeze(0)
        relative = memory - context
        relative = relative.clamp(-self.max_distance, self.max_distance) + self.max_distance
        values = self.bias(relative)
        return values.permute(2, 0, 1)  # [num_heads, q_len, k_len]


class RelativeSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, max_rel_distance: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = RelativePositionBias(num_heads, max_rel_distance)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias = self.rel_bias(L, L)
        attn_scores = attn_scores + bias.unsqueeze(0)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_rel_distance: int,
    ):
        super().__init__()
        self.self_attn = RelativeSelfAttention(d_model, num_heads, dropout, max_rel_distance)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout_ff = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.self_attn(src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src


class TinySeq2Seq(nn.Module):
    """
    Transformer Encoder 기반 모델.
    CLS 토큰 표현 하나로 고정 길이 자리수 분류를 수행합니다.
    """

    def __init__(self, in_vocab: int, out_vocab: int, **kwargs):
        super().__init__()

        d_model = kwargs.get("d_model", 256)
        num_heads = kwargs.get("num_heads", 8)
        num_layers = kwargs.get("num_layers", 4)
        dim_feedforward = kwargs.get("dim_feedforward", 512)
        dropout = kwargs.get("dropout", 0.1)
        max_rel_distance = kwargs.get("max_rel_distance", 64)
        self.output_length = kwargs.get("output_length", 8)

        self.embed_in = nn.Embedding(in_vocab, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_rel_distance=max_rel_distance,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        self.num_digit_classes = len(OUTPUT_CHARS)
        self.classifier = nn.Linear(d_model, self.output_length * self.num_digit_classes)

    def forward(self, src: torch.Tensor, src_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed_in(src)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if src_pad_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=src.device)
            full_mask = torch.cat([cls_mask, src_pad_mask], dim=1)
        else:
            full_mask = None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=full_mask)

        x = self.norm(x)
        cls_repr = x[:, 0, :]
        logits = self.classifier(cls_repr)
        logits = logits.view(B, self.output_length, self.num_digit_classes)
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

        self.model = TinySeq2Seq(
            in_vocab=self.input_tokenizer.vocab_size,
            out_vocab=self.output_tokenizer.vocab_size,
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
