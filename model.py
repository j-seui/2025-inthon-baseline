from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass

import math
import torch
import torch.nn as nn

from do_not_edit.model_template import BaseModel

# ========================
# Tokenizer (통합 버전)
# ========================

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

INPUT_CHARS = list("0123456789+-*/() =")
OUTPUT_CHARS = list("0123456789")


class CharTokenizer:
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
            idx = self.stoi.get(ch, None)
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
    src: torch.Tensor      # [B, S]
    tgt_inp: torch.Tensor  # [B, T]
    tgt_out: torch.Tensor  # [B, T]


def _pad(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    L = max(len(s) for s in seqs) if len(seqs) > 0 else 1
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) > 0:
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def tokenize_batch(
    batch: Dict[str, List[str]],
    input_tokenizer: CharTokenizer,
    output_tokenizer: CharTokenizer,
) -> BatchTensors:
    # 입력 인코딩 (BOS/EOS 없이)
    src_ids = [input_tokenizer.encode(s, add_bos_eos=False) for s in batch["input_text"]]

    empty_indices = [i for i, seq in enumerate(src_ids) if len(seq) == 0]
    if empty_indices:
        inputs = [batch["input_text"][i] for i in empty_indices]
        raise ValueError(f"Empty tokenized source at indices {empty_indices}; inputs: {inputs}")

    # 타겟 입력: BOS + target_text
    # 타겟 출력: target_text + EOS
    tgt_inp_ids = []
    tgt_out_ids = []
    for t in batch["target_text"]:
        base_ids = output_tokenizer.encode(t, add_bos_eos=False)
        inp_ids = [output_tokenizer.bos_id] + base_ids
        out_ids = base_ids + [output_tokenizer.eos_id]
        tgt_inp_ids.append(inp_ids)
        tgt_out_ids.append(out_ids)

    src = _pad(src_ids, input_tokenizer.pad_id)
    tgt_inp = _pad(tgt_inp_ids, output_tokenizer.pad_id)
    tgt_out = _pad(tgt_out_ids, output_tokenizer.pad_id)

    return BatchTensors(src=src, tgt_inp=tgt_inp, tgt_out=tgt_out)


# ========================
# Positional Encoding (absolute, for embeddings)
# ========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        # )
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        # self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L = x.size(1)
        # return x + self.pe[:, :L, :]
        return x


# ========================
# RoPE 구현부
# ========================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_position = max_position

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
        freqs = t * inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int, device: torch.device):
        cos = self.cos_cached[:seq_len, :].to(device)
        sin = self.sin_cached[:seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(2)   # [1, L, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


class RoPEMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, max_position: int = 2048, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary = RotaryEmbedding(self.head_dim, max_position=max_position)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                      # [B, L, D]
        attn_mask: torch.Tensor | None = None,   # [L, L]
        key_padding_mask: torch.Tensor | None = None,  # [B, L]
    ) -> torch.Tensor:
        B, L, D = x.size()
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, L, self.nhead, self.head_dim)
        k = k.view(B, L, self.nhead, self.head_dim)
        v = v.view(B, L, self.nhead, self.head_dim)

        cos, sin = self.rotary(L, device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        q = q.permute(0, 2, 1, 3)  # [B, H, L, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            else:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, L, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out


class EncoderLayerRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, max_position: int = 2048):
        super().__init__()
        self.self_attn = RoPEMultiheadSelfAttention(
            d_model=d_model,
            nhead=nhead,
            max_position=max_position,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        src2 = self.self_attn(src, attn_mask=None, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)
        return src


class DecoderLayerRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, max_position: int = 2048):
        super().__init__()
        self.self_attn = RoPEMultiheadSelfAttention(
            d_model=d_model,
            nhead=nhead,
            max_position=max_position,
            dropout=dropout,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
        self,
        tgt: torch.Tensor,                     # [B, T, D]
        memory: torch.Tensor,                  # [B, S, D]
        tgt_mask: torch.Tensor | None = None,  # [T, T]
        tgt_key_padding_mask: torch.Tensor | None = None,    # [B, T]
        memory_key_padding_mask: torch.Tensor | None = None, # [B, S]
    ) -> torch.Tensor:
        tgt2 = self.self_attn(
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt


class TinySeq2Seq(nn.Module):
    """
    RoPE 기반 Transformer encoder-decoder (이름은 TinySeq2Seq로 유지)
    """
    def __init__(self, in_vocab: int, out_vocab: int, **kwargs):
        super().__init__()
        d_model = kwargs.get("d_model", 256)
        nhead = kwargs.get("nhead", 8)
        num_encoder_layers = kwargs.get("num_encoder_layers", 3)
        num_decoder_layers = kwargs.get("num_decoder_layers", 3)
        dim_feedforward = kwargs.get("dim_feedforward", 512)
        dropout = kwargs.get("dropout", 0.1)
        max_position = kwargs.get("max_position", 2048)

        self.d_model = d_model

        self.embed_in = nn.Embedding(in_vocab, d_model, padding_idx=0)
        self.embed_out = nn.Embedding(out_vocab, d_model, padding_idx=0)

        self.pos_enc_in = PositionalEncoding(d_model)
        self.pos_enc_out = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerRoPE(d_model, nhead, dim_feedforward, dropout, max_position=max_position)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayerRoPE(d_model, nhead, dim_feedforward, dropout, max_position=max_position)
            for _ in range(num_decoder_layers)
        ])

        self.out_proj = nn.Linear(d_model, out_vocab)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        tgt_inp: torch.Tensor,
        src_pad_id: int,
        tgt_pad_id: int | None = None,
        teacher_forcing: float = 1.0,
    ) -> torch.Tensor:
        device = src.device

        # encoder
        src_emb = self.embed_in(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc_in(src_emb)
        src_key_padding_mask = (src == src_pad_id)

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=src_key_padding_mask)

        # decoder
        tgt_emb = self.embed_out(tgt_inp) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc_out(tgt_emb)

        T = tgt_inp.size(1)
        tgt_mask_bool = self._generate_square_subsequent_mask(T, device)

        if tgt_pad_id is not None:
            tgt_key_padding_mask = (tgt_inp == tgt_pad_id)
        else:
            tgt_key_padding_mask = None

        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(
                tgt=out,
                memory=memory,
                tgt_mask=tgt_mask_bool,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

        logits = self.out_proj(out)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_id: int,
        eos_id: int,
        src_pad_id: int,
    ) -> torch.Tensor:
        device = src.device
        B = src.size(0)

        src_emb = self.embed_in(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc_in(src_emb)
        src_key_padding_mask = (src == src_pad_id)

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=src_key_padding_mask)

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = []

        for _ in range(max_len):
            tgt_emb = self.embed_out(generated) * math.sqrt(self.d_model)
            tgt_emb = self.pos_enc_out(tgt_emb)

            t = generated.size(1)
            tgt_mask_bool = self._generate_square_subsequent_mask(t, device)

            out = tgt_emb
            for layer in self.decoder_layers:
                out = layer(
                    tgt=out,
                    memory=memory,
                    tgt_mask=tgt_mask_bool,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=src_key_padding_mask,
                )

            logits = self.out_proj(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1)

            outputs.append(next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished |= (next_token == eos_id)
            if torch.all(finished):
                break

        if outputs:
            return torch.stack(outputs, dim=1)
        return torch.empty((B, 0), dtype=torch.long, device=device)


# ========================
# InThon 제출용 래퍼
# ========================

class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CKPT_PATH = "best_model.pt"

        checkpoint = torch.load(CKPT_PATH, map_location=self.device)

        tokenizer_config_dict = checkpoint.get("tokenizer_config")
        if tokenizer_config_dict is None:
            raise ValueError("체크포인트에 'tokenizer_config'가 없습니다.")

        input_chars = tokenizer_config_dict.get("input_chars", INPUT_CHARS)
        output_chars = tokenizer_config_dict.get("output_chars", OUTPUT_CHARS)
        add_special = tokenizer_config_dict.get("add_special", True)

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

        self.model = TinySeq2Seq(
            in_vocab=self.input_tokenizer.vocab_size,
            out_vocab=self.output_tokenizer.vocab_size,
            **model_config_dict,
        ).to(self.device)

        model_state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(model_state)

        train_config_dict = checkpoint.get("train_config") or {}
        self.max_len = int(train_config_dict.get("max_gen_len", 50))
        if self.max_len <= 0:
            self.max_len = 50
        self.model.eval()

    def predict(self, input_text: str) -> str:
        if not isinstance(input_text, str):
            input_text = str(input_text)

        batch = {"input_text": [input_text], "target_text": ["0"]}

        batch_tensors = tokenize_batch(batch, self.input_tokenizer, self.output_tokenizer)
        src = batch_tensors.src.to(self.device)

        with torch.no_grad():
            gens = self.model.generate(
                src=src,
                max_len=self.max_len,
                bos_id=self.output_tokenizer.bos_id,
                eos_id=self.output_tokenizer.eos_id,
                src_pad_id=self.input_tokenizer.pad_id,
            )

        preds: List[str] = []
        for i in range(gens.size(0)):
            seq_chars: List[str] = []
            for t in gens[i].tolist():
                idx = int(t)
                if idx == self.output_tokenizer.eos_id:
                    break
                if idx in self.output_tokenizer.itos:
                    ch = self.output_tokenizer.itos[idx]
                    if ch.isdigit():
                        seq_chars.append(ch)
            preds.append("".join(seq_chars))

        pred = preds[0] if preds else ""
        return pred if pred.isdigit() else ""
