from __future__ import annotations
import importlib, config, dataloader, model

importlib.reload(config)
importlib.reload(dataloader)
importlib.reload(model)

########################


from typing import List, Any, Tuple, Dict, Iterable, Optional
from config import TrainConfig, ModelConfig, TokenizerConfig
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from tqdm import tqdm

from dataloader import (
    ArithmeticDataset,  # 사칙연산 데이터를 만들어주는 Dataset
    get_dataloader,     # Dataset을 받아서 DataLoader로 바꿔주는 함수
)
from do_not_edit.metric import compute_metrics  # EM, TES 같은 간단한 성능 지표

from model import (
    TinySeq2Seq,
    CharTokenizer,      # 문자 단위 토크나이저
    tokenize_batch,     # batch(dict)를 토크나이즈 + 패딩까지 해주는 함수
    INPUT_CHARS,        # 입력 문자 집합
    OUTPUT_CHARS,       # 출력 문자 집합
)

#############################

import gc; gc.collect()
torch.cuda.empty_cache()


# ======================================================================================
# 1. 학습 루프
# ======================================================================================
def train_loop(
    model: nn.Module,           # 학습할 모델
    dataloader: DataLoader,     # DataLoader를 직접 전달받아 사용합니다.
    input_tokenizer: CharTokenizer,      # (tokenize_batch 유틸리티를 위해 유지)
    output_tokenizer: CharTokenizer,     # 출력 문자 토크나이저
    device: torch.device,       # cpu 또는 cuda
    val_dataloader: DataLoader | None = None,  # 별도 검증 DataLoader (None이면 훈련 배치로 검증)
    *,
    train_config: TrainConfig,  # 학습 설정
    model_config: ModelConfig,  # 모델 설정
    tokenizer_config: TokenizerConfig,  # 토크나이저 설정
):
    # 모델을 GPU/CPU로 보냄
    model.to(device)
    law_lambda = getattr(train_config, "law_lambda", 0.0)
    law_num_variants = max(1, getattr(train_config, "law_num_variants", 1))
    law_max_pairs = getattr(train_config, "law_max_pairs_per_batch", 0)
    law_seed = getattr(train_config, "law_seed", 0)
    law_rng = random.Random(law_seed)

    # 옵티마이저: AdamW는 Adam + weight decay가 들어간 버전
    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr)

    # seq2seq에서 흔히 쓰는 CE loss
    # pad 토큰은 무시하도록(ignore_index) 설정
    loss_fn = nn.CrossEntropyLoss(ignore_index=output_tokenizer.pad_id)

    def _normalize_variants(expr: str, equivalents: Any) -> List[str]:
        """equivalent_text 값을 리스트로 정리 (원본과 동일한 표현은 제외)."""
        if equivalents is None:
            return []
        if isinstance(equivalents, str):
            candidates = [equivalents]
        elif isinstance(equivalents, Iterable):
            candidates = list(equivalents)
        else:
            return []
        unique: List[str] = []
        seen = set()
        for cand in candidates:
            if not isinstance(cand, str):
                continue
            if cand == expr or cand in seen:
                continue
            seen.add(cand)
            unique.append(cand)
        return unique

    class CurriculumManager:
        """Curriculum 스케줄을 관리하며 Dataset stage를 갱신."""

        def __init__(self, cfg: TrainConfig, dataset: Optional[Dataset]):
            self.dataset = dataset
            self.enabled = bool(getattr(cfg, "curriculum_enabled", False))
            self.mode = "steps" if (self.enabled and getattr(cfg, "curriculum_stage_steps", None)) else "epochs"
            self.stage_lengths: List[int] = []
            if self.enabled:
                if self.mode == "steps":
                    steps = list(getattr(cfg, "curriculum_stage_steps", ()))
                    self.stage_lengths = [max(0, int(x)) for x in steps if x is not None]
                else:
                    stage_count = max(1, int(getattr(cfg, "curriculum_num_stages", 1)))
                    custom = getattr(cfg, "curriculum_stage_epochs", None)
                    if custom:
                        schedule = [max(0, int(x)) for x in custom]
                        if len(schedule) < stage_count:
                            schedule.extend([0] * (stage_count - len(schedule)))
                        elif len(schedule) > stage_count:
                            schedule = schedule[:stage_count]
                    else:
                        base = cfg.num_epochs // stage_count
                        remainder = cfg.num_epochs % stage_count
                        schedule = [base] * stage_count
                        for i in range(remainder):
                            schedule[i] += 1
                    gap = cfg.num_epochs - sum(schedule)
                    if schedule and gap > 0:
                        schedule[-1] += gap
                    self.stage_lengths = schedule
            self.stage_count = len(self.stage_lengths)
            if not self.stage_lengths:
                self.enabled = False
            self.stage_idx = 0
            self.remaining = self.stage_lengths[0] if self.stage_lengths else 0
            self.logger: Optional[Any] = None
            if self.enabled and self.stage_lengths:
                self._set_stage(0)
                self._skip_empty("init")

        def attach_logger(self, logger_fn):
            self.logger = logger_fn
            if self.active:
                self.logger(f"[curriculum] stage -> {self.stage_idx+1}/{self.stage_count} (init)")

        @property
        def active(self) -> bool:
            return self.enabled and self.stage_count > 0

        def stage_label(self) -> str:
            if not self.active:
                return ""
            return f"s{self.stage_idx+1}/{self.stage_count}"

        def _set_stage(self, stage_idx: int):
            dataset = self.dataset
            setter = getattr(dataset, "set_curriculum_stage", None) if dataset is not None else None
            if callable(setter):
                setter(stage_idx)

        def _advance(self, reason: str) -> bool:
            self.stage_idx += 1
            if self.stage_idx >= self.stage_count:
                self.enabled = False
                return False
            self.remaining = self.stage_lengths[self.stage_idx]
            self._set_stage(self.stage_idx)
            if self.logger:
                self.logger(f"[curriculum] stage -> {self.stage_idx+1}/{self.stage_count} ({reason})")
            return True

        def _skip_empty(self, reason: str):
            while self.active and self.remaining <= 0:
                if not self._advance(reason):
                    break

        def on_step_end(self):
            if not self.active or self.mode != "steps":
                return
            self.remaining -= 1
            if self.remaining <= 0:
                if self._advance("step budget"):
                    self._skip_empty("step budget")

        def on_epoch_end(self):
            if not self.active or self.mode != "epochs":
                return
            self.remaining -= 1
            if self.remaining <= 0:
                if self._advance("epoch budget"):
                    self._skip_empty("epoch budget")

    def _prepare_curriculum_inputs(batch_dict: Dict[str, List[str]]):
        inputs = batch_dict.get("input_text") or []
        metas = batch_dict.get("meta") or []
        resolved_inputs: List[str] = []
        suffixes: List[str] = []
        for i, raw in enumerate(inputs):
            curriculum_meta = None
            if i < len(metas) and metas[i] is not None:
                curriculum_meta = metas[i].get("curriculum")
            if curriculum_meta and curriculum_meta.get("input_text"):
                resolved_inputs.append(curriculum_meta["input_text"])
                suffixes.append(curriculum_meta.get("suffix", ""))
            else:
                resolved_inputs.append(raw)
                suffixes.append("")
        return resolved_inputs, suffixes

    def _sample_law_pairs(batch_dict: Dict[str, List[str]], resolved_inputs: List[str], suffixes: List[str]):
        """배치에서 augmentation 변형을 뽑아 (idx, variant, target)을 반환."""
        if law_lambda <= 0:
            return []
        inputs = resolved_inputs
        targets = batch_dict.get("target_text") or []
        equivalents = batch_dict.get("equivalent_text")
        if equivalents is None:
            metas = batch_dict.get("meta")
            if metas:
                equivalents = [(meta or {}).get("equivalent_text") for meta in metas]
        if not inputs or not targets or equivalents is None:
            return []

        indices = list(range(len(inputs)))
        if not indices:
            return []
        max_pairs = law_max_pairs if law_max_pairs and law_max_pairs > 0 else len(indices) * law_num_variants
        law_rng.shuffle(indices)

        pairs = []
        for idx in indices:
            expr = inputs[idx]
            if idx >= len(equivalents):
                continue
            variants = _normalize_variants(expr, equivalents[idx])
            if not variants:
                continue

            if len(variants) <= law_num_variants:
                chosen = variants
            else:
                chosen = law_rng.sample(variants, law_num_variants)

            suffix = suffixes[idx] if idx < len(suffixes) else ""
            for variant in chosen:
                decorated = variant + suffix if suffix else variant
                pairs.append((idx, decorated, targets[idx]))
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs

    curriculum_manager = CurriculumManager(train_config, getattr(dataloader, "dataset", None))

    step = 0
    model.train()  # 학습 모드로 전환 (Dropout 등 켜짐)

    # tqdm은 진행 상황을 예쁘게 보여주는 라이브러리입니다.
    pbar = tqdm(total=train_config.max_train_steps if train_config.max_train_steps is not None else None, desc="train", unit="step", ncols=120, dynamic_ncols=False, leave=False)
    curriculum_manager.attach_logger(pbar.write)

    # best EM 추적용 변수 (None이 아니면 개선 시 모델 저장)
    best_em = float("-inf")

    for epoch in range(train_config.num_epochs):
        # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.
        if train_config.max_train_steps is not None and step >= train_config.max_train_steps: break
        stage_desc = curriculum_manager.stage_label()
        desc = f"train e{epoch+1}/{train_config.num_epochs}"
        if stage_desc:
            desc += f" {stage_desc}"
        pbar.set_description(desc)

        # 실제로 배치를 하나씩 뽑아서 학습하는 부분입니다.
        for batch in dataloader:
            # --------------------------------------------------------------
            # 1) 토크나이즈 & 텐서로 변환
            #    `tokenize_batch`는 BatchTensors(src, tgt_inp, tgt_out)를 반환합니다.
            #    변수 역할:
            #      - `src` (encoder input): 모델의 인코더 입력. 정수 텐서, shape (B, S).
            #      - `target_input` (tgt_inp): 디코더에 teacher-forcing으로 넣는 입력. shape (B, T).
            #          일반적으로 BOS를 앞에 붙이고 EOS는 제외한 시퀀스입니다.
            #      - `target_output` (tgt_out): 디코더가 예측해야 하는 정답(손실 대상). shape (B, T).
            #          일반적으로 target_input에서 BOS를 뺀 것에 EOS를 붙인 형태입니다.
            #    예시 (토큰 id가 다음과 같다고 가정):
            #      bos_id=1, eos_id=2, '1'->5, '6'->6
            #      원본 target_text: "16"
            #      target_input ids:  [1, 5, 6]    # [BOS, '1', '6']
            #      target_output ids: [5, 6, 2]    # ['1', '6', EOS']
            #    주의: 모든 텐서는 dtype=torch.long이고 `.to(device)`로 명시적 이동이 필요합니다.
            # --------------------------------------------------------------
            resolved_inputs, stage_suffixes = _prepare_curriculum_inputs(batch)
            batch_for_tokenizer = {
                "input_text": resolved_inputs,
                "target_text": batch["target_text"],
            }
            batch_tensors = tokenize_batch(batch_for_tokenizer, input_tokenizer, output_tokenizer)
            src = batch_tensors.src.to(device)
            target_input = batch_tensors.tgt_inp.to(device)
            target_output = batch_tensors.tgt_out.to(device)

            # Forward: 모델에 입력을 전달하고 출력을 얻습니다.
            # 출력은 (B, T, V) 형태로 반환됩니다.
            logits = model(src, target_input, input_tokenizer.pad_id)

            # --------------------------------------------------------------
            # 4) Loss 계산
            # --------------------------------------------------------------
            ce_loss = loss_fn(
                logits.view(-1, logits.size(-1)),  # (B*T, V)
                target_output.view(-1),             # (B*T,)
            )

            law_loss = torch.tensor(0.0, device=device)
            if law_lambda > 0:
                law_pairs = _sample_law_pairs(batch, resolved_inputs, stage_suffixes)
                if law_pairs:
                    aug_batch = {
                        "input_text": [variant for _, variant, _ in law_pairs],
                        "target_text": [target for _, _, target in law_pairs],
                    }
                    aug_tensors = tokenize_batch(aug_batch, input_tokenizer, output_tokenizer)
                    aug_src = aug_tensors.src.to(device)
                    aug_target_input = aug_tensors.tgt_inp.to(device)
                    target_seq_len = target_input.size(1)
                    aug_len = aug_target_input.size(1)
                    if aug_len != target_seq_len:
                        pad_token_id = output_tokenizer.pad_id if output_tokenizer.pad_id is not None else 0
                        if aug_len < target_seq_len:
                            pad_shape = (aug_target_input.size(0), target_seq_len - aug_len)
                            pad_tensor = torch.full(
                                pad_shape,
                                pad_token_id,
                                dtype=aug_target_input.dtype,
                                device=device,
                            )
                            aug_target_input = torch.cat([aug_target_input, pad_tensor], dim=1)
                        else:
                            aug_target_input = aug_target_input[:, :target_seq_len]
                    aug_logits = model(aug_src, aug_target_input, input_tokenizer.pad_id)

                    ref_indices = torch.tensor([idx for idx, _, _ in law_pairs], dtype=torch.long, device=device)
                    ref_logits = logits.index_select(0, ref_indices)
                    law_loss = F.mse_loss(ref_logits, aug_logits)

            total_loss = ce_loss + law_lambda * law_loss

            # --------------------------------------------------------------
            # 5) Backward + optimizer step
            # --------------------------------------------------------------
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient 폭주 방지를 위해 클리핑
            optim.step()
            optim.zero_grad()

            step += 1
            curriculum_manager.on_step_end()

            # --------------------------------------------------------------
            # 6) Validation
            # --------------------------------------------------------------
            if step % train_config.valid_every == 0:
                model.eval()  # 평가 모드
                # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.
                with torch.no_grad():
                    preds_all: List[str] = []
                    targets_all: List[str] = []
                    inputs_all: List[str] = [] # 검증 데이터셋의 입력, 정답, 예측 결과를 저장할 리스트

                    for val_batch in val_dataloader: # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.
                        val_bt = tokenize_batch(val_batch, input_tokenizer, output_tokenizer)
                        val_src = val_bt.src.to(device)
                        gen_ids = model.generate(
                            src=val_src,
                            max_len=train_config.max_gen_len,
                            bos_id=output_tokenizer.bos_id,
                            eos_id=output_tokenizer.eos_id,
                            src_pad_id=input_tokenizer.pad_id,
                        )

                        for i in range(gen_ids.size(0)):
                            seq_chars: List[str] = []
                            for t in gen_ids[i].tolist():
                                idx = int(t)
                                if idx == output_tokenizer.eos_id:
                                    break
                                if idx in output_tokenizer.itos:
                                    ch = output_tokenizer.itos[idx]
                                    if ch.isdigit() or (ch == '-' and not seq_chars):
                                        seq_chars.append(ch)
                            pred_str = "".join(seq_chars)
                            if pred_str == "-":
                                pred_str = ""
                            preds_all.append(pred_str)
                        # 검증 데이터셋의 정답, 입력을 리스트에 추가합니다.
                        targets_all.extend(val_batch["target_text"])
                        inputs_all.extend(val_batch["input_text"])

                    # 검증 데이터셋의 예측, 정답을 사용하여 성능 지표를 계산합니다.
                    em_batch = compute_metrics(preds_all, targets_all)

                    # 진행바에도 성능을 표시합니다.
                    #pbar.write(f"[valid {step}] EM={em_batch['EM']:.3f} TES={em_batch['TES']:.3f}")
                    pbar.set_postfix(
                        EM=f"{em_batch['EM']:.3f}",
                        TES=f"{em_batch['TES']:.3f}",
                    )

                    pbar.refresh()

                    # 최고 성능 갱신 시 전체 체크포인트 저장
                    if train_config.save_best_path is not None:
                        current_em = float(em_batch.get("EM", -1.0))
                        if current_em > best_em:
                            best_em = current_em
                            # 세 config를 dict로 변환하여 저장
                            ckpt = {
                                "model_state": model.state_dict(),
                                "optim_state": optim.state_dict(),
                                "step": step,
                                "train_config": train_config.__dict__,  # 학습 설정 저장
                                "model_config": model_config.__dict__,  # 모델 설정 저장
                                "tokenizer_config": tokenizer_config.__dict__,  # 토크나이저 설정 저장
                            }
                            torch.save(ckpt, train_config.save_best_path)
                            pbar.write(f"New best EM={best_em:.3f} at step {step}; saved to {train_config.save_best_path}")

                    B = len(preds_all) # 검증 데이터셋의 크기
                    n_show = min(train_config.show_valid_samples, B)
                    pbar.write("Sample validation output:") # 예시로 몇 개만 보여줍니다.
                    for i in range(n_show):
                        input_str = inputs_all[i]
                        tgt = targets_all[i]
                        pred = preds_all[i]
                        ok = "OK" if pred == tgt else "ERR"
                        pbar.write(f"  [{i}] {ok} | input: {input_str} | target: {tgt} | pred: {pred}")
                model.train()  # 다시 학습 모드로

            # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.
            if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
                break # 학습을 종료합니다.

            pbar.update(1) # tqdm 진행 1 step

        # epoch 종료 시 stage 스케줄 갱신 (epoch 기반 curriculum일 경우)
        curriculum_manager.on_epoch_end()

        if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
            break # 학습을 종료합니다.

# ======================================================================================
# 2. main 함수
# ======================================================================================
def main():
    # GPU가 있으면 GPU, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 0) 학습 설정을 가장 먼저 정의 (curriculum 세팅 사용)
    # --------------------------------------------------------------------------
    train_config = TrainConfig(
        max_train_steps=None,
        lr=3e-4,           # config.py와 맞추기
        valid_every=200,
        max_gen_len=24,
        show_valid_samples=5,
        num_epochs=10,     # 처음엔 10 epoch 정도로
        save_best_path="best_model.pt",
        law_lambda=0.4,
        law_num_variants=2,
        law_max_pairs_per_batch=32,
        law_seed=42,
        curriculum_enabled=True,
        curriculum_num_stages=7,
        curriculum_stage_epochs=(1, 1, 1, 1, 1, 1, 4),
    )
    curriculum_cfg = None
    if train_config.curriculum_enabled:
        if train_config.curriculum_stage_steps:
            stage_count = len(train_config.curriculum_stage_steps)
        else:
            stage_count = train_config.curriculum_num_stages
        stage_count = max(1, stage_count)
        curriculum_cfg = {
            "enabled": True,
            "num_stages": stage_count,
            "keep_last_step": True,
            "prefix": " =",
        }

    # --------------------------------------------------------------------------
    # 1) 데이터 준비
    # --------------------------------------------------------------------------

    # Train Dataset, 자세한 설정은 dataloader.py를 참고하세요.
    train_dataset = ArithmeticDataset(
        num_samples=500_000,
        max_depth=3,
        num_digits=(1, 5),
        seed=123,
        mode="train",
        curriculum_config=curriculum_cfg,
    )

    # Train DataLoader, 자세한 설정은 dataloader.py를 참고하세요.
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )

    # Validation Dataset, 자세한 설정은 dataloader.py를 참고하세요.
    val_dataset = ArithmeticDataset(
        num_samples=128,
        max_depth=4,
        num_digits=(1, 5),
        seed=999,
        mode="val",
    )

    # Validation DataLoader, 자세한 설정은 dataloader.py를 참고하세요.
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 2) 토크나이저 설정 준비
    # --------------------------------------------------------------------------
    # 토크나이저 설정 (별도로 관리)
    tokenizer_config = TokenizerConfig(
        input_chars=INPUT_CHARS,
        output_chars=OUTPUT_CHARS,
        add_special=True,
    )

    # --------------------------------------------------------------------------
    # 3) 토크나이저 준비
    # --------------------------------------------------------------------------
    # 입력 문자 토크나이저, 출력 문자 토크나이저를 준비합니다. 자세한 설정은 model.py를 참고하세요.
    input_tokenizer = CharTokenizer(
        tokenizer_config.input_chars if tokenizer_config.input_chars is not None else INPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )
    output_tokenizer = CharTokenizer(
        tokenizer_config.output_chars if tokenizer_config.output_chars is not None else OUTPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )

    # --------------------------------------------------------------------------
    # 4) 모델 설정 준비
    # --------------------------------------------------------------------------
    # 모델 아키텍처 설정 (별도로 관리)
    model_config = ModelConfig(
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    )

    # --------------------------------------------------------------------------
    # 6) 모델 준비
    # --------------------------------------------------------------------------
    # GRU 기반의 TinySeq2Seq 모델을 준비합니다. 자세한 설정은 model.py를 참고하세요.
    # ModelConfig의 모든 필드를 **kwargs로 전달
    model = TinySeq2Seq(
        in_vocab=input_tokenizer.vocab_size,
        out_vocab=output_tokenizer.vocab_size,
        **model_config.__dict__,  # 모델 설정을 **kwargs로 전달
    )

    # --------------------------------------------------------------------------
    # 6) 학습 시작
    # --------------------------------------------------------------------------

    train_loop(
        model=model,
        dataloader=train_dataloader,              # 미리 만든 DataLoader를 직접 전달합니다.
        input_tokenizer=input_tokenizer,        # (tokenize_batch 유틸리티를 위해 전달)
        output_tokenizer=output_tokenizer,
        device=device,
        val_dataloader=val_dataloader,
        train_config=train_config,  # 학습 설정 전달
        model_config=model_config,  # 모델 설정 전달
        tokenizer_config=tokenizer_config,  # 토크나이저 설정 전달
    )

    # --------------------------------------------------------------------------
    # 6) 학습 끝나면 모델 저장
    # --------------------------------------------------------------------------
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")


# python train.py로 실행했을 때만 main()을 돌게 합니다.
if __name__ == "__main__":
    main()
