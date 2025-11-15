from __future__ import annotations
from typing import List, Any, Tuple
from config import TrainConfig, ModelConfig, TokenizerConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

import numpy as np

from dataloader import (
    ArithmeticDataset,  # 사칙연산 데이터를 만들어주는 Dataset
    get_dataloader,     # Dataset을 받아서 DataLoader로 바꿔주는 함수
)
from do_not_edit.metric import compute_metrics  # EM, TES 같은 간단한 성능 지표

from model import (
    TinyCNNSeq2Seq,
    CharTokenizer,      # 문자 단위 토크나이저
    tokenize_batch,     # batch(dict)를 토크나이즈 + 패딩까지 해주는 함수
    digits_to_string,
    INPUT_CHARS,        # 입력 문자 집합
    OUTPUT_CHARS,       # 출력 문자 집합
)


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

    # 옵티마이저: AdamW는 Adam + weight decay가 들어간 버전
    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=train_config.max_train_steps if train_config.max_train_steps is not None else 1000
    )

    # 각 자리수에 대한 CrossEntropy loss
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    model.train()  # 학습 모드로 전환 (Dropout 등 켜짐)

    # tqdm은 진행 상황을 예쁘게 보여주는 라이브러리입니다.
    pbar = tqdm(total=train_config.max_train_steps if train_config.max_train_steps is not None else None, desc="train", unit="step", ncols=120, dynamic_ncols=True, leave=True)

    # best EM 추적용 변수 (None이 아니면 개선 시 모델 저장)
    best_em = float("-inf")

    latest_train_loss: float | None = None

    def run_validation(epoch_idx: int):
        """Run validation loop, returning avg loss and metrics."""
        nonlocal best_em
        if val_dataloader is None:
            return None, {}
        model.eval()
        with torch.no_grad():
            preds_all: List[str] = []
            targets_all: List[str] = []
            inputs_all: List[str] = []
            val_loss_total = 0.0
            val_batches = 0

            for val_batch in val_dataloader:
                val_bt = tokenize_batch(
                    val_batch,
                    input_tokenizer,
                    output_tokenizer,
                    max_input_length=tokenizer_config.max_input_length,
                    max_output_length=tokenizer_config.max_output_length,
                )
                val_src = val_bt.src.to(device)
                val_mask = val_src.eq(input_tokenizer.pad_id)
                val_tgt = val_bt.tgt.to(device)
                logits = model(val_src, val_mask)
                val_loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    val_tgt.view(-1),
                )
                val_loss_total += val_loss.item()
                val_batches += 1

                pred_digits = torch.argmax(logits, dim=-1).cpu().tolist()
                for seq in pred_digits:
                    preds_all.append(digits_to_string(seq))
                targets_all.extend(val_batch["target_text"])
                inputs_all.extend(val_batch["input_text"])

        avg_val_loss = val_loss_total / max(val_batches, 1)
        em_batch = compute_metrics(preds_all, targets_all)
        pbar.write(
            f"[epoch {epoch_idx}] val_loss={avg_val_loss:.4f} "
            f"EM={em_batch['EM']:.3f} TES={em_batch['TES']:.3f}"
        )
        pbar.set_postfix(
            train_loss=f"{latest_train_loss:.4f}" if latest_train_loss is not None else "-",
            val_loss=f"{avg_val_loss:.4f}",
            EM=f"{em_batch['EM']:.3f}",
            TES=f"{em_batch['TES']:.3f}",
        )
        pbar.refresh()

        if train_config.save_best_path is not None:
            current_em = float(em_batch.get("EM", -1.0))
            if current_em > best_em:
                best_em = current_em
                ckpt = {
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "step": step,
                    "train_config": train_config.__dict__,
                    "model_config": model_config.__dict__,
                    "tokenizer_config": tokenizer_config.__dict__,
                }
                torch.save(ckpt, train_config.save_best_path)
                pbar.write(f"New best EM={best_em:.3f} at step {step}; saved to {train_config.save_best_path}")

        B = len(preds_all)
        n_show = min(train_config.show_valid_samples, B)
        pbar.write("Sample validation output:")
        for i in range(n_show):
            input_str = inputs_all[i]
            tgt = targets_all[i]
            pred = preds_all[i]
            ok = "OK" if pred == tgt else "ERR"
            pbar.write(f"  [{i}] {ok} | input: {input_str} | target: {tgt} | pred: {pred}")
        model.train()
        return avg_val_loss, em_batch

    max_steps_reached = False
    for epoch in range(train_config.num_epochs):
        # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.
        if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
            break
        pbar.write(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")

        # 실제로 배치를 하나씩 뽑아서 학습하는 부분입니다.
        for batch in dataloader:
            # --------------------------------------------------------------
            # 1) 토크나이즈 & 텐서로 변환
            #    입력은 후위표기 토큰 ID, 출력은 고정 길이 숫자 시퀀스입니다.
            # --------------------------------------------------------------
            batch_tensors = tokenize_batch(
                batch,
                input_tokenizer,
                output_tokenizer,
                max_input_length=tokenizer_config.max_input_length,
                max_output_length=tokenizer_config.max_output_length,
            )
            src = batch_tensors.src.to(device)
            target_digits = batch_tensors.tgt.to(device)
            src_pad_mask = src.eq(input_tokenizer.pad_id)

            # Forward: Transformer Encoder 결과로 자리별 logits 생성
            logits = model(src, src_pad_mask)  # [B, L, 10]
            B, L, V = logits.shape
            targets = target_digits  # [B, L]

            # (1) mask 계산: leading zero는 무시
            with torch.no_grad():
                # CPU numpy로 편하게 처리해도 되고, torch로만 해도 됨
                t = targets.cpu().numpy()
                mask = np.zeros_like(t, dtype=bool)
                for b in range(B):
                    seq = t[b]
                    nonzero = np.where(seq != 0)[0]
                    if len(nonzero) == 0:
                        # 값이 0이면 마지막 자리만 학습
                        mask[b, -1] = True
                    else:
                        first = nonzero[0]
                        mask[b, first:] = True

                mask = torch.from_numpy(mask).to(targets.device)  # [B, L]

            logits_flat  = logits.view(-1, V)
            targets_flat = targets.view(-1)
            mask_flat    = mask.view(-1)

            loss = loss_fn(
                logits_flat[mask_flat],
                targets_flat[mask_flat],
            )


            # --------------------------------------------------------------
            # 4) Loss 계산
            # --------------------------------------------------------------
            # loss = loss_fn(
            #     logits.view(-1, logits.size(-1)),  # (B*T, V)
            #     target_digits.view(-1),             # (B*T,)
            # )
            train_loss = loss.item()
            latest_train_loss = train_loss

            # --------------------------------------------------------------
            # 5) Backward + optimizer step
            # --------------------------------------------------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient 폭주 방지를 위해 클리핑
            optim.step()
            optim.zero_grad()

            step += 1 
            pbar.set_postfix(train_loss=f"{train_loss:.4f}")
            pbar.update(1)

            if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
                max_steps_reached = True
                break # 학습을 종료합니다.

        if val_dataloader is not None:
            run_validation(epoch + 1)

        scheduler.step()

        if max_steps_reached:
            break

# ======================================================================================
# 2. main 함수
# ======================================================================================
def main():
    # GPU가 있으면 GPU, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 1) 데이터 준비
    # --------------------------------------------------------------------------
    
    # Train Dataset, 자세한 설정은 dataloader.py를 참고하세요.
    train_dataset = ArithmeticDataset(
        num_samples=500_000,
        max_depth=4,
        num_digits=(1, 3),
        seed=123,
        mode="train",
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
        max_input_length=64,
        max_output_length=20,
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
    # 5) 학습 설정 준비
    # --------------------------------------------------------------------------
    # 학습 하이퍼파라미터 설정
    train_config = TrainConfig(
        max_train_steps=None,
        lr=1e-3,
        valid_every=200,
        max_gen_len=24,
        show_valid_samples=5,
        num_epochs=50,
        save_best_path="best_model.pt",
    )

    # --------------------------------------------------------------------------
    # 6) 모델 준비
    # --------------------------------------------------------------------------
    # CNN 기반의 TinyCNNSeq2Seq 모델을 준비합니다.
    model_config = ModelConfig(
        d_model=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_rel_distance=64,
        output_length=tokenizer_config.max_output_length,
    )

    # ModelConfig의 모든 필드를 **kwargs로 전달
    model = TinyCNNSeq2Seq(
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
