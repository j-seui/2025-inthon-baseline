"""베이스라인 데이터 생성 및 DataLoader"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Union
import random
from functools import partial
from torch.utils.data import DataLoader, Dataset

from do_not_edit.dataloader_validator import collate_fn_with_validation


def _rand_int(rng: random.Random, num_digits: Tuple[int, int]) -> int:
    """랜덤 정수 생성 (선행 0 없음)"""
    lo, hi = num_digits
    n = rng.randint(lo, hi)
    if n == 1:
        return rng.randint(0, 9)
    first = rng.randint(1, 9)
    rest = [rng.randint(0, 9) for _ in range(n - 1)]
    return int(str(first) + "".join(str(x) for x in rest))


def _needs_wrap(expr: str) -> bool:
    """Check if the expression needs parentheses to preserve operation order."""
    if expr.isdigit():
        return False
    return not (expr.startswith("(") and expr.endswith(")"))


def _maybe_wrap(expr: str) -> str:
    """Wrap expression in parentheses if needed."""
    return f"({expr})" if _needs_wrap(expr) else expr


def _gen_expr(rng: random.Random, depth: int, num_digits: Tuple[int, int]):
    """exact depth의 arithmetic expression 생성"""
    
    def number():
        v = _rand_int(rng, num_digits)
        return str(v), v

    def random_split(total):
        if total <= 0:
            return 0, 0
        left = rng.randint(0, total - 1)
        right = total - 1 - left
        return left, right

    def expr(d):
        if d == 0:
            return number()

        op = rng.choice(["+", "-", "*", "//"])
        left, right = random_split(d)
        le, lv = expr(left)
        re, rv = expr(right)

        # 음수 방지
        if op == "-" and lv < rv:
            lv, rv = rv, lv
            le, re = re, le
        
        if op == "*":
            v = lv * rv
        elif op == "//":
            # 0으로 나누기 방지
            if rv == 0:
                rv = 2
                re = "2"
            v = lv // rv
        elif op == "+":
            v = lv + rv
        else:  # op == "-"
            if lv < rv:
                lv, rv = rv, lv
                le, re = re, le
            v = lv - rv

        le_fmt = _maybe_wrap(le)
        re_fmt = _maybe_wrap(re)
        expr_core = f"{le_fmt}{op}{re_fmt}"
        if rng.random() < 0.5:
            return f"({expr_core})", v
        else:
            return expr_core, v

    return expr(depth)


class ArithmeticDataset(Dataset):
    """사칙연산 수식 생성 Dataset"""
    
    def __init__(
        self,
        num_samples: int,
        max_depth: int = 2,
        num_digits: Tuple[int, int] = (1, 3),
        seed: int = 42,
        mode: str = "train",
    ):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.num_digits = num_digits
        self.seed = seed
        self.mode = mode
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        depth = rng.randint(0, self.max_depth)
        expr, val = _gen_expr(rng, depth, self.num_digits)
        return {"input_text": expr, "target_text": str(val), "meta": {"depth": depth}}


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    mode: str = "train",
) -> DataLoader:
    """DataLoader 생성 (mode="train"일 때 자동 검증)"""
    is_training = (getattr(dataset, "mode", mode) == "train")
    collate_fn = partial(collate_fn_with_validation, is_training=is_training) if is_training else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
