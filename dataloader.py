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


def _gen_expr(rng: random.Random, depth: int, num_digits: Tuple[int, int]):
    """exact depth의 arithmetic expression 생성"""
    precedence = {"+": 1, "-": 1, "*": 2, "//": 2}
    operators = ("+", "-", "*", "//")

    def _base_number():
        value = _rand_int(rng, num_digits)
        return str(value), value, None

    def _maybe_wrap(expr: str, child_op: Union[str, None], parent_op: str, is_left: bool) -> str:
        if child_op is None:
            wrap = rng.random() < 0.05
        else:
            child_prec = precedence[child_op]
            parent_prec = precedence[parent_op]
            if is_left:
                wrap = child_prec < parent_prec
            else:
                wrap = child_prec <= parent_prec
            if not wrap:
                wrap = rng.random() < 0.15
        return f"({expr})" if wrap else expr

    def _compose(op: str, left_meta, right_meta) -> str:
        le, _, lo = left_meta
        re, _, ro = right_meta
        le = _maybe_wrap(le, lo, op, True)
        re = _maybe_wrap(re, ro, op, False)
        expr = f"{le}{op}{re}"
        if rng.random() < 0.2:
            expr = f"({expr})"
        return expr

    def _build(d: int):
        if d == 0:
            return _base_number()

        left_depth = rng.randrange(d)
        right_depth = d - 1 - left_depth
        left_meta = _build(left_depth)
        right_meta = _build(right_depth)

        op = rng.choice(operators)

        if op == "-":
            if left_meta[1] < right_meta[1]:
                left_meta, right_meta = right_meta, left_meta
        elif op == "//":
            while right_meta[1] == 0:
                right_meta = _build(right_depth)

        lv = left_meta[1]
        rv = right_meta[1]

        if op == "+":
            value = lv + rv
        elif op == "-":
            value = lv - rv
        elif op == "*":
            value = lv * rv
        else:
            value = lv // rv

        expression = _compose(op, left_meta, right_meta)
        return expression, value, op

    expr_str, expr_val, _ = _build(depth)
    return expr_str, expr_val


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
