"""베이스라인 데이터 생성 및 DataLoader"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Union, List, Optional
import math
import random
from functools import partial
from torch.utils.data import DataLoader, Dataset

from augmentation import generate_equivalents
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
            wrap = child_prec < parent_prec
            if not wrap and child_prec == parent_prec:
                if parent_op in ("-", "//") and not is_left:
                    wrap = True
                elif parent_op == "*" and child_op == "//" and not is_left:
                    wrap = True
                elif parent_op == "//" and child_op in ("*", "//") and not is_left:
                    wrap = True
            if not wrap:
                wrap = rng.random() < 0.15
        return f"({expr})" if wrap else expr

    def _compose(op: str, left_meta, right_meta) -> str:
        le, _, lo = left_meta[:3]
        re, _, ro = right_meta[:3]
        le = _maybe_wrap(le, lo, op, True)
        re = _maybe_wrap(re, ro, op, False)
        expr = f"{le}{op}{re}"
        if rng.random() < 0.2:
            expr = f"({expr})"
        return expr

    def _build(d: int):
        if d == 0:
            expr, value, op = _base_number()
            return expr, value, op, []

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
        steps = left_meta[3] + right_meta[3]
        step_desc = f"{lv} {op} {rv} = {value}"
        steps.append(step_desc)
        return expression, value, op, steps

    expr_str, expr_val, _, steps = _build(depth)
    return expr_str, expr_val, steps


class ArithmeticDataset(Dataset):
    """사칙연산 수식 생성 Dataset"""
    
    def __init__(
        self,
        num_samples: int,
        max_depth: int = 2,
        num_digits: Tuple[int, int] = (1, 3),
        seed: int = 42,
        mode: str = "train",
        *,
        equivalent_variants: int = 1,
        curriculum_config: Optional[Dict[str, Any]] = None,
    ):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.num_digits = num_digits
        self.seed = seed
        self.mode = mode
        self.num_equivalents = max(1, equivalent_variants)

        cfg = curriculum_config or {}
        enabled = bool(cfg) and bool(cfg.get("enabled", True))
        self._curriculum_enabled = enabled
        self._curriculum_num_stages = max(1, int(cfg.get("num_stages", 1))) if enabled else 1
        initial_stage = int(cfg.get("initial_stage", 0))
        self._curriculum_stage = min(max(initial_stage, 0), self._curriculum_num_stages - 1)
        self._curriculum_keep_last = bool(cfg.get("keep_last_step", True))
        self._curriculum_prefix = cfg.get("prefix", " =")
        self._curriculum_wrapper = cfg.get("step_wrapper", "({step})")
    
    def __len__(self) -> int:
        return self.num_samples
    
    @property
    def curriculum_enabled(self) -> bool:
        return self._curriculum_enabled

    @property
    def curriculum_stage_count(self) -> int:
        return self._curriculum_num_stages

    def set_curriculum_stage(self, stage: int) -> None:
        if not self._curriculum_enabled:
            return
        stage = max(0, min(stage, self._curriculum_num_stages - 1))
        self._curriculum_stage = stage

    def _curriculum_steps_for_stage(self, steps: List[str]) -> List[str]:
        if not (self._curriculum_enabled and steps):
            return []
        if self._curriculum_num_stages <= 1:
            return steps
        stage_idx = max(0, min(self._curriculum_stage, self._curriculum_num_stages - 1))
        if stage_idx >= self._curriculum_num_stages - 1:
            return []
        ratio = stage_idx / (self._curriculum_num_stages - 1)
        drop = int(math.ceil(ratio * len(steps)))
        selected = steps[drop:]
        if self._curriculum_keep_last and stage_idx < self._curriculum_num_stages - 1 and not selected:
            selected = steps[-1:]
        return selected

    def _format_curriculum_suffix(self, stage_steps: List[str]) -> str:
        if not stage_steps:
            return ""
        wrapped = " ".join(self._curriculum_wrapper.format(step=step) for step in stage_steps)
        prefix = self._curriculum_prefix or " "
        if prefix and not prefix.endswith(" "):
            prefix = f"{prefix} "
        return f"{prefix}{wrapped}".rstrip()
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        depth = rng.randint(0, self.max_depth)
        expr, val, steps = _gen_expr(rng, depth, self.num_digits)
        equivalents = generate_equivalents(expr, num_variants=self.num_equivalents, seed=self.seed + idx)

        meta: Dict[str, Any] = {
            "depth": depth,
            "equivalent_text": equivalents,
        }

        if self._curriculum_enabled:
            stage_steps = self._curriculum_steps_for_stage(steps)
            suffix = self._format_curriculum_suffix(stage_steps)
            stage_input = f"{expr}{suffix}"
            meta["curriculum"] = {
                "stage_index": self._curriculum_stage,
                "num_stages": self._curriculum_num_stages,
                "suffix": suffix,
                "input_text": stage_input,
                "steps_kept": stage_steps,
                "steps_total": steps,
            }

        return {
            "input_text": expr,
            "equivalent_text": equivalents,
            "target_text": str(val),
            "meta": meta,
        }


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
