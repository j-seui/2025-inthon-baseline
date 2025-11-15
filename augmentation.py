import random
from dataclasses import dataclass
from typing import List, Optional


# ===== AST 정의 =====

@dataclass
class Node:
    kind: str                  # 'num' or 'op'
    value: Optional[str] = None
    op: Optional[str] = None
    children: Optional[List["Node"]] = None

    def copy(self) -> "Node":
        if self.kind == "num":
            return Node("num", value=self.value)
        return Node("op", op=self.op, children=[c.copy() for c in self.children])


# ===== 토크나이저 =====

def tokenize(expr: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
        elif c.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif c in "+-*()":
            tokens.append(c)
            i += 1
        elif c == "/":
            if i + 1 < len(expr) and expr[i + 1] == "/":
                tokens.append("//")
                i += 2
            else:
                raise ValueError("Only '//' is allowed")
        else:
            raise ValueError(f"Invalid character: {c!r}")
    return tokens


# ===== 파서 =====

class Parser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of input")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_expr()
        if self.peek() is not None:
            raise ValueError("Extra tokens after parsing expression")
        return node

    # expr -> term ((+|-) term)*
    def parse_expr(self) -> Node:
        node = self.parse_term()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self.parse_term()
            node = Node("op", op=op, children=[node, right])
        return node

    # term -> factor ((*|//) factor)*
    def parse_term(self) -> Node:
        node = self.parse_factor()
        while self.peek() in ("*", "//"):
            op = self.consume()
            right = self.parse_factor()
            node = Node("op", op=op, children=[node, right])
        return node

    # factor -> NUMBER | '(' expr ')'
    def parse_factor(self) -> Node:
        tok = self.peek()
        if tok == "(":
            self.consume("(")
            node = self.parse_expr()
            self.consume(")")
            return node
        elif tok is not None and tok[0].isdigit():
            self.consume()
            return Node("num", value=tok)
        else:
            raise ValueError(f"Unexpected token: {tok}")


def parse_expression(text: str) -> Node:
    return Parser(tokenize(text)).parse()


# ===== N-ary → binary normalize =====

def binary_normalize(node: Node) -> Node:
    """+, * 등에서 children가 2개 초과일 때 같은 op로 이진트리 재구성."""
    if node.kind == "num":
        return node

    # 먼저 자식들도 normalize
    children = [binary_normalize(c) for c in node.children]

    if len(children) <= 2:
        return Node("op", op=node.op, children=children)

    cur = children[0]
    for nxt in children[1:]:
        cur = Node("op", op=node.op, children=[cur, nxt])
    return cur


# ===== 변환들 =====

def flatten_associative(node: Node) -> None:
    """+, * 에 대해 (A+(B+C)) -> (A+B+C) 같은 flatten."""
    if node.kind != "op":
        return

    for c in node.children:
        flatten_associative(c)

    if node.op in ("+", "*"):
        new_ch = []
        for c in node.children:
            if c.kind == "op" and c.op == node.op:
                new_ch.extend(c.children)
            else:
                new_ch.append(c)
        node.children = new_ch


def apply_commutativity(node: Node, rng: random.Random) -> None:
    """+, * 에서 자식 순서를 랜덤하게 섞기 (강화 버전)."""
    if node.kind != "op":
        return
    for c in node.children:
        apply_commutativity(c, rng)
    if node.op in ("+", "*") and len(node.children) > 1:
        # 확률을 0.7에서 0.95로 증가하여 거의 항상 교환
        if rng.random() < 0.95:
            rng.shuffle(node.children)


def random_group_associative(node: Node, rng: random.Random) -> None:
    """+, * 에 대해 N항을 랜덤 이진트리로 재구성."""
    if node.kind != "op":
        return

    for c in node.children:
        random_group_associative(c, rng)

    if node.op in ("+", "*") and len(node.children) > 2 and rng.random() < 0.7:
        ch = node.children
        cur = ch[0]
        for nxt in ch[1:]:
            if rng.random() < 0.5:
                cur = Node("op", op=node.op, children=[cur, nxt])
            else:
                cur = Node("op", op=node.op, children=[nxt, cur])
        node.kind = cur.kind
        node.op = cur.op
        node.value = cur.value
        node.children = cur.children


def try_distribute(node: Node, rng: random.Random) -> Node:
    """분배 / 역분배 일부 적용 (항상 children 길이 체크)."""
    if node.kind != "op":
        return node

    # 아래부터 처리
    node.children = [try_distribute(c, rng) for c in node.children]

    # 필요하면 이항으로 normalize
    if node.op in ("*", "+", "-") and len(node.children) > 2:
        node = binary_normalize(node)

    # ----- 분배법칙: A*(B±C) -----
    if node.op == "*" and len(node.children) == 2:
        A, B = node.children
        # A*(B±C)
        if B.kind == "op" and B.op in ("+", "-") and len(B.children) == 2 and rng.random() < 0.5:
            B1, B2 = B.children
            left = Node("op", op="*", children=[A.copy(), B1.copy()])
            right = Node("op", op="*", children=[A.copy(), B2.copy()])
            return Node("op", op=B.op, children=[left, right])

        # (A±B)*C 형태도 처리하고 싶으면 여기서도 추가 가능
        if A.kind == "op" and A.op in ("+", "-") and len(A.children) == 2 and rng.random() < 0.5:
            A1, A2 = A.children
            left = Node("op", op="*", children=[A1.copy(), B.copy()])
            right = Node("op", op="*", children=[A2.copy(), B.copy()])
            return Node("op", op=A.op, children=[left, right])

    # ----- 역분배: A*B + A*C → A*(B±C) -----
    if node.op in ("+", "-") and len(node.children) == 2 and rng.random() < 0.3:
        L, R = node.children
        if L.kind == "op" and L.op == "*" and R.kind == "op" and R.op == "*":
            if len(L.children) == 2 and len(R.children) == 2:
                La, Lb = L.children
                Ra, Rb = R.children

                def same(a: Node, b: Node) -> bool:
                    return strict_to_string(a) == strict_to_string(b)

                if same(La, Ra):
                    common = La.copy()
                    left_rem = Lb.copy()
                    right_rem = Rb.copy()
                elif same(Lb, Rb):
                    common = Lb.copy()
                    left_rem = La.copy()
                    right_rem = Ra.copy()
                else:
                    return node

                inner = Node("op", op=node.op, children=[left_rem, right_rem])
                return Node("op", op="*", children=[common, inner])

    return node


# ===== strict pretty-printer =====

def strict_to_string(node: Node) -> str:
    """항상 괄호를 유지하여 AST 의미 100% 보존."""
    if node.kind == "num":
        return node.value

    # children 길이가 2가 아니면 normalize
    if len(node.children) != 2:
        node = binary_normalize(node)

    left, right = node.children
    return "(" + strict_to_string(left) + node.op + strict_to_string(right) + ")"


# ===== minimal pretty-printer =====

def minimal_to_string(node: Node, parent_prec: int = 0, is_right_child: bool = False) -> str:
    """필요한 괄호만 유지하면서 의미 보존."""
    if node.kind == "num":
        return node.value

    if len(node.children) != 2:
        node = binary_normalize(node)

    left, right = node.children
    op = node.op

    if op in ("+", "-"):
        prec = 1
    else:  # '*', '//'
        prec = 2

    left_s = minimal_to_string(left, prec, False)
    right_s = minimal_to_string(right, prec, True)
    s = left_s + op + right_s

    need_paren = False

    # 부모보다 우선순위 낮으면 괄호 필요
    if prec < parent_prec:
        need_paren = True

    # -, // 는 오른쪽 자식일 때 더 조심
    if op == "-" and is_right_child:
        need_paren = True
    if op == "//" and is_right_child:
        need_paren = True

    if need_paren:
        return "(" + s + ")"
    return s


# ===== equivalence helper =====

def equivalent(a: str, b: str) -> bool:
    try:
        return eval(a) == eval(b)
    except Exception:
        return False


# ===== generate_equivalents =====

def generate_equivalents(expr_str: str, num_variants: int = 5, seed: Optional[int] = None):
    """
    하나의 AE를 받아, 의미가 동일한 다른 AE들을 최대 num_variants개 생성.
    strict → minimal 두 단계로 보장.
    교환법칙 증강 강화.
    """
    rng = random.Random(seed)

    base_ast = parse_expression(expr_str)
    gt_val = eval(expr_str)

    variants = set()

    # 원본도 minimal 형태로 한 번 넣어주기
    base_ast_norm = binary_normalize(base_ast)
    strict_base = strict_to_string(base_ast_norm)
    minimal_base = minimal_to_string(base_ast_norm)
    if eval(minimal_base) == gt_val:
        variants.add(minimal_base)
    else:
        # 혹시 모르니 strict라도
        variants.add(strict_base)

    trials = 0
    max_trials = num_variants * 40

    while len(variants) < num_variants and trials < max_trials:
        trials += 1
        ast = base_ast.copy()

        # 변환 적용
        flatten_associative(ast)
        
        # 교환법칙을 더 강하게 적용하기 위해 여러 번 호출
        apply_commutativity(ast, rng)
        if rng.random() < 0.5:
            apply_commutativity(ast, rng)  # 50% 확률로 한 번 더
        
        ast = try_distribute(ast, rng)
        random_group_associative(ast, rng)
        
        # 교환법칙을 마지막에 한 번 더 적용
        if rng.random() < 0.7:
            apply_commutativity(ast, rng)

        # 안전하게 이항 normalize
        ast = binary_normalize(ast)

        # strict 표현
        strict_expr = strict_to_string(ast)
        try:
            if eval(strict_expr) != gt_val:
                continue
        except Exception:
            continue

        # minimal 표현
        minimal_expr = minimal_to_string(ast)
        try:
            if eval(minimal_expr) != gt_val:
                continue
        except Exception:
            continue

        variants.add(minimal_expr)

    return list(variants)