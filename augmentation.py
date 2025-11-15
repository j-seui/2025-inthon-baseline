import random
from dataclasses import dataclass
from typing import List, Optional

# ===== AST 정의 =====

@dataclass
class Node:
    kind: str                  # 'num' or 'op'
    value: Optional[str] = None  # 숫자 리터럴(문자열)
    op: Optional[str] = None     # '+', '-', '*', '//' 중 하나
    children: Optional[List["Node"]] = None  # 자식 노드들

    def copy(self) -> "Node":
        if self.kind == "num":
            return Node("num", value=self.value)
        else:
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
            # '//'만 허용
            if i + 1 < len(expr) and expr[i + 1] == "/":
                tokens.append("//")
                i += 2
            else:
                raise ValueError("Single '/' is not allowed, use '//'")
        else:
            raise ValueError(f"Invalid character: {c}")
    return tokens

# ===== 파서 (precedence-aware) =====

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
    tokens = tokenize(text)
    return Parser(tokens).parse()

# ===== 변환들 =====

def flatten_associative(node: Node) -> None:
    """+, * 에 대해 (A+(B+C)) -> (A+B+C) 같은 flatten 수행"""
    if node.kind == "op" and node.op in ("+", "*"):
        new_children = []
        for c in node.children:
            flatten_associative(c)
            if c.kind == "op" and c.op == node.op:
                new_children.extend(c.children)
            else:
                new_children.append(c)
        node.children = new_children
    elif node.kind == "op":
        for c in node.children:
            flatten_associative(c)

def apply_commutativity(node: Node, rng: random.Random) -> None:
    """+, * 에서 피연산자 순서를 랜덤하게 섞기"""
    if node.kind == "op":
        for c in node.children:
            apply_commutativity(c, rng)
        if node.op in ("+", "*") and len(node.children) > 1:
            if rng.random() < 0.7:
                rng.shuffle(node.children)

def random_group_associative(node: Node, rng: random.Random) -> None:
    """+, * 의 n-항 연산을 랜덤한 이진트리 구조로 재구성"""
    if node.kind == "op":
        for c in node.children:
            random_group_associative(c, rng)
        if node.op in ("+", "*") and len(node.children) > 2:
            if rng.random() < 0.7:
                children = node.children
                cur = children[0]
                for child in children[1:]:
                    if rng.random() < 0.5:
                        cur = Node("op", op=node.op, children=[cur, child])
                    else:
                        cur = Node("op", op=node.op, children=[child, cur])
                node.kind = cur.kind
                node.op = cur.op
                node.value = cur.value
                node.children = cur.children

def try_distribute(node: Node, rng: random.Random) -> Node:
    """분배/역분배 일부 적용"""
    if node.kind != "op":
        return node

    # 먼저 아래쪽부터 처리
    node.children = [try_distribute(c, rng) for c in node.children]

    # 분배법칙: A*(B±C) 또는 (A±B)*C
    if node.op == "*":
        A, B = node.children
        if B.kind == "op" and B.op in ("+", "-"):
            if rng.random() < 0.5 and len(B.children) == 2:
                B1, B2 = B.children
                left = Node("op", op="*", children=[A.copy(), B1.copy()])
                right = Node("op", op="*", children=[A.copy(), B2.copy()])
                return Node("op", op=B.op, children=[left, right])
        if A.kind == "op" and A.op in ("+", "-"):
            if rng.random() < 0.5 and len(A.children) == 2:
                A1, A2 = A.children
                left = Node("op", op="*", children=[A1.copy(), B.copy()])
                right = Node("op", op="*", children=[A2.copy(), B.copy()])
                return Node("op", op=A.op, children=[left, right])

    # 역분배: A*B + A*C -> A*(B+C)
    if node.op in ("+", "-") and len(node.children) == 2 and rng.random() < 0.3:
        L, R = node.children
        if (
            L.kind == "op" and L.op == "*" and
            R.kind == "op" and R.op == "*"
        ):
            La, Lb = L.children
            Ra, Rb = R.children

            def same(a: Node, b: Node) -> bool:
                return to_string(a) == to_string(b)

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

# ===== 문자열로 다시 만들기 =====

def to_string(node: Node, parent_prec: int = 0, rng: Optional[random.Random] = None) -> str:
    """우선순위를 지키면서 문자열로 변환. 가끔 의미 없는 괄호도 추가."""
    if node.kind == "num":
        return node.value

    op = node.op
    if op in ("+", "-"):
        prec = 1
    else:  # '*', '//'
        prec = 2

    left, right = node.children
    s_left = to_string(left, prec, rng)
    s_right = to_string(right, prec, rng)
    s = s_left + op + s_right

    need_paren = prec < parent_prec
    if need_paren or (rng is not None and rng.random() < 0.2):
        return "(" + s + ")"
    return s

# ===== 외부에서 부를 함수 =====

def generate_equivalents(expr_str: str, num_variants: int = 5, seed: Optional[int] = None) -> List[str]:
    """
    input AE 하나를 받아, 의미가 같은 다른 AE들을 최대 num_variants개 생성.
    """
    rng = random.Random(seed)
    base_ast = parse_expression(expr_str)
    variants = set()

    # 원본도 포함 (정규화된 형태)
    variants.add(to_string(base_ast))

    trials = 0
    max_trials = num_variants * 20  # 너무 오래 안 돌게 제한
    while len(variants) < num_variants and trials < max_trials:
        trials += 1
        ast = base_ast.copy()

        # 몇 가지 랜덤 변환 적용
        flatten_associative(ast)
        apply_commutativity(ast, rng)
        ast = try_distribute(ast, rng)
        random_group_associative(ast, rng)

        s = to_string(ast, rng=rng)
        variants.add(s)

    return list(variants)