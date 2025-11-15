import random
from dataloader import _gen_expr
from augmentation import generate_equivalents

def main():
    rng = random.Random(42)
    expr = _gen_expr(rng, depth=5, num_digits=(1,3))
    print("Original expression:", expr)

    # equivalents = generate_equivalents(expr, num_variants=5, seed=42)
    # print("Equivalent expressions:")
    # for eq in equivalents:
    #     print(eq)

if __name__ == "__main__":
    main()