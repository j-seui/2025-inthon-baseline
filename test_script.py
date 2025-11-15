import random
from dataloader import _gen_expr
from augmentation import generate_equivalents

def main():
    rng = random.Random()
    expr = _gen_expr(rng, depth=4, num_digits=(1,5))
    print("Original expression:", expr)
    equivalents = generate_equivalents(expr[0], num_variants=5, seed=42)
    print("Equivalent expressions:")
    for eq in equivalents:
        print(eq)

if __name__ == "__main__":
    main()