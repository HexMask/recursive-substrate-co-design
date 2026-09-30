from __future__ import annotations

import argparse

from .core import evolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RSCD Generation Zero toy search.")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    history = evolve(args.population, args.generations, args.seed)
    best = history[-1][0]
    print(f"best fitness: {best.result.fitness:.3f}")
    print(f"scores: {best.result.scores}")
    print(f"candidate: {best.candidate}")


if __name__ == "__main__":
    main()
