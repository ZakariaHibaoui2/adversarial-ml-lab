"""advlab command line: run the full experiment suite and print / save results."""
from __future__ import annotations

import argparse
import json
import sys

from .experiments import run_all


def table(name: str, curve: dict) -> str:
    rows = [f"{name}", f"  {'eps':>6} {'noise':>8} {'FGSM':>8} {'PGD-20':>8}"]
    for i, e in enumerate(curve["eps"]):
        rows.append(f"  {e:>6.2f} {curve['noise'][i]:>8.1%} {curve['fgsm'][i]:>8.1%} {curve['pgd'][i]:>8.1%}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="advlab", description="Adversarial ML experiments (digits, NumPy MLP)")
    ap.add_argument("--train-eps", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also save raw results to this file")
    a = ap.parse_args(argv)

    r = run_all(a.seed, a.train_eps, a.epochs)
    print(table("Standard training (test accuracy under attack)", r["standard"]))
    print()
    print(table(f"Adversarial training, PGD eps={r['train_eps']}", r["adversarial_training"]))
    print(f"\nPGD (eps={r['train_eps']}) success rate on the standard model : {r['pgd_success_rate_standard']:.1%}")
    print(f"Targeted PGD 'turn every digit into 0' (eps=0.2)          : {r['targeted_to_zero_rate']:.1%}")
    d = r["detection"]
    print("\nDetection of PGD examples (true-positive rate; calibrated on held-out clean data)")
    print(f"  {'eps':>6} {'manifold':>9} {'squeezing':>10}")
    for i, e in enumerate(d["eps"]):
        print(f"  {e:>6.2f} {d['manifold_tpr'][i]:>9.1%} {d['squeeze_tpr'][i]:>10.1%}")
    print(f"  {'FPR':>6} {d['manifold_fpr']:>9.1%} {d['squeeze_fpr']:>10.1%}   (clean test images flagged)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
