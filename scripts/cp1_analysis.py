import argparse
import csv
import math
import random
from collections import defaultdict

SFT = "sft_task4"
RUNS = {"a1_sft_s0_task4_accum20_joint15_lr2.5e-7": "a1_sft_s0_task4_accum20_joint15_lr2.5e-7_task4",
        "a1_sft_s0_task4_accum20_joint15_lr2.5e-7_seed10": "a1_sft_s0_task4_accum20_joint15_lr2.5e-7_seed10_task4"}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(max(b, c), n + 1)) / 2 ** n)


def slope(points):
    n = len(points)
    mb = sum(b for b, _ in points) / n
    ms = sum(s for _, s in points) / n
    return sum((b - mb) * (s - ms) for b, s in points) / sum((b - mb) ** 2 for b, _ in points)


def trend(rows, permutations, block):
    points = [(int(r["round"]) // block, int(r["success"])) for r in rows]
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[r["run_id"], r["task_id"], r["init_state"]].append(i)
    observed, hits = slope(points), 0
    rng = random.Random(0)
    for _ in range(permutations):
        shuffled = list(points)
        for idx in strata.values():
            blocks = [points[i][0] for i in idx]
            rng.shuffle(blocks)
            for i, b in zip(idx, blocks):
                shuffled[i] = (b, points[i][1])
        hits += slope(shuffled) >= observed
    return 100 * observed, (hits + 1) / (permutations + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", default="logs/episodes_task4.csv")
    parser.add_argument("--training", default="logs/rl_episodes.csv")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--block", type=int, default=20)
    args = parser.parse_args()
    evals = defaultdict(dict)
    for r in csv.DictReader(open(args.evals)):
        evals[r["run_id"]][r["episode_idx"]] = int(r["success"])
    training = [r for r in csv.DictReader(open(args.training)) if r["run_id"] in RUNS]
    sft = evals[SFT]
    for run, eval_id in RUNS.items():
        rl = evals[eval_id]
        b = sum(rl[k] > sft[k] for k in sft)
        c = sum(rl[k] < sft[k] for k in sft)
        rise, p = trend([r for r in training if r["run_id"] == run], args.permutations, args.block)
        print(f"{run}: ODE {sum(rl.values())}/{len(rl)} against SFT {sum(sft.values())}/{len(sft)}, fixed {b}, broke {c}, "
              f"McNemar p {mcnemar(b, c):.3f}; training slope {rise:+.2f} points per block, p {p:.4f}")
    rise, p = trend(training, args.permutations, args.block)
    print(f"pooled training slope {rise:+.2f} points per block, one sided permutation p {p:.4f}")


if __name__ == "__main__":
    main()
