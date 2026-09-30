import argparse
import csv
import random
from collections import defaultdict


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
    parser.add_argument("--training", default="logs/rl_episodes.csv")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--block", type=int, default=20)
    args = parser.parse_args()
    runs = [f"a1_rl_s{k}" for k in args.seeds.split(",")]
    rows = [r for r in csv.DictReader(open(args.training)) if r["run_id"] in runs]
    for run in runs:
        rise, p = trend([r for r in rows if r["run_id"] == run], args.permutations, args.block)
        print(f"{run}: training slope {rise:+.2f} points per block, one sided permutation p {p:.4f}")
    rise, p = trend(rows, args.permutations, args.block)
    print(f"pooled: {rise:+.2f} points per block, one sided permutation p {p:.4f}")


if __name__ == "__main__":
    main()
