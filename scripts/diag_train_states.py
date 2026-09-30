import argparse
import csv
import math
from collections import defaultdict


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(max(b, c), n + 1)) / 2 ** n)


def load(path):
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        out[r["run_id"]][r["task_id"], int(r["episode_idx"])] = int(r["success"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", default="logs/episodes_diag.csv")
    parser.add_argument("--primary", default="logs/episodes.csv")
    parser.add_argument("--sft", default="diag50_a1_sft_s1")
    parser.add_argument("--rl", default="diag50_a1_rl_s1")
    parser.add_argument("--sft-primary", default="a1_sft_s1")
    parser.add_argument("--rl-primary", default="a1_rl_s1")
    parser.add_argument("--first-train", type=int, default=20)
    args = parser.parse_args()
    diag, primary = load(args.diag), load(args.primary)
    sft, rl = diag[args.sft], diag[args.rl]
    assert len(sft) == len(rl) == 500, (len(sft), len(rl))
    for name, keep in (("training states 20 to 49", lambda e: e >= args.first_train),
                       ("held out states 0 to 19", lambda e: e < args.first_train)):
        keys = sorted(k for k in sft if keep(k[1]))
        b = sum(rl[k] > sft[k] for k in keys)
        c = sum(rl[k] < sft[k] for k in keys)
        s, r = 100 * sum(sft[k] for k in keys) / len(keys), 100 * sum(rl[k] for k in keys) / len(keys)
        print(f"{name}: n {len(keys)}, A1-SFT {s:.1f}, A1-RL {r:.1f}, lift {r - s:+.1f}, fixed {b}, broke {c}, "
              f"exact McNemar p {mcnemar(b, c):.3f}")
    for label, run, ref in (("A1-SFT", sft, primary[args.sft_primary]), ("A1-RL", rl, primary[args.rl_primary])):
        keys = sorted(k for k in ref if k in run)
        same = sum(run[k] == ref[k] for k in keys)
        print(f"{label}, episodes 0 to 19 of the 50 episode run against the 20 episode primary eval: {same}/{len(keys)} identical outcomes, "
              f"{sum(ref[k] for k in keys)} and {sum(run[k] for k in keys)} successes")


if __name__ == "__main__":
    main()
