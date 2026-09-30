import argparse
import csv
import json
import math
from collections import defaultdict
from statistics import mean, stdev

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(max(b, c), n + 1)) / 2 ** n)


def interval(xs):
    m, h = mean(xs), T975[len(xs) - 1] * stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="logs/episodes_plus.csv")
    parser.add_argument("--sample", default="configs/libero_plus_sample.json")
    args = parser.parse_args()
    category = json.load(open(args.sample))["category_of"]
    ep = defaultdict(dict)
    for r in csv.DictReader(open(args.episodes)):
        ep[r["condition"], r["seed"]][r["task_id"]] = int(r["success"])
    lifts = []
    by_cat = defaultdict(lambda: defaultdict(list))
    print("seed  A1-SFT  A1-RL   lift  fixed  broke  McNemar p")
    for k in "01234":
        sft, rl = ep["a1_sft", k], ep["a1_rl", k]
        keys = sorted(set(sft) & set(rl))
        assert len(keys) == len(sft) == len(rl) == 196, (k, len(keys))
        b = sum(rl[x] > sft[x] for x in keys)
        c = sum(rl[x] < sft[x] for x in keys)
        s, r = 100 * mean(sft[x] for x in keys), 100 * mean(rl[x] for x in keys)
        lifts.append(r - s)
        for x in keys:
            by_cat[category[x]]["sft"].append(sft[x])
            by_cat[category[x]]["rl"].append(rl[x])
        print(f"{k:>4}  {s:6.1f}  {r:5.1f}  {r - s:+5.1f}  {b:5d}  {c:5d}  {mcnemar(b, c):9.3f}")
    print("\ncondition  seed mean  sd     95% t interval (df 4)")
    for condition in ("a1_sft", "a25_sft", "a1_rl"):
        rates = [100 * mean(ep[condition, k].values()) for k in "01234"]
        cm, clo, chi = interval(rates)
        print(f"{condition:9s}  {cm:9.2f}  {stdev(rates):5.2f}  [{clo:.2f}, {chi:.2f}]")
    m, lo, hi = interval(lifts)
    print(f"\npaired LIBERO-Plus lift: mean {m:+.2f}, sd {stdev(lifts):.2f}, 95% t interval (df 4) [{lo:+.2f}, {hi:+.2f}]")
    for k in "01234":
        for x, v in ep["a25_sft", k].items():
            by_cat[category[x]]["a25"].append(v)
    print("\ncategory                  A1-SFT   A1-RL    lift  A25-SFT   n")
    for cat in sorted(by_cat):
        s, r, a = (100 * mean(by_cat[cat][c]) for c in ("sft", "rl", "a25"))
        print(f"{cat:24s} {s:7.1f} {r:7.1f} {r - s:+7.1f} {a:8.1f} {len(by_cat[cat]['sft']):4d}")


if __name__ == "__main__":
    main()
