import argparse
import csv
import math
from collections import defaultdict
from statistics import mean, stdev

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
A25_LOWER = 81.2


def log_pmf(k, n, p):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def tail(k, n, p, upper):
    ks = range(k, n + 1) if upper else range(0, k + 1)
    return sum(math.exp(log_pmf(i, n, p)) for i in ks)


def solve(f, target):
    lo, hi = 1e-12, 1 - 1e-12
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def clopper_pearson(k, n):
    low = 0.0 if k == 0 else solve(lambda p: tail(k, n, p, True), 0.025)
    high = 1.0 if k == n else solve(lambda p: -tail(k, n, p, False), -0.025)
    return 100 * low, 100 * high


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(max(b, c), n + 1)) / 2 ** n)


def interval(xs):
    m, h = mean(xs), T975[len(xs) - 1] * stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def score(low, high, bar, above):
    if above:
        return "supported" if low > bar else "falsified" if high <= bar else "undetermined"
    return "supported" if low >= bar else "falsified" if high < bar else "undetermined"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="logs/episodes.csv")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = args.seeds.split(",")
    ep = defaultdict(dict)
    for r in csv.DictReader(open(args.episodes)):
        ep[r["condition"], r["stage"], r["seed"]][r["task_id"], r["episode_idx"]] = int(r["success"])
    lifts, rates, pooled = [], [], defaultdict(lambda: [0, 0])
    print("seed  A1-SFT  A1-RL  lift  fixed  broke  McNemar p")
    for k in seeds:
        rl, sft = ep["a1_rl", "rl", k], ep["a1_sft", "sft", k]
        keys = sorted(set(rl) & set(sft))
        assert len(keys) == len(rl) == len(sft) == 200, (k, len(keys), len(rl), len(sft))
        b = sum(rl[x] > sft[x] for x in keys)
        c = sum(rl[x] < sft[x] for x in keys)
        r_rate, s_rate = 100 * mean(rl[x] for x in keys), 100 * mean(sft[x] for x in keys)
        lifts.append(r_rate - s_rate)
        rates.append(r_rate)
        for name, d in (("A1-RL", rl), ("A1-SFT", sft)):
            pooled[name][0] += sum(d.values())
            pooled[name][1] += len(d)
        print(f"{k:>4}  {s_rate:6.1f}  {r_rate:5.1f}  {r_rate - s_rate:+5.1f}  {b:5d}  {c:5d}  {mcnemar(b, c):9.3f}")
    for s in "01234":
        d = ep["a25_sft", "sft", s]
        pooled["A25-SFT"][0] += sum(d.values())
        pooled["A25-SFT"][1] += len(d)
    df = len(seeds) - 1
    m, lo, hi = interval(lifts)
    print(f"\npaired lift: mean {m:+.2f}, sd {stdev(lifts):.2f}, 95% t interval (df {df}) [{lo:+.2f}, {hi:+.2f}]")
    print(f"prediction (a), lift above 5: {score(lo, hi, 5.0, True)}")
    m2, lo2, hi2 = interval(rates)
    print(f"A1-RL seed mean {m2:.2f}, sd {stdev(rates):.2f}, 95% t interval (df {df}) [{lo2:.2f}, {hi2:.2f}]")
    print(f"prediction (b), A1-RL at or above {A25_LOWER}: {score(lo2, hi2, A25_LOWER, False)}")
    a25 = [100 * mean(ep["a25_sft", "sft", s].values()) for s in "01234"]
    gap = mean(a25) - m2
    band = "above 20: decisive" if gap > 20 else "10 to 20: needs the 10 point power figure" if gap >= 10 else "under 10: needs the 5 point power figure"
    print(f"decision rule gap, A25-SFT {mean(a25):.2f} minus A1-RL {m2:.2f} = {gap:.2f} ({band}); eval episodes per condition {200 * len(seeds)}")
    print("\npooled episodes, Clopper Pearson 95%:")
    for name, (k, n) in pooled.items():
        low, high = clopper_pearson(k, n)
        print(f"  {name:8s} {k}/{n} = {100 * k / n:.1f} [{low:.2f}, {high:.2f}]")


if __name__ == "__main__":
    main()
