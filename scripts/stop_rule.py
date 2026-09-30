import argparse
import csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("run_id")
parser.add_argument("--episodes", default="logs/rl_episodes.csv")
parser.add_argument("--block", type=int, default=20)
parser.add_argument("--margin", type=int, default=30)
args = parser.parse_args()

rows = sorted((r for r in csv.DictReader(open(args.episodes)) if r["run_id"] == args.run_id), key=lambda r: int(r["round"]))
first, blocks = {}, defaultdict(list)
for r in rows:
    key = (r["task_id"], r["init_state"])
    first.setdefault(key, int(r["success"]))
    blocks[int(r["round"]) // args.block].append((key, int(r["success"])))
for b in sorted(blocks):
    now = sum(s for _, s in blocks[b])
    before = sum(first[k] for k, _ in blocks[b])
    flag = "STOP" if now < before - args.margin else "ok"
    print(f"block {b}: {now}/{len(blocks[b])} against first visits of the same pairs {before}/{len(blocks[b])}, "
          f"difference {now - before:+d}, {flag}")
