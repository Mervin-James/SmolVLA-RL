import argparse
import csv
import itertools
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default="logs/episodes.csv")
    parser.add_argument("--diag", default="logs/episodes_diag.csv")
    parser.add_argument("--repeat", default="logs/episodes_repeat.csv")
    args = parser.parse_args()
    runs = defaultdict(dict)
    for r in csv.DictReader(open(args.primary)):
        if r["run_id"] == "a1_sft_s1":
            runs["primary, 20 episodes per task"][r["task_id"], int(r["episode_idx"])] = int(r["success"])
    for r in csv.DictReader(open(args.repeat)):
        runs[f"{r['run_id']}, 20 episodes per task"][r["task_id"], int(r["episode_idx"])] = int(r["success"])
    for r in csv.DictReader(open(args.diag)):
        if r["run_id"] == "diag50_a1_sft_s1" and int(r["episode_idx"]) < 20:
            runs["diag50, 50 episodes per task"][r["task_id"], int(r["episode_idx"])] = int(r["success"])
    keys = sorted(set.intersection(*(set(v) for v in runs.values())))
    for name, run in runs.items():
        print(f"{name}: {sum(run[k] for k in keys)}/{len(keys)} successes")
    for a, b in itertools.combinations(runs, 2):
        print(f"{a} against {b}: {sum(runs[a][k] != runs[b][k] for k in keys)} of {len(keys)} outcomes differ")


if __name__ == "__main__":
    main()
