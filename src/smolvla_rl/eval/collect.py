import argparse
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COLUMNS = [
    "run_id", "condition", "demo_budget", "seed", "stage", "step", "task_id", "episode_idx",
    "success", "reward_total", "eval_distribution", "wallclock_s", "git_sha", "eval_dir",
]


def rows_from_eval(eval_dirs: list[Path], run_id: str, condition: str, budget: int, seed: int,
                   stage: str, step: int, distribution: str, git_sha: str) -> list[dict]:
    infos = [json.loads((d / "eval_info.json").read_text()) for d in eval_dirs]
    wallclock = max(info["overall"]["eval_s"] for info in infos)
    rows = []
    for eval_dir, info in zip(eval_dirs, infos):
        for group in info.get("per_task", []):
            metrics = group["metrics"]
            for episode_idx, success in enumerate(metrics["successes"]):
                rows.append({
                    "run_id": run_id, "condition": condition, "demo_budget": budget, "seed": seed,
                    "stage": stage, "step": step, "task_id": group["task_id"],
                    "episode_idx": episode_idx, "success": int(bool(success)),
                    "reward_total": metrics["sum_rewards"][episode_idx],
                    "eval_distribution": distribution, "wallclock_s": wallclock,
                    "git_sha": git_sha, "eval_dir": str(eval_dir),
                })
    return rows


def append(path: Path, rows: list[dict]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", nargs="+")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--stage", default="sft")
    parser.add_argument("--distribution", default="libero_goal")
    parser.add_argument("--out", default="logs/episodes.csv")
    parser.add_argument("--expect-task-ids")
    args = parser.parse_args()

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    rows = rows_from_eval([Path(d) for d in args.eval_dir], args.run_id, args.condition, args.budget,
                          args.seed, args.stage, args.step, args.distribution, git_sha)
    if args.expect_task_ids:
        expected = json.loads(Path(args.expect_task_ids).read_text())["task_ids"]
        got = [row["task_id"] for row in rows]
        if sorted(got) != sorted(expected):
            raise SystemExit(
                f"{args.run_id}: collected {len(got)} episodes for {len(set(got))} task_ids, "
                f"expected {len(expected)}; missing {sorted(set(expected) - set(got))}, "
                f"extra {sorted(set(got) - set(expected))}, "
                f"duplicated {sorted({t for t in got if got.count(t) > 1})}")
    append(ROOT / args.out, rows)
    successes = sum(row["success"] for row in rows)
    print(f"{args.run_id}: {successes}/{len(rows)} = {100 * successes / len(rows):.1f}% -> {args.out}")


if __name__ == "__main__":
    main()
