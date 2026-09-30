import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def build_command(policy_path: str, output_dir: str, episodes: int, batch_size: int,
                  parallel_tasks: int, seed: int, task_ids: str | None, env_type: str,
                  episode_length: int | None = None) -> list[str]:
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    suite = json.loads((ROOT / base["subsets"]).read_text())["suite"]
    args = {
        "policy.path": policy_path,
        "policy.n_action_steps": base["policy"]["n_action_steps"],
        "policy.device": base["policy"]["device"],
        "rename_map": json.dumps(base["rename_map"], separators=(",", ":")),
        "env.type": env_type,
        "env.task": suite,
        "env.max_parallel_tasks": parallel_tasks,
        "eval.n_episodes": episodes,
        "eval.batch_size": batch_size,
        "seed": seed,
        "output_dir": output_dir,
    }
    if task_ids:
        args["env.task_ids"] = task_ids
    if episode_length:
        args["env.episode_length"] = episode_length
    return [str(Path(sys.executable).with_name("lerobot-eval")), *(f"--{k}={v}" for k, v in args.items())]


def resolve_task_ids(raw: str | None, task_ids_file: str | None, shard: int, num_shards: int) -> str | None:
    if task_ids_file:
        raw = json.dumps(json.loads(Path(task_ids_file).read_text())["task_ids"])
    if raw is None:
        return None
    ids = json.loads(raw) if raw.lstrip().startswith("[") else [int(x) for x in raw.split(",")]
    return json.dumps(ids[shard::num_shards], separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--parallel-tasks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task-ids")
    parser.add_argument("--task-ids-file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--env-type", default="libero")
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--noise-method")
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--noise-joint", action="store_true")
    parser.add_argument("--stored-dtype", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_ids = resolve_task_ids(args.task_ids or os.environ.get("TASK_IDS"), args.task_ids_file,
                                args.shard, args.num_shards)
    command = build_command(args.policy_path, args.output_dir, args.episodes,
                            min(args.batch_size, args.episodes), args.parallel_tasks, args.seed,
                            task_ids, args.env_type, args.episode_length)
    if args.noise_method or args.stored_dtype:
        patches = ["--noise-method", args.noise_method, "--noise-level", str(args.noise_level)] if args.noise_method else []
        patches += ["--noise-joint"] if args.noise_joint else []
        patches += ["--stored-dtype"] if args.stored_dtype else []
        command = [sys.executable, "-m", "smolvla_rl.rl.noisy_eval", *patches, *command[1:]]
    print(" \\\n  ".join(command), flush=True)
    if not args.dry_run:
        os.chdir(ROOT)
        os.execv(command[0], command)


if __name__ == "__main__":
    main()
