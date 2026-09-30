import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def build_command(base: dict, condition_name: str, condition: dict, seed: int, steps: int,
                  batch_size: int, save_freq: int) -> list[str]:
    subsets = json.loads((ROOT / base["subsets"]).read_text())
    episodes = subsets["episodes"][str(seed)][str(condition["budget"])]
    run_id = f"{condition_name}_s{seed}"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args = {
        **{f"policy.{k}": v for k, v in base["policy"].items()},
        **{f"dataset.{k}": v for k, v in base["dataset"].items()},
        "dataset.episodes": json.dumps(episodes, separators=(",", ":")),
        "rename_map": json.dumps(base["rename_map"], separators=(",", ":")),
        "num_workers": base["train"]["num_workers"],
        "save_freq": save_freq,
        "log_freq": base["train"]["log_freq"],
        "batch_size": batch_size,
        "steps": steps,
        "seed": seed,
        "wandb.enable": False,
        "job_name": run_id,
        "output_dir": str(ROOT / os.environ.get("OUTPUTS", "outputs") / run_id / stamp),
    }
    return [str(Path(sys.executable).with_name("lerobot-train")), *(f"--{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in args.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_yaml(ROOT / "configs/base.yaml")
    condition = load_yaml(ROOT / "configs/conditions" / f"{args.condition}.yaml")
    steps = args.steps or base["train"]["steps"]
    batch_size = args.batch_size or base["train"]["batch_size"]
    assert steps and batch_size, "set train.steps and train.batch_size in base.yaml or pass them"

    save_freq = args.save_freq or base["train"]["save_freq"]
    command = build_command(base, args.condition, condition, args.seed, steps, batch_size, save_freq)
    print(" \\\n  ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
