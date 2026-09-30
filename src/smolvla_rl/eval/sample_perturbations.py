import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUITE = "libero_goal"
OUT = ROOT / "configs" / "libero_plus_sample.json"


def classification(clone: Path) -> list[dict]:
    path = clone / "libero" / "libero" / "benchmark" / "task_classification.json"
    return json.loads(path.read_text())[SUITE]


def suite_task_names() -> list[str]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[SUITE]()
    return [suite.get_task(i).name for i in range(suite.n_tasks)]


def stratified_sample(entries: list[dict], per_category: int, seed: int) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = {}
    for entry in entries:
        by_category.setdefault(entry["category"], []).append(entry["name"])
    sample = {}
    for index, category in enumerate(sorted(by_category)):
        names = sorted(by_category[category])
        rng = np.random.default_rng([seed, index])
        chosen = rng.permutation(names)[:per_category]
        sample[category] = sorted(chosen.tolist())
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", default=os.path.expanduser("~/libero-plus"))
    parser.add_argument("--per-category", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    entries = classification(Path(args.clone))
    sample = stratified_sample(entries, args.per_category, args.seed)
    names = suite_task_names()
    index_of = {name: i for i, name in enumerate(names)}

    missing = [n for group in sample.values() for n in group if n not in index_of]
    assert not missing, f"{len(missing)} sampled names absent from the suite, e.g. {missing[:3]}"

    task_ids = sorted(index_of[n] for group in sample.values() for n in group)
    category_of = {index_of[n]: category for category, group in sample.items() for n in group}
    OUT.write_text(json.dumps({
        "suite": SUITE,
        "source": "LIBERO-plus @ 4976dc3 task_classification.json",
        "per_category": args.per_category,
        "seed": args.seed,
        "n_tasks": len(task_ids),
        "task_ids": task_ids,
        "category_of": {str(k): v for k, v in sorted(category_of.items())},
    }, indent=1))
    for category, group in sorted(sample.items()):
        print(f"{category:24s} {len(group)}")
    print(f"total {len(task_ids)} variants -> {OUT}")


if __name__ == "__main__":
    main()
