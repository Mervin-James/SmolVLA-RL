import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import benchmark

REPO_ID = "lerobot/libero"
SUITE = "libero_goal"
BUDGETS = [1, 25]
SEEDS = [0, 1, 2, 3, 4]
OUT = Path(__file__).resolve().parents[3] / "configs" / "subsets" / f"{SUITE}.json"


def load_frames(revision: str) -> tuple[LeRobotDataset, dict[str, np.ndarray]]:
    ds = LeRobotDataset(REPO_ID, revision=revision, download_videos=False)
    columns = ds.hf_dataset.select_columns(["episode_index", "task_index", "action"]).with_format("numpy")[:]
    columns["action"] = np.stack(columns["action"])
    return ds, columns


def episode_table(ds: LeRobotDataset, frames: dict[str, np.ndarray]) -> pd.DataFrame:
    task_by_index = {int(i): task for task, i in ds.meta.tasks["task_index"].items()}
    per_frame = pd.DataFrame({"episode_index": frames["episode_index"], "task_index": frames["task_index"]})
    tasks_per_episode = per_frame.groupby("episode_index")["task_index"].unique()
    assert tasks_per_episode.map(len).eq(1).all(), "an episode spans more than one task"
    episodes = ds.meta.episodes.select_columns(["episode_index", "length"]).to_pandas()
    episodes["task"] = episodes["episode_index"].map(tasks_per_episode.map(lambda t: task_by_index[int(t[0])]))
    return episodes


def suite_tasks(suite_name: str) -> list[str]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    return [suite.get_task(i).language for i in range(suite.n_tasks)]


def draw_subsets(pool: dict[str, list[int]], tasks: list[str]) -> dict[int, dict[int, list[int]]]:
    subsets = {}
    for seed in SEEDS:
        subsets[seed] = {budget: [] for budget in BUDGETS}
        for task_index, task in enumerate(tasks):
            order = np.random.default_rng([seed, task_index]).permutation(pool[task]).tolist()
            for budget in BUDGETS:
                subsets[seed][budget].extend(order[:budget])
    return subsets


def subset_mean_shift(frames: dict[str, np.ndarray], episodes: list[int], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    actions = frames["action"][np.isin(frames["episode_index"], episodes)]
    return (actions.mean(0) - mean) / std


def main() -> None:
    revision = HfApi().dataset_info(REPO_ID).sha
    ds, frames = load_frames(revision)
    episodes = episode_table(ds, frames)
    print(f"[1] {REPO_ID} @ {revision}")
    print(f"[2] {len(episodes)} episodes, {episodes['task'].nunique()} tasks, {len(ds)} frames")

    tasks = suite_tasks(SUITE)
    missing = [task for task in tasks if task not in set(episodes["task"])]
    assert not missing, f"benchmark tasks absent from dataset: {missing}"
    print(f"[3] {len(tasks)} {SUITE} tasks matched")

    pool = {task: sorted(episodes.loc[episodes["task"] == task, "episode_index"].tolist()) for task in tasks}
    for task_index, task in enumerate(tasks):
        print(f"    task {task_index}: {len(pool[task]):3d} demos | {task}")
    assert min(len(demos) for demos in pool.values()) >= max(BUDGETS)

    subsets = draw_subsets(pool, tasks)
    lengths = dict(zip(episodes["episode_index"], episodes["length"]))
    for seed in SEEDS:
        for budget in BUDGETS:
            chosen = [lengths[e] for e in subsets[seed][budget]]
            print(f"[5] seed {seed} budget {budget:2d}: {len(chosen):3d} episodes, {sum(chosen):5d} frames, "
                  f"length {min(chosen)}..{max(chosen)}")

    stats = ds.meta.stats["action"]
    mean, std = np.asarray(stats["mean"], dtype=float), np.asarray(stats["std"], dtype=float)
    print(f"[6] global action mean {np.round(mean, 3)}")
    print(f"    global action std  {np.round(std, 3)}")
    for budget in BUDGETS:
        shift = subset_mean_shift(frames, subsets[0][budget], mean, std)
        print(f"    budget {budget:2d} subset mean shift (global std units): {np.round(shift, 2)}")

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "dataset": REPO_ID,
        "dataset_revision": revision,
        "suite": SUITE,
        "tasks": tasks,
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "episodes": {str(s): {str(b): subsets[s][b] for b in BUDGETS} for s in SEEDS},
        "normalization": "lerobot ds.meta.stats over the full dataset, identical for every arm",
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "made_by_git_sha": git_sha,
    }, indent=1))
    print(f"[7] wrote {OUT}")


if __name__ == "__main__":
    main()
