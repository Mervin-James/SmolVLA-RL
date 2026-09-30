import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def check_variant(job: tuple[str, int]) -> tuple[int, float, str | None]:
    suite, task_id = job
    start = time.time()
    try:
        import numpy as np
        from lerobot.envs.configs import LiberoPlusEnv

        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            envs = LiberoPlusEnv(task=suite, task_ids=[task_id]).create_envs(n_envs=1)
            env = envs[suite][task_id]
            name, instruction = env.call("task")[0], env.call("task_description")[0]
            if "_language_" not in name and any(ch.isdigit() for ch in instruction):
                raise ValueError(f"instruction carries the variant encoding: {instruction!r}")
            env.reset(seed=1000)
            env.step(np.zeros((1, 7), dtype=np.float32))
            env.close()
        return task_id, time.time() - start, None
    except Exception:
        return task_id, time.time() - start, traceback.format_exc(limit=-2).strip()


def available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def riskiest_first(task_ids: list[int], category_of: dict[str, str]) -> list[int]:
    by_category: dict[str, list[int]] = {}
    for task_id in task_ids:
        by_category.setdefault(category_of[str(task_id)], []).append(task_id)
    order = by_category.pop("Sensor Noise", [])
    queues = list(by_category.values())
    while any(queues):
        order += [queue.pop(0) for queue in queues if queue]
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=str(ROOT / "configs/libero_plus_sample.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--deadline", type=float, default=50)
    args = parser.parse_args()

    start = time.time()
    sample = json.loads(Path(args.sample).read_text())
    category_of = sample["category_of"]
    order = riskiest_first(sample["task_ids"], category_of)
    results: dict[int, tuple[float, str | None]] = {}
    min_available = available_gb()

    pool = mp.get_context("fork").Pool(args.workers)
    pending = pool.imap_unordered(check_variant, [(sample["suite"], task_id) for task_id in order])
    try:
        while len(results) < len(order):
            remaining = args.deadline - (time.time() - start)
            if remaining <= 0:
                break
            try:
                task_id, seconds, error = pending.next(timeout=remaining)
            except mp.TimeoutError:
                break
            results[task_id] = (seconds, error)
            min_available = min(min_available, available_gb())
            if error:
                print(f"FAIL {task_id} {category_of[str(task_id)]} after {seconds:.1f} s\n{error}\n", flush=True)
    finally:
        pool.terminate()

    elapsed = time.time() - start
    failed = sorted(task_id for task_id, (_, error) in results.items() if error)
    untested = sorted(set(order) - set(results))
    slowest = max(results.items(), key=lambda item: item[1][0], default=(None, (0.0, None)))
    print(f"envs: {len(results) - len(failed)}/{len(order)} variants built, reset and stepped in {elapsed:.1f} s "
          f"on {args.workers} workers, min RAM available {min_available:.1f} GB, "
          f"slowest {slowest[0]} at {slowest[1][0]:.1f} s")
    for category in sorted(set(category_of.values())):
        ids = [task_id for task_id in order if category_of[str(task_id)] == category]
        ok = sum(1 for task_id in ids if task_id in results and not results[task_id][1])
        print(f"  {category:22s} {ok:3d}/{len(ids)}")
    if untested:
        print(f"untested, deadline reached: {untested}")
    if failed:
        print(f"failed: {failed}")
    sys.exit(1 if failed else 2 if untested else 0)


if __name__ == "__main__":
    main()
