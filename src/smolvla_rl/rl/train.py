import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from lerobot.policies.factory import make_pre_post_processors

from .likelihood import NoiseConfig, chain_logprob, prefill
from .loss import LossConfig, StateBaseline, chunk_logprob, group_advantages, total_loss
from .rollout import Batch, TaskEnv, collect, to_batches
from .verify import load_policy

ROOT = Path(__file__).resolve().parents[3]
STEP_COLUMNS = ["run_id", "round", "update", "epoch", "task_id", "loss_total", "loss_policy", "kl_to_ref", "advantage_mean",
                "advantage_std", "ratio_mean", "ratio_clipfrac", "approx_kl_old", "grad_norm", "lr", "denoise_steps",
                "noise_level", "peak_vram_gb", "git_sha"]
EPISODE_COLUMNS = ["run_id", "round", "task_id", "init_state", "episode_idx", "seed", "success", "baseline",
                   "advantage", "episode_length", "git_sha"]
UPDATE_COLUMNS = ["run_id", "update", "last_round", "chunks", "loss_policy", "advantage_mean", "advantage_std",
                  "ratio_mean", "grad_norm", "lr", "approx_kl_step", "max_log_ratio_step", "denoise_steps",
                  "noise_level", "noise_joint", "git_sha"]


def default_checkpoint(condition: str, seed: int) -> str:
    runs = sorted(ROOT.glob(f"outputs/cp0/{condition}_s{seed}/*/checkpoints/008000/pretrained_model"))
    assert runs, f"no {condition}_s{seed} checkpoint under outputs/cp0"
    return str(runs[-1])


def trainable(policy) -> list[torch.nn.Parameter]:
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    upcast = [(n, p) for n, p in named if p.dtype != torch.float32]
    for _, p in upcast:
        p.data = p.data.float()
    moved = sum(p.numel() for _, p in upcast) / 1e6
    total = sum(p.numel() for _, p in named) / 1e6
    print(f"upcast {len(upcast)} of {len(named)} trainable tensors to fp32 ({moved:.1f}M of {total:.1f}M parameters), "
          f"e.g. {upcast[0][0] if upcast else 'none'}", flush=True)
    return [p for _, p in named]


def check_header(path: Path, columns: list[str]) -> None:
    if path.exists():
        with path.open() as handle:
            header = handle.readline().strip().split(",")
        assert header == columns, f"{path} has columns {header}, expected {columns}; move it aside first"


def append(path: Path, columns: list[str], rows: list[dict]) -> None:
    check_header(path, columns)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


RECIPE = ["condition", "seed", "checkpoint", "tasks", "train_states", "baseline", "baseline_window", "group", "epochs",
          "lr", "warmup_updates", "rounds_per_update", "clip", "noise_level", "noise_joint"]


def trainer_state(params, optimizer, baseline, visits, generator, next_round: int, args) -> dict:
    return {"next_round": next_round, "params": [p.detach().cpu().clone() for p in params],
            "optimizer": optimizer.state_dict(), "baseline": dict(baseline.history), "visits": dict(visits),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "generator": generator.get_state(), "recipe": {key: getattr(args, key) for key in RECIPE}}


def restore(state: dict, params, optimizer, baseline, visits, generator, args) -> int:
    recipe = {key: getattr(args, key) for key in RECIPE}
    changed = {k: (state["recipe"][k], recipe[k]) for k in RECIPE if state["recipe"][k] != recipe[k]}
    assert not changed, f"resume with a different recipe (saved, now): {changed}"
    with torch.no_grad():
        for param, saved in zip(params, state["params"], strict=True):
            param.copy_(saved.to(param.device))
    optimizer.load_state_dict(state["optimizer"])
    baseline.history = dict(state["baseline"])
    visits.update(state["visits"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    generator.set_state(state["generator"])
    return state["next_round"]


def recompute(policy, batch: Batch, config: NoiseConfig, n_action_steps: int, action_dim: int) -> torch.Tensor:
    prefix = prefill(policy.model, batch.embeddings, batch.pad_masks, batch.att_masks)
    return chunk_logprob(chain_logprob(policy, None, batch.chain(), config, prefix=prefix), n_action_steps, action_dim)


def update(policy, optimizer, params, data: list[Batch], noise: NoiseConfig, loss_config: LossConfig, epochs: int,
           dims: tuple[int, int], generator: torch.Generator, zero_advantage: bool = False):
    total = sum(len(b) for b in data)
    stats_rows = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        summed: dict[str, float] = {}
        for index in torch.randperm(len(data), generator=generator).tolist():
            batch = data[index].to("cuda")
            live = batch.live
            advantage = torch.zeros_like(batch.advantage[live]) if zero_advantage else batch.advantage[live]
            logprob = recompute(policy, batch, noise, *dims)[live]
            loss, stats = total_loss(logprob, batch.old_logprob[live], advantage, loss_config)
            weight = len(batch) / total
            (loss * weight).backward()
            for key, value in stats.items():
                summed[key] = summed.get(key, 0.0) + value * weight
        summed["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        summed.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"], denoise_steps=noise.num_steps,
                      noise_level=noise.noise_level, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                      chunks=total)
        stats_rows.append(summed)
    return stats_rows


def accumulate(policy, data: list[Batch], noise: NoiseConfig, loss_config: LossConfig,
               dims: tuple[int, int]) -> tuple[int, dict[str, float]]:
    chunks, summed = 0, {}
    for batch in data:
        batch = batch.to("cuda")
        live = batch.live
        logprob = recompute(policy, batch, noise, *dims)[live]
        loss, stats = total_loss(logprob, batch.old_logprob[live], batch.advantage[live], loss_config)
        (loss * len(batch)).backward()
        chunks += len(batch)
        for key, value in stats.items():
            summed[key] = summed.get(key, 0.0) + value * len(batch)
    return chunks, summed


def apply_update(policy, optimizer, params, chunks: int, summed: dict[str, float], data: list[Batch],
                 noise: NoiseConfig, dims: tuple[int, int]) -> dict[str, float]:
    for param in params:
        if param.grad is not None:
            param.grad /= chunks
    row = {key: value / chunks for key, value in summed.items()}
    row["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    row["approx_kl_step"], row["max_log_ratio_step"] = step_kl(policy, data, noise, dims)
    return row


def mean_gradient(policy, params, data: list[Batch], noise: NoiseConfig, loss_config: LossConfig,
                  dims: tuple[int, int]) -> torch.Tensor:
    for param in params:
        param.grad = None
    total = sum(len(b) for b in data)
    for batch in data:
        batch = batch.to("cuda")
        live = batch.live
        loss, _ = total_loss(recompute(policy, batch, noise, *dims)[live], batch.old_logprob[live],
                             batch.advantage[live], loss_config)
        (loss * len(batch) / total).backward()
    flat = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
    for param in params:
        param.grad = None
    return flat


def ratio_gap(policy, data: list[Batch], noise: NoiseConfig, dims: tuple[int, int]) -> tuple[float, float]:
    worst = 0.0
    with torch.no_grad():
        for batch in data:
            batch = batch.to("cuda")
            logprob = recompute(policy, batch, noise, *dims)
            worst = max(worst, ((logprob - batch.old_logprob)[batch.live].exp() - 1).abs().max().item())
        merged = Batch(*(torch.cat([getattr(b, f) for b in data]) for f in Batch.__dataclass_fields__)).to("cuda")
        merged_gap = ((recompute(policy, merged, noise, *dims) - merged.old_logprob)[merged.live].exp() - 1)
    return worst, merged_gap.abs().max().item()


def probe(policy, optimizer, params, env, preprocessor, postprocessor, noise, loss_config, dims, generator,
          states: list[int]) -> bool:
    tick = time.perf_counter()
    env.start_at(states)
    episodes = collect(policy, preprocessor, postprocessor, env, noise, seeds=list(range(env.group)))
    landed = env.state_ids()
    data = to_batches(episodes, group_advantages(episodes.success), lambda lp: chunk_logprob(lp, *dims))
    chunks = sum(len(b) for b in data)
    print(f"collected {chunks} chunks in {len(data)} policy calls from {env.group} episodes in "
          f"{time.perf_counter() - tick:.1f} s, success {episodes.success.tolist()}, lengths {episodes.length.tolist()}",
          flush=True)
    expected = [state + env.group for state in states]
    checks = [("0 episodes start at the set states", landed == expected,
               f"init_state_id {states} before reset, {landed} after, expected {expected}")]
    gap, merged = ratio_gap(policy, data, noise, dims)
    checks.append(("3 ratio == 1 on stored rollouts", gap < 1e-3,
                   f"max |ratio - 1| {gap:.2e} recomputed per policy call; merged into one batch it would be {merged:.2e}"))
    before = [p.detach().clone() for p in params]
    update(policy, optimizer, params, data, noise, loss_config, 1, dims, generator, zero_advantage=True)
    moved = max((p.detach() - b).abs().max().item() for p, b in zip(params, before))
    checks.append(("4 constant reward moves nothing", moved == 0.0, f"max |delta theta| {moved:.1e} after one epoch"))
    for batch in data:
        batch.advantage = torch.linspace(-1.0, 1.0, batch.advantage.numel())
    reference = mean_gradient(policy, params, data, noise, loss_config, dims)
    half = len(data) // 2
    first, first_stats = accumulate(policy, data[:half], noise, loss_config, dims)
    second, _ = accumulate(policy, data[half:], noise, loss_config, dims)
    split = torch.cat([p.grad.flatten() for p in params if p.grad is not None]) / (first + second)
    gap = ((split - reference).abs().max() / reference.abs().max()).item()
    checks.append(("accumulating two halves equals one mean", gap < 1e-4,
                   f"max relative gap {gap:.1e} over {first + second} chunks"))
    row = apply_update(policy, optimizer, params, first + second, first_stats, data, noise, dims)
    checks.append(("accumulated update steps and reports kl", row["approx_kl_step"] > 0 and
                   math.isfinite(row["approx_kl_step"]),
                   f"step kl {row['approx_kl_step']:.4f}, grad norm {row['grad_norm']:.2f}"))
    for param, saved in zip(params, before):
        param.data.copy_(saved)
    rows = update(policy, optimizer, params, data, noise, loss_config, 1, dims, generator)
    moved = max((p.detach() - b).abs().max().item() for p, b in zip(params, before))
    peak = max(r["peak_vram_gb"] for r in rows)
    checks.append(("synthetic advantage moves expert", moved > 0, f"max |delta theta| {moved:.1e}, peak {peak:.2f} GiB"))
    recipe = argparse.Namespace(**{key: key for key in RECIPE})
    saved = trainer_state(params, optimizer, StateBaseline(5), {0: 3}, generator, 7, recipe)
    weights = [p.detach().clone() for p in params]
    moments = [optimizer.state[p]["exp_avg"].clone() for p in params if p in optimizer.state]
    for param in params:
        param.data.add_(1.0)
    fresh, visits = torch.optim.AdamW(params, lr=optimizer.param_groups[0]["lr"], weight_decay=0.0), {}
    next_round = restore(saved, params, fresh, StateBaseline(5), visits, generator, recipe)
    exact = (all(torch.equal(p.detach(), w) for p, w in zip(params, weights)) and
             all(torch.equal(fresh.state[p]["exp_avg"], m) for p, m in zip([p for p in params if p in fresh.state], moments)))
    checks.append(("resume restores weights and adam", exact and next_round == 7 and visits == {0: 3},
                   f"{len(weights)} tensors, {len(moments)} adam moments, next round {next_round}"))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name:34s} {detail}", flush=True)
    print("one optimizer step from the same start, same batch, synthetic advantage:", flush=True)
    for lr in (5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7):
        for param, saved in zip(params, before):
            param.data.copy_(saved)
        sweep = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        update(policy, sweep, params, data, noise, loss_config, 1, dims, generator)
        kl, worst = step_kl(policy, data, noise, dims)
        print(f"  lr {lr:.0e}: approx kl {kl:.4f}, max |log ratio| {worst:.3f}", flush=True)
    for param, saved in zip(params, before):
        param.data.copy_(saved)
    return all(passed for _, passed, _ in checks)


def step_kl(policy, data: list[Batch], noise: NoiseConfig, dims: tuple[int, int]) -> tuple[float, float]:
    kls, worst = [], 0.0
    with torch.no_grad():
        for batch in data:
            batch = batch.to("cuda")
            log_ratio = (recompute(policy, batch, noise, *dims) - batch.old_logprob)[batch.live]
            kls.append(((log_ratio.exp() - 1) - log_ratio))
            worst = max(worst, log_ratio.abs().max().item())
    return torch.cat(kls).mean().item(), worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="a1_sft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--train-states", default="20:50")
    parser.add_argument("--baseline", choices=["state", "group"], default="state")
    parser.add_argument("--baseline-window", type=int, default=5)
    parser.add_argument("--group", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--rounds-per-update", type=int, default=1)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--noise-joint", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--run-id")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()

    started = time.perf_counter()
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    suite = json.loads((ROOT / base["subsets"]).read_text())["suite"]
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    run_id = args.run_id or f"{args.condition}_s{args.seed}_rl"
    tasks = [int(task) for task in args.tasks.split(",")]
    low, high = (int(bound) for bound in args.train_states.split(":"))
    pool = list(range(low, high))
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    checkpoint = args.checkpoint = args.checkpoint or default_checkpoint(args.condition, args.seed)
    policy, preprocessor = load_policy(checkpoint)
    policy.config.n_action_steps = base["policy"]["n_action_steps"]
    _, postprocessor = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    params = trainable(policy)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    noise = NoiseConfig(noise_level=args.noise_level, joint=args.noise_joint)
    loss_config = LossConfig(clip=args.clip)
    dims = (policy.config.n_action_steps, policy.config.action_feature.shape[0])
    print(f"{run_id}: checkpoint {checkpoint}\n{sum(p.numel() for p in params) / 1e6:.1f}M trainable fp32 "
          f"parameters, {noise}, {loss_config}, logprob over {dims[0]} executed steps x {dims[1]} action dims\n"
          f"tasks {tasks}, train states {low} to {high - 1}, {args.baseline} baseline"
          f"{f' over the last {args.baseline_window} visits' if args.baseline == 'state' else ''}", flush=True)

    if args.probe:
        env = TaskEnv(suite, tasks[0], 2, policy.config, episode_length=20)
        try:
            passed = probe(policy, optimizer, params, env, preprocessor, postprocessor, noise, loss_config, dims,
                           generator, [pool[0]] * env.group)
        finally:
            env.close()
        print(f"probe {'PASSED' if passed else 'FAILED'} in {time.perf_counter() - started:.1f} s")
        sys.exit(0 if passed else 1)

    steps_csv, episodes_csv = ROOT / "logs/rl_steps.csv", ROOT / "logs/rl_episodes.csv"
    check_header(steps_csv, STEP_COLUMNS)
    check_header(episodes_csv, EPISODE_COLUMNS)
    updates_csv = ROOT / "logs/rl_updates.csv"
    check_header(updates_csv, UPDATE_COLUMNS)
    per_update = args.rounds_per_update
    pending_chunks, pending_stats = 0, {}
    optimizer.zero_grad(set_to_none=True)
    baseline = StateBaseline(args.baseline_window)
    visits = {task: 0 for task in tasks}
    start = 0
    if args.resume:
        start = restore(torch.load(Path(args.resume) / "trainer.pt", weights_only=False), params, optimizer, baseline,
                        visits, generator, args)
        print(f"resumed from {args.resume} at round {start}; rows logged after it by the interrupted run stay in "
              f"the csvs, keep the last per round", flush=True)
    env = None
    try:
        for round_index in range(start, args.rounds):
            tick = time.perf_counter()
            task = tasks[round_index % len(tasks)]
            if env is None or env.task_id != task:
                if env is not None:
                    env.close()
                env = TaskEnv(suite, task, args.group, policy.config)
            states = [pool[(visits[task] * args.group + i) % len(pool)] for i in range(args.group)]
            visits[task] += 1
            env.start_at(states)
            seeds = [args.seed * 1_000_000 + round_index * args.group + i for i in range(args.group)]
            episodes = collect(policy, preprocessor, postprocessor, env, noise, seeds)
            collected = time.perf_counter() - tick
            if args.baseline == "state":
                advantage, reference = baseline.advantages(task, states, episodes.success)
            else:
                advantage = group_advantages(episodes.success)
                reference = episodes.success - advantage
            append(episodes_csv, EPISODE_COLUMNS, [
                {"run_id": run_id, "round": round_index, "task_id": task, "init_state": states[i], "episode_idx": i,
                 "seed": seeds[i], "success": int(episodes.success[i]), "baseline": float(reference[i]),
                 "advantage": float(advantage[i]), "episode_length": int(episodes.length[i]), "git_sha": git_sha}
                for i in range(args.group)])
            data = to_batches(episodes, advantage, lambda lp: chunk_logprob(lp, *dims))
            outcome = (f"[{time.strftime('%H:%M:%S')}] {run_id} round {round_index:3d} task {task}: success "
                       f"{int(episodes.success.sum())}/{args.group}, |A| {advantage.abs().mean():.2f}, "
                       f"{sum(len(b) for b in data)} chunks, collect {collected:.0f} s")
            if per_update == 1:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * min(1.0, (round_index + 1) / max(args.warmup_updates, 1))
                rows = update(policy, optimizer, params, data, noise, loss_config, args.epochs, dims, generator)
                for i, row in enumerate(rows):
                    row.update(run_id=run_id, round=round_index, update=round_index * len(rows) + i, task_id=task,
                               git_sha=git_sha)
                append(steps_csv, STEP_COLUMNS, rows)
                print(f"{outcome}, update {time.perf_counter() - tick - collected:.0f} s, last epoch clipfrac "
                      f"{rows[-1]['ratio_clipfrac']:.2f}, approx kl {rows[-1]['approx_kl_old']:.4f}, grad norm "
                      f"{rows[-1]['grad_norm']:.2f}, peak {max(r['peak_vram_gb'] for r in rows):.2f} GiB", flush=True)
            else:
                chunks, stats = accumulate(policy, data, noise, loss_config, dims)
                pending_chunks += chunks
                for key, value in stats.items():
                    pending_stats[key] = pending_stats.get(key, 0.0) + value
                print(f"{outcome}, gradient {time.perf_counter() - tick - collected:.0f} s, "
                      f"{pending_chunks} chunks pending", flush=True)
                if (round_index + 1) % per_update == 0 or round_index + 1 == args.rounds:
                    update_index = round_index // per_update
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr * min(1.0, (update_index + 1) / max(args.warmup_updates, 1))
                    row = apply_update(policy, optimizer, params, pending_chunks, pending_stats, data, noise, dims)
                    row.update(run_id=run_id, update=update_index, last_round=round_index, chunks=pending_chunks,
                               lr=optimizer.param_groups[0]["lr"], denoise_steps=noise.num_steps,
                               noise_level=noise.noise_level, noise_joint=noise.joint, git_sha=git_sha)
                    append(updates_csv, UPDATE_COLUMNS, [row])
                    print(f"[{time.strftime('%H:%M:%S')}] {run_id} update {update_index} over {pending_chunks} "
                          f"chunks: lr {row['lr']:.1e}, grad norm {row['grad_norm']:.2f}, step kl "
                          f"{row['approx_kl_step']:.4f}, max |log ratio| {row['max_log_ratio_step']:.3f}", flush=True)
                    pending_chunks, pending_stats = 0, {}
            if (round_index + 1) % args.save_every == 0 or round_index + 1 == args.rounds:
                target = ROOT / f"outputs/rl/{run_id}/round_{round_index + 1:03d}"
                policy.save_pretrained(target)
                for source in Path(checkpoint).glob("policy_*"):
                    shutil.copy2(source, target / source.name)
                if pending_chunks == 0:
                    torch.save(trainer_state(params, optimizer, baseline, visits, generator, round_index + 1, args),
                               target / "trainer.tmp")
                    (target / "trainer.tmp").rename(target / "trainer.pt")
                    for old in (ROOT / f"outputs/rl/{run_id}").glob("round_*/trainer.pt"):
                        if old.parent != target:
                            old.unlink()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
