import argparse
import json
import resource
import sys
import time

import torch
import yaml
from lerobot.policies.factory import make_pre_post_processors

from .likelihood import NoiseConfig
from .loss import chunk_logprob
from .rollout import Batch, TaskEnv, collect, to_batches
from .train import ROOT, default_checkpoint, recompute, trainable
from .verify import load_policy


def episode_scores(policy, params, data: list[Batch], noise: NoiseConfig, dims: tuple[int, int],
                   index: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.zeros(group, index.numel(), device="cuda")
    chunks = torch.zeros(group)
    for batch in data:
        batch = batch.to("cuda")
        logprob = recompute(policy, batch, noise, *dims)
        live = batch.live.nonzero().flatten().tolist()
        for position, episode in enumerate(live):
            grads = torch.autograd.grad(logprob[episode], params, retain_graph=position < len(live) - 1,
                                        allow_unused=True)
            flat = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(grads, params)])
            scores[episode] += flat[index].float()
            chunks[episode] += 1
    return scores.cpu(), chunks


def summarize(grads: torch.Tensor) -> dict:
    rounds = grads.shape[0]
    mean = grads.mean(dim=0)
    spread = ((grads - mean) ** 2).sum(dim=1).mean().item() * rounds / (rounds - 1)
    signal = mean.pow(2).sum().item() - spread / rounds
    unit = grads / grads.norm(dim=1, keepdim=True)
    cosines = (unit @ unit.T)[~torch.eye(rounds, dtype=torch.bool)]
    return {"cosine_mean": cosines.mean().item(), "cosine_min": cosines.min().item(),
            "cosine_max": cosines.max().item(), "trace_noise": spread, "signal_sq": signal,
            "noise_scale_rounds": spread / signal if signal > 0 else float("inf")}


def advantages(success: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:
    flat_success, flat_states = success.flatten(), states.flatten()
    loo = torch.empty_like(flat_success)
    group_mean = success.mean(dim=1, keepdim=True).expand_as(success).flatten()
    for i in range(flat_success.numel()):
        others = (flat_states == flat_states[i]) & (torch.arange(flat_success.numel()) != i)
        loo[i] = flat_success[others].mean() if others.any() else group_mean[i]
    return {"none": success, "group": success - success.mean(dim=1, keepdim=True),
            "state_leave_one_out": success - loo.view_as(success)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="a1_sft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--task", type=int, default=4)
    parser.add_argument("--group", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--coords", type=int, default=2_000_000)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--noise-joint", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        args.rounds = 1

    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    suite = json.loads((ROOT / base["subsets"]).read_text())["suite"]
    checkpoint = args.checkpoint or default_checkpoint(args.condition, args.seed)
    policy, preprocessor = load_policy(checkpoint)
    policy.config.n_action_steps = base["policy"]["n_action_steps"]
    _, postprocessor = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    params = trainable(policy)
    noise = NoiseConfig(noise_level=args.noise_level, joint=args.noise_joint)
    dims = (policy.config.n_action_steps, policy.config.action_feature.shape[0])
    size = sum(p.numel() for p in params)
    index = torch.randint(0, size, (args.coords,), generator=torch.Generator().manual_seed(0)).cuda()
    name = f"grad_noise_{args.condition}_s{args.seed}_task{args.task}_noise{args.noise_level}{'_joint' if args.noise_joint else ''}{'_probe' if args.probe else ''}"
    store = ROOT / "outputs/rl" / name
    store.mkdir(parents=True, exist_ok=True)
    success, states, chunks = [], [], []
    env = TaskEnv(suite, args.task, args.group, policy.config, episode_length=20 if args.probe else None)
    started = time.perf_counter()
    try:
        for round_index in range(args.rounds):
            tick = time.perf_counter()
            state = [(round_index * args.group + i) % 50 for i in range(args.group)]
            env.start_at(state)
            seeds = [args.seed * 1_000_000 + round_index * args.group + i for i in range(args.group)]
            episodes = collect(policy, preprocessor, postprocessor, env, noise, seeds)
            data = to_batches(episodes, torch.zeros(args.group), lambda lp: chunk_logprob(lp, *dims))
            torch.cuda.reset_peak_memory_stats()
            scored = time.perf_counter()
            scores, counted = episode_scores(policy, params, data, noise, dims, index, args.group)
            scored = time.perf_counter() - scored
            torch.save(scores, store / f"round_{round_index:03d}.pt")
            success.append(episodes.success)
            states.append(torch.tensor(state))
            chunks.append(counted)
            print(f"round {round_index}: success {int(episodes.success.sum())}/{args.group}, "
                  f"{int(counted.sum())} chunks, {time.perf_counter() - tick:.0f} s", flush=True)
    finally:
        env.close()
    if args.probe:
        per_pass = scored / counted.sum().item()
        peak = torch.cuda.max_memory_allocated() / 2**30
        host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        norms = scores.norm(dim=1)
        checks = [("scores finite", bool(torch.isfinite(scores).all()), ""),
                  ("every episode has a nonzero score", bool((norms > 0).all()),
                   f"norms {[round(n, 2) for n in norms.tolist()]}"),
                  ("peak VRAM under 13 GiB", peak < 13, f"{peak:.2f} GiB of about 15.3 usable"),
                  ("host memory under 8 GiB", host < 8, f"{host:.2f} GiB max resident")]
        for label, passed, detail in checks:
            print(f"{'PASS' if passed else 'FAIL'}  {label:36s} {detail}", flush=True)
        minutes = 20 * (90 + 175 * per_pass) / 60
        print(f"{noise}: {per_pass:.2f} s per episode score pass over {int(counted.sum())} passes; a full round of "
              f"about 175 passes plus about 90 s of rollout projects to {minutes:.0f} min for 20 rounds", flush=True)
        passed = all(p for _, p, _ in checks)
        print(f"probe {'PASSED' if passed else 'FAILED'} in {time.perf_counter() - started:.0f} s after model load")
        sys.exit(0 if passed else 1)

    success, states, chunks = torch.stack(success), torch.stack(states), torch.stack(chunks)
    scores = torch.stack([torch.load(store / f"round_{r:03d}.pt") for r in range(args.rounds)])
    visits = {}
    for s, r in zip(states.flatten().tolist(), success.flatten().tolist()):
        visits.setdefault(s, []).append(r)
    fixed = sum(1 for v in visits.values() if len(set(v)) == 1)
    result = {"checkpoint": checkpoint, "task": args.task, "rounds": args.rounds, "group": args.group,
              "noise_level": args.noise_level, "noise_joint": args.noise_joint, "coords": args.coords, "params": size,
              "success_rate": success.mean().item(), "states_all_same_outcome": fixed, "states": len(visits)}
    print(f"{args.coords / 1e6:.0f}M of {size / 1e6:.1f}M coordinates, success {success.mean():.3f}, "
          f"{fixed} of {len(visits)} states had the same outcome on every visit", flush=True)
    for label, advantage in advantages(success, states).items():
        weights = -advantage / chunks.sum(dim=1, keepdim=True)
        stats = summarize(torch.einsum("rg,rgc->rc", weights, scores))
        result[label] = stats
        print(f"{label:20s} mean cosine {stats['cosine_mean']:+.3f} (min {stats['cosine_min']:+.3f}, max "
              f"{stats['cosine_max']:+.3f}), noise scale {stats['noise_scale_rounds']:.1f} rounds, "
              f"signal^2 {stats['signal_sq']:.3e}, noise trace {stats['trace_noise']:.3e}", flush=True)
    (ROOT / "outputs/rl" / f"{name}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
