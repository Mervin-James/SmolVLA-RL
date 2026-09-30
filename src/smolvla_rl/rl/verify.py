import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .likelihood import Chain, NoiseConfig, chain_logprob, encode, sample, step_distribution, velocity
from .noisy_eval import install
from .noise import gaussian_entropy, gaussian_logprob, timesteps

ROOT = Path(__file__).resolve().parents[3]
Z_LIMIT = 5.5


@dataclass
class Result:
    name: str
    passed: bool
    detail: str


def default_checkpoint() -> str:
    runs = sorted(ROOT.glob("outputs/cp0/a25_sft_s0/*/checkpoints/008000/pretrained_model"))
    assert runs, "no a25_sft_s0 checkpoint under outputs/cp0"
    return str(runs[-1])


def load_policy(checkpoint: str) -> tuple[SmolVLAPolicy, object]:
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    policy = SmolVLAPolicy.from_pretrained(checkpoint).to("cuda").eval()
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": base["rename_map"]}})
    return policy, preprocessor


def load_batch(preprocessor, size: int) -> dict:
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    subsets = json.loads((ROOT / base["subsets"]).read_text())
    dataset = LeRobotDataset(subsets["dataset"], revision=subsets["dataset_revision"],
                             episodes=subsets["episodes"]["0"]["25"][:4],
                             video_backend=base["dataset"]["video_backend"])
    stride = max(1, len(dataset) // size)
    rows = [dataset[i * stride] for i in range(size)]
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        batch[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return preprocessor(batch)


def subset(batch: dict, rows: slice) -> dict:
    return {k: (v[rows] if torch.is_tensor(v) or isinstance(v, list) else v) for k, v in batch.items()}


def ode_matches_lerobot(policy, batch, config) -> Result:
    noise = torch.randn(batch[OBS_STATE].shape[0], policy.config.chunk_size,
                        policy.config.max_action_dim, device="cuda")
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)

    def lerobot(x):
        with torch.no_grad():
            return policy.model.sample_actions(images, image_masks, batch[OBS_LANGUAGE_TOKENS],
                                               batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=x.clone())

    theirs = lerobot(noise)
    ours = sample(policy, batch, replace(config, method="flow_ode"), noise=noise).actions
    gap = (ours - theirs).abs().max().item()
    floor = max((lerobot(noise * (1 + 1e-6 * torch.randn_like(noise))) - theirs).abs().max().item()
                for _ in range(3))
    scale = theirs.abs().max().item()
    return Result("1a flow_ode chain == lerobot sample_actions", gap <= max(3 * floor, 1e-4 * scale),
                  f"max abs gap {gap:.2e}; lerobot against itself under 1e-6 input jitter {floor:.2e}; "
                  f"actions of scale {scale:.2f}")


def eval_patch_routes_to_sampler(policy, batch, config) -> Result:
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    original = VLAFlowMatching.sample_actions
    noise = torch.randn(batch[OBS_STATE].shape[0], policy.config.chunk_size,
                        policy.config.max_action_dim, device="cuda")
    images, image_masks = policy.prepare_images(batch)
    args = (images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            policy.prepare_state(batch))
    try:
        install(replace(config, method="flow_ode"))
        with torch.no_grad():
            patched_ode = policy.model.sample_actions(*args, noise=noise.clone())
        install(config)
        with torch.no_grad():
            patched_sde = policy.model.sample_actions(*args, noise=noise.clone())
    finally:
        VLAFlowMatching.sample_actions = original
    direct = sample(policy, batch, replace(config, method="flow_ode"), noise=noise).actions
    same = (patched_ode - direct).abs().max().item()
    moved = (patched_sde - patched_ode).abs().mean().item()
    passed = same == 0 and moved > 0 and patched_sde.shape == direct.shape
    return Result("1c eval patch routes lerobot to our sampler", passed,
                  f"patched flow_ode vs direct {same:.1e}, flow_sde moves actions by {moved:.3f} on average")


def closed_form_vs_monte_carlo(policy, batch, config, draws: int) -> Result:
    one = subset(batch, slice(0, 1))
    chain = sample(policy, one, config)
    step = chain.noisy_step
    x_t = chain.states[0:1, step.item()]
    with torch.no_grad():
        v_t = velocity(policy, encode(policy, one), x_t, timesteps(config.num_steps, "cuda")[step])
    mean, std = step_distribution(x_t, v_t, step, torch.ones(1, dtype=torch.bool, device="cuda"), config)
    draws_t = mean + std * torch.randn((draws, *mean.shape[1:]), device="cuda")
    se = std / draws ** 0.5
    z_mean = ((draws_t.mean(0) - mean[0]) / se[0]).abs().max().item()
    std_ratio = (draws_t.std(0) / std[0]).sub(1).abs().max().item()
    logprob = gaussian_logprob(draws_t, mean, std)
    z_entropy = ((logprob.mean(0) + gaussian_entropy(std[0])) / (0.5 / draws) ** 0.5).abs().max().item()
    reference = torch.distributions.Normal(mean, std).log_prob(draws_t)
    gap = (reference - logprob).abs().max().item()
    passed = z_mean < Z_LIMIT and std_ratio < 0.06 and z_entropy < Z_LIMIT and gap < 1e-4
    return Result("1b closed form vs Monte Carlo", passed,
                  f"step {step.item()}, std {std.mean().item():.3f}, {draws} draws: max|z| mean {z_mean:.2f}, "
                  f"std off {100 * std_ratio:.1f}%, max|z| E[log p]+H {z_entropy:.2f}, vs torch Normal {gap:.1e}")


def ratio_is_one(policy, batch, config) -> Result:
    chain = sample(policy, batch, config)
    worst = 0.0
    for mode in ("eval", "train"):
        getattr(policy, mode)()
        with torch.no_grad():
            recomputed = chain_logprob(policy, batch, chain, config)
        ratio = (recomputed - chain.logprob).sum(dim=(1, 2)).exp()
        worst = max(worst, (ratio - 1).abs().max().item())
    policy.eval()
    return Result("3 ratio == 1 at initialisation", worst < 1e-3,
                  f"max |ratio - 1| {worst:.2e} over {chain.states.shape[0]} chunks, eval and train mode")


def score_mean_zero(policy, batch, config, draws: int) -> Result:
    one = subset(batch, slice(0, 1))
    chain = sample(policy, one, config)
    step = chain.noisy_step
    x_t = chain.states[0:1, step.item()]
    with torch.no_grad():
        v_t = velocity(policy, encode(policy, one), x_t, timesteps(config.num_steps, "cuda")[step])
    v = v_t.expand(draws, -1, -1).clone().requires_grad_(True)
    steps = step.expand(draws)
    noisy = torch.ones(draws, dtype=torch.bool, device="cuda")
    mean, std = step_distribution(x_t.expand(draws, -1, -1), v, steps, noisy, config)
    draws_t = (mean + std * torch.randn_like(mean)).detach()
    gaussian_logprob(draws_t, mean, std).sum().backward()
    score = v.grad
    z = (score.mean(0) / (score.std(0) / draws ** 0.5 + 1e-12)).abs().max().item()
    return Result("2 expected score == 0", z < Z_LIMIT,
                  f"{draws} draws, max|z| of mean d log p / d v over {score[0].numel()} elements {z:.2f}")


def gradient_flow(policy, batch, config) -> Result:
    chain = sample(policy, batch, config)
    policy.train()
    policy.zero_grad(set_to_none=True)
    chain_logprob(policy, batch, chain, config).sum().backward()
    expert = [p for n, p in policy.named_parameters() if "vlm_with_expert.lm_expert." in n and p.requires_grad]
    vlm = [p for n, p in policy.named_parameters() if "vlm_with_expert.vlm." in n]
    expert_live = sum(1 for p in expert if p.grad is not None and p.grad.abs().sum() > 0)
    vlm_live = sum(1 for p in vlm if p.grad is not None and p.grad.abs().sum() > 0)
    projections = [policy.model.action_out_proj.weight.grad, policy.model.action_in_proj.weight.grad]
    projections_live = all(g is not None and g.abs().sum() > 0 for g in projections)
    policy.zero_grad(set_to_none=True)
    policy.eval()
    passed = expert_live > 0 and vlm_live == 0 and projections_live
    return Result("gradient reaches the action expert only", passed,
                  f"expert tensors with gradient {expert_live}/{len(expert)}, vlm {vlm_live}/{len(vlm)}, "
                  f"action projections {'yes' if projections_live else 'no'}")


def memory_profile(policy, pool: dict, sizes: list[int], joint_steps: list[int]) -> list[str]:
    lines = []
    policy.train()
    for joint, steps_list in ((False, [10]), (True, joint_steps)):
        for num_steps in steps_list:
            for size in sizes if not joint else sizes[:1]:
                config = NoiseConfig(num_steps=num_steps, joint=joint)
                batch = subset(pool, slice(0, size))
                chain = sample(policy, batch, config)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                chain_logprob(policy, batch, chain, config).sum().backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                policy.zero_grad(set_to_none=True)
                label = f"joint over {num_steps} steps" if joint else f"one noisy step of {num_steps}"
                lines.append(f"  {label:26s} batch {size:3d}: likelihood + backward {elapsed:5.2f} s, "
                             f"peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")
    policy.eval()
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--memory-sizes", default="4,8,16")
    parser.add_argument("--joint-steps", default="4,10")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint or default_checkpoint()
    policy, preprocessor = load_policy(checkpoint)
    sizes = [int(s) for s in args.memory_sizes.split(",") if s]
    joint_steps = [int(s) for s in args.joint_steps.split(",") if s]
    pool = load_batch(preprocessor, max([args.batch, *sizes]))
    batch = subset(pool, slice(0, args.batch))
    config = NoiseConfig(noise_level=args.noise_level)
    print(f"checkpoint {checkpoint}\nloaded in {time.perf_counter() - started:.1f} s, {config}", flush=True)

    checks = [
        lambda: ode_matches_lerobot(policy, batch, config),
        lambda: eval_patch_routes_to_sampler(policy, batch, config),
        lambda: closed_form_vs_monte_carlo(policy, batch, config, args.draws),
        lambda: score_mean_zero(policy, batch, config, args.draws),
        lambda: ratio_is_one(policy, batch, config),
        lambda: gradient_flow(policy, batch, config),
    ]
    results = []
    for check in checks:
        tick = time.perf_counter()
        result = check()
        results.append(result)
        print(f"{'PASS' if result.passed else 'FAIL'}  {result.name:44s} {result.detail}  "
              f"[{time.perf_counter() - tick:.1f} s]", flush=True)

    if sizes:
        print("memory, trainable expert, fp32 likelihood:", flush=True)
        for line in memory_profile(policy, pool, sizes, joint_steps):
            print(line, flush=True)
    print(f"total {time.perf_counter() - started:.1f} s")
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
