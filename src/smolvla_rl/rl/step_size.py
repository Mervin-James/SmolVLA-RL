import argparse
import time

import torch
import yaml
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .likelihood import NoiseConfig, chain_logprob, encode, sample
from .loss import chunk_logprob
from .noisy_eval import keep_stored_dtype
from .train import ROOT, default_checkpoint, trainable
from .verify import load_batch, load_policy

ODE = NoiseConfig(method="flow_ode")


def executed_actions(policy, batch: dict, noise: torch.Tensor, dims: tuple[int, int]) -> torch.Tensor:
    with torch.no_grad():
        return sample(policy, batch, ODE, noise=noise).actions[:, :dims[0], :dims[1]].float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="a1_sft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--lrs", default="1e-7,2.5e-7,5e-7,1e-6")
    parser.add_argument("--noise-level", type=float, default=1.5)
    parser.add_argument("--references", nargs="*", default=[])
    args = parser.parse_args()

    started = time.perf_counter()
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    dims = (base["policy"]["n_action_steps"], 7)
    keep_stored_dtype()
    policy, preprocessor = load_policy(default_checkpoint(args.condition, args.seed))
    params = trainable(policy)
    trained = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    batch = load_batch(preprocessor, args.size)
    generator = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.randn((args.size, policy.config.chunk_size, policy.config.max_action_dim), device="cuda",
                        generator=generator)
    reference = executed_actions(policy, batch, noise, dims)
    scale = reference.abs().mean().item()
    print(f"SFT {args.condition} seed {args.seed}, fp32 expert: mean |action| {scale:.4f} over {args.size} dataset "
          f"frames, executed {dims[0]} steps x {dims[1]} dims, ODE with fixed noise", flush=True)

    for path in args.references:
        other = SmolVLAPolicy.from_pretrained(path).to("cuda").eval()
        shift = (executed_actions(other, batch, noise, dims) - reference).abs().mean().item()
        theirs = dict(other.named_parameters())
        moved = max((theirs[name].detach().float() - mine.detach()).abs().max().item() for name, mine in trained)
        print(f"reference {path}: mean |action shift| {shift:.5f} ({100 * shift / scale:.2f}% of mean |action|), "
              f"max |delta theta| {moved:.2e}", flush=True)
        del other
        torch.cuda.empty_cache()

    lrs = [float(lr) for lr in args.lrs.split(",") if lr]
    if lrs:
        config = NoiseConfig(noise_level=args.noise_level, joint=True)
        chain = sample(policy, batch, config, generator=generator)
        old = chunk_logprob(chain.logprob, *dims)
        advantage = torch.tensor([1.0, -1.0] * (args.size // 2), device="cuda")
        prefix = encode(policy, batch)
        before = [p.detach().clone() for p in params]
        for lr in lrs:
            for param, saved in zip(params, before):
                param.data.copy_(saved)
                param.grad = None
            logprob = chunk_logprob(chain_logprob(policy, batch, chain, config, prefix=prefix), *dims)
            (-(advantage * logprob).mean()).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            torch.optim.AdamW(params, lr=lr, weight_decay=0.0).step()
            moved = max((p.detach() - b).abs().max().item() for p, b in zip(params, before))
            shift = (executed_actions(policy, batch, noise, dims) - reference).abs().mean().item()
            with torch.no_grad():
                log_ratio = chunk_logprob(chain_logprob(policy, batch, chain, config, prefix=prefix), *dims) - old
            kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
            print(f"one first Adam step at lr {lr:.1e}: max |delta theta| {moved:.2e}, mean |action shift| "
                  f"{shift:.5f} ({100 * shift / scale:.2f}%), chain kl {kl:.4f}", flush=True)
        for param, saved in zip(params, before):
            param.data.copy_(saved)
    print(f"done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
