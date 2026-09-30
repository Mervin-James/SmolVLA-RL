from dataclasses import dataclass

import torch


@dataclass
class LossConfig:
    clip: float = 0.2
    kl_coef: float = 0.0
    normalize_advantage: bool = False
    value_coef: float = 0.5
    value_clip: float | None = 0.2


def group_advantages(rewards: torch.Tensor, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
    advantages = rewards - rewards.mean(dim=-1, keepdim=True)
    if normalize:
        advantages = advantages / (rewards.std(dim=-1, keepdim=True) + eps)
    return advantages


class StateBaseline:
    def __init__(self, window: int):
        self.window = window
        self.history: dict[tuple[int, int], list[float]] = {}

    def advantages(self, task_id: int, states: list[int],
                   rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fallback = rewards.mean().item()
        seen = [self.history.get((task_id, state)) for state in states]
        baseline = torch.tensor([sum(h) / len(h) if h else fallback for h in seen], dtype=rewards.dtype)
        for state, reward in zip(states, rewards.tolist()):
            self.history[(task_id, state)] = (self.history.get((task_id, state), []) + [reward])[-self.window:]
        return rewards - baseline, baseline


def chunk_logprob(logprob: torch.Tensor, n_action_steps: int | None, action_dim: int | None) -> torch.Tensor:
    return logprob[:, :n_action_steps, :action_dim].sum(dim=(1, 2))


def policy_loss(logprob: torch.Tensor, old_logprob: torch.Tensor, advantage: torch.Tensor,
                clip: float) -> tuple[torch.Tensor, dict[str, float]]:
    log_ratio = logprob - old_logprob
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    stats = {
        "ratio_mean": ratio.mean().item(),
        "ratio_clipfrac": ((ratio - 1).abs() > clip).float().mean().item(),
        "approx_kl_old": ((ratio - 1) - log_ratio).mean().item(),
    }
    return loss, stats


def kl_to_reference(logprob: torch.Tensor, reference_logprob: torch.Tensor) -> torch.Tensor:
    log_ratio = reference_logprob - logprob
    return (log_ratio.exp() - log_ratio - 1).mean()


def value_loss(value: torch.Tensor, old_value: torch.Tensor, target: torch.Tensor,
               clip: float | None) -> torch.Tensor:
    loss = (value - target) ** 2
    if clip is not None:
        clipped = old_value + (value - old_value).clamp(-clip, clip)
        loss = torch.maximum(loss, (clipped - target) ** 2)
    return 0.5 * loss.mean()


def total_loss(logprob: torch.Tensor, old_logprob: torch.Tensor, advantage: torch.Tensor, config: LossConfig,
               reference_logprob: torch.Tensor | None = None, value: torch.Tensor | None = None,
               old_value: torch.Tensor | None = None,
               value_target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    loss, stats = policy_loss(logprob, old_logprob, advantage, config.clip)
    stats["loss_policy"] = loss.item()
    if reference_logprob is not None and config.kl_coef > 0:
        kl = kl_to_reference(logprob, reference_logprob)
        loss = loss + config.kl_coef * kl
        stats["kl_to_ref"] = kl.item()
    if value is not None:
        critic = value_loss(value, old_value, value_target, config.value_clip)
        loss = loss + config.value_coef * critic
        stats["loss_value"] = critic.item()
    stats["loss_total"] = loss.item()
    stats["advantage_mean"] = advantage.mean().item()
    stats["advantage_std"] = advantage.std().item() if advantage.numel() > 1 else 0.0
    return loss, stats
