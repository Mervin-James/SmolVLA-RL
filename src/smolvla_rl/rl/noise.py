import math

import torch

METHODS = ("flow_ode", "flow_sde")


def timesteps(num_steps: int, device: torch.device | str) -> torch.Tensor:
    dt = -1.0 / num_steps
    return torch.tensor([1.0 + k * dt for k in range(num_steps)] + [0.0], dtype=torch.float32, device=device)


def transition(x_t: torch.Tensor, v_t: torch.Tensor, step: torch.Tensor, num_steps: int,
               method: str, noise_level: float) -> tuple[torch.Tensor, torch.Tensor]:
    grid = timesteps(num_steps, x_t.device).to(x_t.dtype)
    t = grid[step].view(-1, 1, 1)
    delta = (grid[step] - grid[step + 1]).view(-1, 1, 1)
    data_pred = x_t - v_t * t
    noise_pred = x_t + v_t * (1 - t)
    data_weight = 1 - (t - delta)
    if method == "flow_ode":
        return data_pred * data_weight + noise_pred * (t - delta), torch.zeros_like(x_t)
    if method == "flow_sde":
        ratio = grid / (1 - torch.where(grid == 1, grid[1], grid))
        sigma = (noise_level * ratio.sqrt())[step].view(-1, 1, 1)
        mean = data_pred * data_weight + noise_pred * ((t - delta) - sigma * sigma * delta / (2 * t))
        return mean, (delta.sqrt() * sigma).expand_as(x_t)
    raise ValueError(f"unknown noise method {method!r}, expected one of {METHODS}")


def gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    deterministic = std == 0
    safe = torch.where(deterministic, torch.ones_like(std), std)
    logprob = -safe.log() - 0.5 * math.log(2 * math.pi) - 0.5 * ((sample - mean) / safe) ** 2
    return torch.where(deterministic, torch.zeros_like(logprob), logprob)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    deterministic = std == 0
    safe = torch.where(deterministic, torch.ones_like(std), std)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * safe ** 2)
    return torch.where(deterministic, torch.zeros_like(entropy), entropy)
