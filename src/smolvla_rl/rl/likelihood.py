from dataclasses import dataclass

import torch
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .noise import gaussian_logprob, timesteps, transition


@dataclass
class Prefix:
    pad_masks: torch.Tensor
    cache: object


@dataclass
class Chain:
    states: torch.Tensor
    noisy_step: torch.Tensor
    logprob: torch.Tensor

    @property
    def actions(self) -> torch.Tensor:
        return self.states[:, -1]


@dataclass
class NoiseConfig:
    num_steps: int = 10
    method: str = "flow_sde"
    noise_level: float = 0.5
    joint: bool = False


@torch.no_grad()
def embed_inputs(model, images, image_masks, lang_tokens, lang_masks, state):
    return model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)


@torch.no_grad()
def prefill(model, embeddings: torch.Tensor, pad_masks: torch.Tensor, att_masks: torch.Tensor) -> Prefix:
    _, cache = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad_masks, att_masks),
        position_ids=torch.cumsum(pad_masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=True,
    )
    return Prefix(pad_masks, cache)


def encode_inputs(model, images, image_masks, lang_tokens, lang_masks, state) -> Prefix:
    return prefill(model, *embed_inputs(model, images, image_masks, lang_tokens, lang_masks, state))


def encode(policy: SmolVLAPolicy, batch: dict) -> Prefix:
    images, image_masks = policy.prepare_images(batch)
    return encode_inputs(policy.model, images, image_masks, batch[OBS_LANGUAGE_TOKENS],
                         batch[OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_state(batch))


def velocity(policy: SmolVLAPolicy, prefix: Prefix, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return model_velocity(policy.model, prefix, x_t, t)


def model_velocity(model, prefix: Prefix, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return model.denoise_step(
        prefix_pad_masks=prefix.pad_masks, past_key_values=prefix.cache, x_t=x_t, timestep=t).float()


def step_distribution(x_t: torch.Tensor, v_t: torch.Tensor, step: torch.Tensor, noisy: torch.Tensor,
                      config: NoiseConfig) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = transition(x_t, v_t, step, config.num_steps, config.method, config.noise_level)
    ode_mean, _ = transition(x_t, v_t, step, config.num_steps, "flow_ode", config.noise_level)
    noisy = noisy.view(-1, 1, 1)
    return torch.where(noisy, mean, ode_mean), torch.where(noisy, std, torch.zeros_like(std))


@torch.no_grad()
def sample_chain(model, prefix: Prefix, config: NoiseConfig, noise: torch.Tensor | None = None,
                 generator: torch.Generator | None = None) -> Chain:
    size = prefix.pad_masks.shape[0]
    device = prefix.pad_masks.device
    shape = (size, model.config.chunk_size, model.config.max_action_dim)
    x_t = torch.randn(shape, device=device, generator=generator) if noise is None else noise.float()
    noisy_step = (torch.full((size,), -1, device=device, dtype=torch.long) if config.joint
                  else torch.randint(0, config.num_steps, (size,), device=device, generator=generator))
    grid = timesteps(config.num_steps, device)
    states, logprob = [x_t], torch.zeros_like(x_t)
    for k in range(config.num_steps):
        step = torch.full((size,), k, device=device, dtype=torch.long)
        v_t = model_velocity(model, prefix, x_t, grid[step])
        noisy = torch.full((size,), True, device=device) if config.joint else noisy_step == k
        mean, std = step_distribution(x_t, v_t, step, noisy, config)
        x_t = mean + std * torch.randn(shape, device=device, generator=generator)
        logprob = logprob + gaussian_logprob(x_t, mean, std)
        states.append(x_t)
    return Chain(torch.stack(states, dim=1), noisy_step, logprob)


def sample(policy: SmolVLAPolicy, batch: dict, config: NoiseConfig, noise: torch.Tensor | None = None,
           generator: torch.Generator | None = None) -> Chain:
    return sample_chain(policy.model, encode(policy, batch), config, noise, generator)


def chain_logprob(policy: SmolVLAPolicy, batch: dict, chain: Chain, config: NoiseConfig,
                  prefix: Prefix | None = None) -> torch.Tensor:
    prefix = encode(policy, batch) if prefix is None else prefix
    size = chain.states.shape[0]
    device = chain.states.device
    grid = timesteps(config.num_steps, device)
    rows = torch.arange(size, device=device)
    steps = range(config.num_steps) if config.joint else [None]
    logprob = torch.zeros_like(chain.states[:, 0])
    for k in steps:
        step = chain.noisy_step if k is None else torch.full((size,), k, device=device, dtype=torch.long)
        x_t, x_next = chain.states[rows, step], chain.states[rows, step + 1]
        v_t = velocity(policy, prefix, x_t, grid[step])
        mean, std = step_distribution(x_t, v_t, step, torch.ones(size, dtype=torch.bool, device=device), config)
        logprob = logprob + gaussian_logprob(x_next, mean, std)
    return logprob
