from dataclasses import dataclass

import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.scripts.lerobot_eval import rollout

from .likelihood import Chain, NoiseConfig, embed_inputs, prefill, sample_chain


@dataclass
class Call:
    embeddings: torch.Tensor
    pad_masks: torch.Tensor
    att_masks: torch.Tensor
    states: torch.Tensor
    noisy_step: torch.Tensor
    logprob: torch.Tensor


@dataclass
class Batch:
    embeddings: torch.Tensor
    pad_masks: torch.Tensor
    att_masks: torch.Tensor
    states: torch.Tensor
    noisy_step: torch.Tensor
    old_logprob: torch.Tensor
    advantage: torch.Tensor
    live: torch.Tensor

    def __len__(self) -> int:
        return int(self.live.sum())

    def to(self, device: str) -> "Batch":
        return Batch(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))

    def chain(self) -> Chain:
        return Chain(self.states, self.noisy_step, self.old_logprob)


class Recorder:
    def __init__(self, config: NoiseConfig):
        self.config = config
        self.calls: list[Call] = []
        self.original = None

    def __enter__(self) -> "Recorder":
        self.original = VLAFlowMatching.sample_actions
        recorder = self

        def sample_actions(model, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
            embedded = embed_inputs(model, images, img_masks, lang_tokens, lang_masks, state)
            chain = sample_chain(model, prefill(model, *embedded), recorder.config, noise=noise)
            recorder.calls.append(Call(*(t.cpu() for t in embedded), chain.states.cpu(),
                                       chain.noisy_step.cpu(), chain.logprob.cpu()))
            return chain.actions

        VLAFlowMatching.sample_actions = sample_actions
        return self

    def __exit__(self, *exc) -> None:
        VLAFlowMatching.sample_actions = self.original


class TaskEnv:
    def __init__(self, suite: str, task_id: int, group: int, policy_config, episode_length: int | None = None):
        config = LiberoEnv(task=suite, task_ids=[task_id], episode_length=episode_length)
        self.env = make_env(config, n_envs=group, use_async_envs=group > 1)[suite][task_id]
        self.pre, self.post = make_env_pre_post_processors(env_cfg=config, policy_cfg=policy_config)
        self.task_id = task_id
        self.group = group

    def _vector(self):
        if hasattr(self.env, "_ensure"):
            self.env._ensure()
        return getattr(self.env, "_env", self.env)

    def start_at(self, states: list[int]) -> None:
        self._vector().set_attr("init_state_id", list(states))
        assert self.state_ids() == list(states), f"init states {self.state_ids()} after setting {states}"

    def state_ids(self) -> list[int]:
        return [int(i) for i in self._vector().get_attr("init_state_id")]

    def close(self) -> None:
        self.env.close()


@dataclass
class Episodes:
    success: torch.Tensor
    length: torch.Tensor
    calls: list[Call]
    n_action_steps: int


def collect(policy, preprocessor, postprocessor, task_env: TaskEnv, config: NoiseConfig,
            seeds: list[int]) -> Episodes:
    policy.reset()
    with Recorder(config) as recorder, torch.no_grad():
        data = rollout(task_env.env, policy, task_env.pre, task_env.post, preprocessor, postprocessor, seeds=seeds)
    done, success = data["done"], data["success"]
    end = done.int().argmax(dim=1)
    reached = torch.stack([success[b, :end[b] + 1].any() for b in range(done.shape[0])])
    return Episodes(reached.float(), end + 1, [_clone(c) for c in recorder.calls], policy.config.n_action_steps)


def _clone(call: Call) -> Call:
    return Call(*(getattr(call, f).clone() for f in call.__dataclass_fields__))


def to_batches(episodes: Episodes, advantage: torch.Tensor, reduce) -> list[Batch]:
    batches = []
    for index, call in enumerate(episodes.calls):
        live = index * episodes.n_action_steps < episodes.length
        if live.any():
            batches.append(Batch(call.embeddings, call.pad_masks, call.att_masks, call.states, call.noisy_step,
                                 reduce(call.logprob), advantage.clone(), live))
    return batches
