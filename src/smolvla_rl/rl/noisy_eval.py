import argparse
import sys

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from safetensors.torch import load_file
from lerobot.scripts.lerobot_eval import main as lerobot_eval

from .likelihood import NoiseConfig, encode_inputs, sample_chain


def install(config: NoiseConfig) -> None:
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        prefix = encode_inputs(self, images, img_masks, lang_tokens, lang_masks, state)
        return sample_chain(self, prefix, config, noise=noise).actions

    VLAFlowMatching.sample_actions = sample_actions


def keep_stored_dtype() -> None:
    original = SmolVLAPolicy._load_as_safetensor.__func__

    def load(cls, model, model_file, map_location, strict):
        stored = {key: value.dtype for key, value in load_file(model_file).items()}
        upcast = 0
        for name, tensor in model.state_dict(keep_vars=True).items():
            if stored.get(name) == torch.float32 and tensor.dtype != torch.float32:
                tensor.data = tensor.data.float()
                upcast += 1
        print(f"upcast {upcast} tensors to their stored fp32 before loading {model_file}", flush=True)
        return original(cls, model, model_file, map_location, strict)

    SmolVLAPolicy._load_as_safetensor = classmethod(load)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-method")
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--noise-joint", action="store_true")
    parser.add_argument("--noise-steps", type=int, default=10)
    parser.add_argument("--stored-dtype", action="store_true")
    args, rest = parser.parse_known_args()
    if args.stored_dtype:
        keep_stored_dtype()
    if args.noise_method:
        config = NoiseConfig(num_steps=args.noise_steps, method=args.noise_method,
                             noise_level=args.noise_level, joint=args.noise_joint)
        print(f"noisy sampler installed: {config}", flush=True)
        install(config)
    sys.argv = [sys.argv[0], *rest]
    lerobot_eval()


if __name__ == "__main__":
    main()
