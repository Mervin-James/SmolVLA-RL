import sys

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file

from smolvla_rl.rl.noisy_eval import keep_stored_dtype


def differing(path: str) -> str:
    stored = load_file(f"{path}/model.safetensors")
    state = SmolVLAPolicy.from_pretrained(path).state_dict()
    bad = [k for k, v in stored.items() if k in state and (state[k].dtype != v.dtype or not torch.equal(state[k].cpu(), v))]
    return f"{len(bad)} of {len(stored)} tensors differ from the file"


if __name__ == "__main__":
    path, mode = sys.argv[1], sys.argv[2]
    if mode == "stored":
        keep_stored_dtype()
    print(f"{mode} load of {path}: {differing(path)}", flush=True)
