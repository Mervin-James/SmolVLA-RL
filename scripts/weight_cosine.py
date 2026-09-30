import argparse
import glob
import itertools

import torch
from safetensors.torch import load_file

EXPERT = ("lm_expert", "action_", "state_proj", "time_mlp")


def delta(sft_path: str, rl_path: str) -> torch.Tensor:
    sft, rl = load_file(sft_path), load_file(rl_path)
    keys = sorted(k for k in rl if any(part in k for part in EXPERT))
    return torch.cat([(rl[k].float() - sft[k].float()).flatten() for k in keys])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--round", default="160")
    args = parser.parse_args()
    deltas = {}
    for k in args.seeds.split(","):
        sft = sorted(glob.glob(f"outputs/cp0/a1_sft_s{k}/*/checkpoints/008000/pretrained_model/model.safetensors"))[-1]
        deltas[k] = delta(sft, f"outputs/rl/a1_rl_s{k}/round_{args.round}/model.safetensors")
    print(f"{deltas[next(iter(deltas))].numel()} coordinates; |delta| " +
          ", ".join(f"s{k} {d.norm():.4f}" for k, d in deltas.items()))
    values = []
    for a, b in itertools.combinations(deltas, 2):
        c = (deltas[a] @ deltas[b] / (deltas[a].norm() * deltas[b].norm())).item()
        values.append(c)
        print(f"seeds {a} and {b}: cosine {c:+.4f}")
    print(f"mean {sum(values) / len(values):+.4f}, every pair below 0.01: {all(v < 0.01 for v in values)}")


if __name__ == "__main__":
    main()
