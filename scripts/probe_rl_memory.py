import json
import time
from pathlib import Path

import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
SUBSETS = json.loads((ROOT / BASE["subsets"]).read_text())
BATCHES = [4, 8, 16]


def latest_checkpoint() -> str:
    runs = sorted((ROOT / "outputs").glob("a1_sft_s0/*/checkpoints/*/pretrained_model"))
    assert runs, "no trained checkpoint found"
    return str(runs[-1])


def gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def load(path: str, trainable: bool) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.from_pretrained(path).to("cuda")
    policy.train() if trainable else policy.eval()
    if not trainable:
        for p in policy.parameters():
            p.requires_grad_(False)
    return policy


def make_batch(dataset: LeRobotDataset, preprocessor, size: int) -> dict:
    rows = [dataset[i] for i in range(size)]
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        batch[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return preprocessor(batch)


def main() -> None:
    torch.cuda.reset_peak_memory_stats()
    checkpoint = latest_checkpoint()
    print(f"checkpoint: {checkpoint}")
    policy = load(checkpoint, trainable=True)
    print(f"policy loaded: {gb():.2f} GB")

    reference = load(checkpoint, trainable=False)
    print(f"+ frozen reference: {gb():.2f} GB")

    hidden = policy.model.vlm_with_expert.expert_hidden_size
    critic = torch.nn.Sequential(
        torch.nn.Linear(hidden, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(), torch.nn.Linear(1024, 1),
    ).to("cuda")
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    print(f"+ critic and optimizers: {gb():.2f} GB")

    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    dataset = LeRobotDataset(SUBSETS["dataset"], revision=SUBSETS["dataset_revision"],
                             episodes=SUBSETS["episodes"]["0"]["1"])

    for size in BATCHES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = make_batch(dataset, preprocessor, size)
        reps = 5
        for _ in range(2):
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(reps):
            with torch.no_grad():
                reference.forward(batch)
        torch.cuda.synchronize()
        reference_s = (time.perf_counter() - start) / reps

        start = time.perf_counter()
        for _ in range(reps):
            with torch.no_grad():
                reference.forward(batch)
            loss, _ = policy.forward(batch)
            value = critic(torch.zeros(size, hidden, device="cuda"))
            (loss + value.mean() * 0).backward()
            optimizer.step(), critic_optimizer.step()
            optimizer.zero_grad(set_to_none=True), critic_optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        update_s = (time.perf_counter() - start) / reps

        print(f"batch {size:3d}: {update_s:.3f} s per update "
              f"({3600 / update_s:.0f} updates per hour, {size * 3600 / update_s:.0f} samples per hour), "
              f"reference forward {reference_s:.3f} s, peak {gb():.2f} GB")


if __name__ == "__main__":
    main()
