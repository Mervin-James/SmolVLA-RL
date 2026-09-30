# SmolVLA-RL

**Does one demonstration plus reinforcement learning lift a 450M vision language action model the way it lifts a 7B one?**

SimpleVLA-RL (arXiv 2509.09674) took OpenVLA-OFT, a 7B model, from 17.3 to 91.7 on LIBERO-Long after supervised fine tuning on one trajectory per task followed by online RL (their Table 5). This repository tests the same recipe at 450M on SmolVLA (arXiv 2506.01844), on LIBERO-Goal, using a single 16 GB consumer GPU.

## Status

| checkpoint | question | record |
| --- | --- | --- |
| CP0 | How well does supervised fine tuning do from 1 and from 25 demonstrations per task? | [docs/cp0-cold-start.md](docs/cp0-cold-start.md) |
| CP1 | Can RL be run correctly on SmolVLA's flow matching action expert? | [docs/cp1-rl-loop.md](docs/cp1-rl-loop.md) |
| CP2 | Does RL from one demonstration lift SmolVLA? Preregistered, five seeds | [docs/cp2-lift.md](docs/cp2-lift.md), [preregistration](docs/cp2-preregistration.md) |
| CP3 | Does anything change under perturbation? LIBERO-Plus, every policy | [docs/cp3-perturbation.md](docs/cp3-perturbation.md) |

CP0 result: 62.0 percent success from one demonstration per task (95% CI [47.5, 76.5]) and 84.2 percent from 25 (95% CI [81.2, 87.2]), five seeds each.

CP1 result: an exact per step likelihood for the action expert, verified numerically, and an update recipe (200 episodes per update) under which training success on a single task rises in two independent runs (permutation p = 0.0005).

CP2 result: no. RL changed held out success by -0.70 points (95% CI [-2.13, +0.73], five seeds); both parts of the preregistered prediction are falsified.

CP3 result: RL changes LIBERO-Plus success by -0.31 points (95% CI [-3.88, +3.27]); the demonstration gap shrinks from 22.2 points in distribution to 15.4 under perturbation but remains.

## Setup

WSL2 Ubuntu 24.04, Python 3.12, one CUDA GPU.

    uv venv ~/venvs/vla --python 3.12
    uv pip install --python ~/venvs/vla torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python ~/venvs/vla -e .
    export MUJOCO_GL=egl LP_NUM_THREADS=1

System packages: cmake build-essential libegl1 libgl1 libegl-dev libgl-dev libosmesa6.

## Layout

| path | contents |
| --- | --- |
| `configs/` | shared recipe (`base.yaml`), per condition deltas, demonstration subsets |
| `src/smolvla_rl/` | supervised fine tuning, the RL loop and its verification, evaluation |
| `scripts/` | experiment launchers and figure generation |
| `logs/` | raw per episode results, append only |
| `figures/` | every figure, regenerated from `logs/` by `scripts/make_figures.py` |
| `docs/` | one record per checkpoint: setup, decisions, results |
