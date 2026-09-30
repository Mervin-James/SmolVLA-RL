# CP1: an RL loop for SmolVLA's flow matching action expert, and evidence that it is correct

SmolVLA produces actions by integrating a learned velocity field over 10 Euler steps. That deterministic path has no tractable log probability, so policy gradient methods cannot be applied to it directly. This checkpoint makes the path stochastic, derives an exact per step likelihood, verifies it numerically, and establishes an update recipe whose effect on a single task is reproducible across seeds.

## Method

**Noise injection.** Each denoising step becomes a Gaussian transition whose standard deviation is `sigma_i * sqrt(delta_i)`, with `sigma_i = a * sqrt(t_i / (1 - t_i))` and `delta_i` the step size, and whose mean is the Euler step plus the drift correction that keeps the per step marginals equal to those of the ODE (the SDE form of Flow-GRPO, arXiv 2505.05470). `a` is the noise level. The implementation follows the `flow_sde` sampler of RLinf (github.com/RLinf/RLinf, reported in πRL, arXiv 2510.25889), whose timestep convention (t from 1 to 0, dt = -1/N) is the one SmolVLA's sampler already uses. The chain log probability is the sum of Gaussian log densities over denoising steps, the 10 executed actions of a chunk and the 7 action dimensions. At evaluation the noise is removed and the original ODE sampler is used.

**Update.** A PPO clipped surrogate on the chain log probability, AdamW with weight decay 0, gradient norm clipped to 1.0. Only the action expert trains (99.9M parameters, held in fp32). Reward is binary task success.

Code: `src/smolvla_rl/rl/noise.py`, `likelihood.py`, `loss.py`, `rollout.py`, `train.py`; evaluation of RL checkpoints through `src/smolvla_rl/rl/noisy_eval.py`.

## Verification gate

| check | result | tolerance |
| --- | --- | --- |
| 1a chain against lerobot's own sampler (ODE, no noise) | max gap 5.7e-3 to 6.5e-3, below lerobot's own spread under 1e-6 input jitter | 3x that jitter floor |
| 1b closed form log probability against Monte Carlo | 4,096 draws, max \|z\| of the mean 3.5 to 4.0, std within 4.3% | \|z\| under 5.5, std within 6% |
| 2 expected score is zero | max \|z\| 3.3 to 4.1 over 1,600 elements | \|z\| under 5.5 |
| 3 importance ratio is 1 before any update | 6.1e-5 on stored rollouts, recomputed per policy call | 1e-3 |
| 4 constant reward moves nothing | max parameter change exactly 0 | 0 |
| 5 reward rises with training | see the accumulated update recipe below | one sided permutation p under 0.05 |

Checks 1 to 4 run with `python -m smolvla_rl.rl.verify` and, on the live configuration, `python -m smolvla_rl.rl.train --probe`.

## Faults the gate caught

| fault | measured effect | fix |
| --- | --- | --- |
| Trainable weights in bf16 | updates at RL learning rates round away | upcast trainable tensors to fp32 |
| Log probabilities recomputed on a merged batch | \|ratio - 1\| of 3.1e-2 before any update, because bf16 kernels depend on batch size | recompute per policy call, in the rollout's exact layout (6.1e-5) |
| lerobot reloads a saved expert in bf16 | only 10.5M of 99.8M changed weights survive a reload, cosine 0.41 with the true update | load at the stored dtype (`--stored-dtype`); SFT checkpoints load identically either way |

## First recipe: one update per round of 10 episodes

On LIBERO-Goal task 4, from A1-SFT (100 episode ODE evals on the same episodes, `logs/episodes_task4.csv`):

| start | SFT | RL, run A | RL, run B |
| --- | --- | --- | --- |
| A1-SFT seed 0, group baseline | 54 | 71 | 47 |
| A1-SFT seed 2, group baseline | 23 | 14 | 27 |
| A1-SFT seed 0, per initial state baseline | 54 | 68 | 51 |

Identical runs landed up to 24 points apart; the mean change was about zero. Two measurements explain why.

**The initial state decides the outcome.** 38 of task 4's 50 initial states gave the same outcome on every visit. A baseline averaged over a group of different states therefore scores state difficulty rather than action quality. Advantages are instead computed against a running mean of the last 5 visits to the same (task, state) pair.

**Ten episodes are far below the gradient noise scale.** With the policy frozen, per episode score gradients over 200 episodes (`src/smolvla_rl/rl/grad_noise.py`) give a pairwise cosine between round gradients of 0.00 ± 0.02 under every sampler and baseline tested. That bounds the per round signal to noise ratio at about 0.05, which puts the noise scale at 200 episodes or more per update. The per state baseline lowered the noise trace 2 to 3 times relative to the group baseline in every configuration.

**Exploration noise costs little at deployment.** SFT policies evaluated with injected noise (`logs/episodes_noise.csv`, 200 episodes each): A25 seed 0 scores 90.0 without noise and 89.0 to 90.5 at noise levels 0.3 to 0.8; A1 seed 0 scores 61.5 to 65.5. Noise applied jointly at all 10 denoising steps at level 1.5 reduced the number of task 4 states with a fixed outcome from 40 to 27 without changing success (0.545 against 0.56, within one standard error).

## Accumulated update recipe

Gradients accumulated over 20 rounds (200 episodes) per optimizer step, joint noise 1.5 over all 10 denoising steps, per state baseline, 8 updates, learning rate 2.5e-7. The learning rate was chosen before any outcome at this recipe from `src/smolvla_rl/rl/step_size.py`: the action shift of one Adam step is linear in the learning rate (0.0009, 0.0022, 0.0044, 0.0088 at 1e-7, 2.5e-7, 5e-7, 1e-6).

Two runs from A1-SFT seed 0 on task 4, RL seeds 0 and 10, all 50 initial states used for training:

| evidence | RL seed 0 | RL seed 10 |
| --- | --- | --- |
| ODE eval, 100 episodes, against SFT's 54 on the same episodes | 59 (fixed 5, broke 0), exact McNemar p = 0.062 | 58 (fixed 5, broke 1), p = 0.219 |
| training success slope per 20 round block, permutation within initial state | +0.81 points, p = 0.019 | +1.43 points, p = 0.0005 |
| step KL, first to last update | 0.0088 to 0.0022 | 0.0149 to 0.0020 |

Pooled over the two independent runs, training success rose by +1.12 points per block (one sided permutation p = 0.0005, 2,000 permutations, the smallest value that many can resolve). Permutation p values are Monte Carlo estimates; the ones here come from `scripts/cp1_analysis.py` with its fixed seed. The expert weight changes of the two runs have cosine 0.176 at the end of training, against about 1e-4 for random directions at this dimension; this shows a shared direction, and the success data show that the direction improves the policy.

**Check 5, with its deviations recorded.** Training success rises with updates in both runs, which passes the gate. Four deviations from the check as originally specified: task 4 is a middling task (54 percent at SFT), not a deliberately easy one; success did not rise monotonically block by block; all 50 states were training states; and the deployment (ODE) lift is positive in both runs but not significant in either.

## Compute

Peak allocated VRAM 3.3 GiB per round with one noisy step and 5.4 GiB in the joint noise probe, well within 16 GB. The binding resource is host memory: each LIBERO simulator process holds about 2 GB, so a group of 10 needs about 25 GB. About 100 s per round of 10 episodes.

## Reproducing

    python -m smolvla_rl.rl.verify
    python -m smolvla_rl.rl.train --condition a1_sft --seed 0 --tasks 4 --train-states 0:50 --baseline state --noise-joint --noise-level 1.5 --rounds 160 --rounds-per-update 20 --lr 2.5e-7 --save-every 40 --run-id task4_s0 --probe
    python -m smolvla_rl.rl.train --condition a1_sft --seed 0 --tasks 4 --train-states 0:50 --baseline state --noise-joint --noise-level 1.5 --rounds 160 --rounds-per-update 20 --lr 2.5e-7 --save-every 40 --run-id task4_s0
    python -m smolvla_rl.eval.run outputs/rl/task4_s0/round_160 --output-dir outputs/rl/eval/task4_s0 --episodes 100 --task-ids 4 --stored-dtype

`--condition a1_sft --seed 0` resolves to the CP0 checkpoint of A1-SFT seed 0 under `outputs/cp0/`.

The accumulated recipe's statistics regenerate with `python scripts/cp1_analysis.py`.

Per episode training rows are in `logs/rl_episodes.csv`, per update rows in `logs/rl_updates.csv`, the first recipe's rows in `logs/cp1/`.
