# CP2 preregistration

Committed on 2026-09-27 in the development repository (commit `28ae0b7`) before any CP2 run, with one amendment to the stop rule (commit `7101e02`) made after the first seed's second block and before any stop decision. This copy makes editorial changes only: references to internal planning documents are replaced by their content, personal names by "the author", and the two compute planning sections (estimated GPU hours) are omitted. No prediction, metric, threshold or rule is changed. The prediction is the author's hypothesis going in, kept as stated although the CP1 evidence pointed lower.

## Registration

```
Prediction (the author): (a) A1-RL shows a positive, meaningful paired lift over A1-SFT, above 5 points.
  (b) A1-RL reaches A25-SFT: its seed mean is at or above 81.2, the lower bound of A25-SFT's 95% interval.
Primary metric: paired lift, A1-RL seed k minus A1-SFT seed k, held out success on LIBERO-Goal
  initial states 0 to 19, 20 episodes per task, 200 per policy; mean over seeds with a t interval.
Headline comparison: A1-RL against A25-SFT, 84.2 [81.2, 87.2] (CP0, same protocol).
Rollout budget per condition: 160 rounds x 10 episodes = 1,600 training episodes per seed,
  8 updates of 200 episodes. Seeds: 0 and 1 first; escalate to 0 to 4 if inconclusive, where
  inconclusive means the paired lift's 95% t interval (df 1) includes 0, or the seed trigger fires
  (A1-RL against A25-SFT gap under 20 points, or between seed sd above 8).
Decision rule: A1-RL against A25-SFT gap above 20 points decisive; 10 to 20, escalate eval
  rollouts to the power figure for a 10 point gap (200 to 390 per condition); under 10, no
  direction claimed until the figure for a 5 point gap (430 to 690 per condition) is met.
What result would falsify the prediction, with seeds as the unit and 95% t intervals:
  (a) supported if the paired lift's interval lies entirely above 5; falsified if it lies entirely
      at or below 5; otherwise undetermined.
  (b) supported if the A1-RL seed mean is at or above 81.2; falsified if the upper bound of A1-RL's
      interval is below 81.2; otherwise undetermined.
  An undetermined part is reported as undetermined, never as support.
```

## Recipe, frozen for every seed

| setting | value | evidence |
| --- | --- | --- |
| start | A1-SFT seed k, RL seed k | CP0 checkpoints |
| tasks | all 10, one per round in rotation, so each update holds exactly 2 rounds of every task | `train.py` |
| training states | 20 to 49; held out 0 to 19 | |
| baseline | per state running mean, last 5 visits, group mean on first visit | lowest gradient noise trace under every sampler tested (CP1) |
| sampler | joint noise 1.5 over 10 denoising steps | fixed outcome states 27 of 50 against 40 at one step noise 0.5, success 0.545 against 0.56, within one standard error (CP1) |
| update | gradient accumulated over 20 rounds (200 episodes), one AdamW step, grad clip 1.0, weight decay 0 | single task noise scale at least 200 episodes (CP1) |
| learning rate | 2.5e-7, constant | declared, see below |
| checkpoints | every 40 rounds, with the full trainer state for exact resume | |
| eval | `scripts/eval_rl.sh <round_160> a1_rl_s<k> a1_rl 1 <k> 160`: 20 per task, batch 10, seed 1000, `--stored-dtype` | same layout as CP0 (states 0 to 19); SFT checkpoints load identically with or without `--stored-dtype` (CP1) |

Launch, per seed: `python -m smolvla_rl.rl.train --condition a1_sft --seed <k> --noise-joint --noise-level 1.5 --rounds 160 --rounds-per-update 20 --lr 2.5e-7 --save-every 40 --run-id a1_rl_s<k>`, preceded by the same command with `--probe`.

## Why the learning rate is declared, not swept

2.5e-7 was chosen before any outcome at this recipe, from the measured action shift of one Adam step, which is linear in the learning rate. At it, 16 real updates over two runs had step KL 0.0020 to 0.0149 with no instability. A sweep was not run because it could not resolve the differences it would look for: the task 4 lift was +4 to +5 points, and detecting a 5 point gap needs 430 to 690 episodes per condition, against 100 to 200 per sweep evaluation. Consequence: the lift is reported at one declared learning rate.

## Stop rule

Block 0 (rounds 0 to 19) is collected before the first update, so it is SFT's own success under the training sampler. Stop a run if any later block is more than 30 successes (15 points) below block 0. Why 15: the difference of two 200 episode blocks has sd about 5 points at p = 0.5 under binomial noise; 15 is 3 sd. Blocks revisit the same states, which lowers the true sd, so the rule errs toward not stopping.

**Amendment 1 (2026-09-27 22:50 PDT, after seed 0's block 1, before any stop decision).** The rule above is wrong for this layout. With 10 tasks, 30 training states and a group of 10, each block holds 2 visits per task and visit v uses states 20 + 10 (v mod 3) onward, so the states a block covers repeat only every 3 blocks (block 0: 20 to 39, block 1: 40 to 49 and 20 to 29, block 2: 30 to 49; checked in `logs/rl_episodes.csv`). "Blocks revisit the same states" held for the single task runs, not here, so block against block 0 would compare state difficulty. Replaced by: stop if a block's successes fall more than 30 below the first visits of the same 200 (task, state) pairs, computed by `scripts/stop_rule.py`. Each block holds 200 distinct pairs once, so the comparison is paired and equal sized; the 30 keeps the same 3 sd binomial argument, and pairing makes it err toward not stopping. The stop rule is a safety guard only; the prediction, metric and decision rules are unchanged. At amendment time: block 0 +0, block 1 −4.

## Interpretation rule, set before seeing results

A small or null lift is ambiguous between "450M does not lift" and "updates were noise". The single task noise scale was measured; the multi task one was not. So each pair of seeds gets the cosine of their weight changes at round 160 (the measurement used in the CP1 gate: 0.176 on task 4, chance about 1e-4). If every pair is below 0.01, the run is reported as noise dominated and underpowered, not as a scale finding.

## Evidence available to the prediction

- Task 4, A1 seed 0: 1,600 episodes on that one task gave +4 and +5 under the ODE sampler on training states, and a training trend of +1.12 points per block (CP1).
- CP2 spends 1,600 episodes across 10 tasks: 160 per task, one tenth of what produced the task 4 lift. Whether updates from other tasks help or hinder a given task was not measured.
- At five seeds the eval holds 1,000 episodes per condition, which meets the power figure for a 5 point gap (430 to 690 per condition in the 75 to 90 percent regime; A1 sits near 68 percent, where binomial variance is about 16 percent higher, so the requirement is somewhat above that range, inferred).
- Seed escalation was expected: A1-SFT seeds 0 and 1 score 64.0 and 80.0 (mean 72.0), 12.2 below A25-SFT, so the gap trigger fires unless RL lowers success by more than 7.8 points. With two seeds the 95% t interval half width is 12.71 s / √2 = 9.0 s, against 1.24 s at five.

## Deviations from the original plan, recorded

1. Eval: 20 episodes per task on held out states, not 50 per task.
2. Learning rate declared on A1 task 4 from a step size measurement, not swept on the A25 condition.
3. No KL term and no critic, so neither is swept; step KL is logged per update.
4. Training denoising steps: 10 with joint noise, not 1.
5. CP1 design choices (baseline, sampler, learning rate) were made on A1 seed 0 task 4, whose states include the held out 0 to 19. CP2 trains fresh from SFT, but the choices were informed by that task.
6. Seeds: two first, not three, with the escalation rule above.
