import csv
import collections
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
MARK, ACCENT = "#2a78d6", "#eb6834"
LABELS = {"a1_sft": "1 demo per task", "a25_sft": "25 demos per task"}
SERIES = {"a1_sft": ("A1-SFT", "#2a78d6"), "a1_rl": ("A1-RL", "#eb6834"), "a25_sft": ("A25-SFT", "#1baf7a")}


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def per_seed(rows: list[dict]) -> dict[str, dict[str, float]]:
    cell = collections.defaultdict(list)
    for row in rows:
        cell[(row["condition"], row["seed"])].append(int(row["success"]))
    out = collections.defaultdict(dict)
    for (condition, seed), values in cell.items():
        out[condition][seed] = 100 * sum(values) / len(values)
    return out


def mean_interval(values: list[float]) -> tuple[float, float, float]:
    mean = st.mean(values)
    half = stats.t.ppf(0.975, len(values) - 1) * st.stdev(values) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def figure_budget(scores: dict[str, dict[str, float]], path: Path) -> None:
    conditions = ["a1_sft", "a25_sft"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor=SURFACE)
    for x, condition in enumerate(conditions):
        values = list(scores[condition].values())
        mean, low, high = mean_interval(values)
        ax.scatter([x] * len(values), values, s=46, color=MARK, zorder=3, alpha=0.85)
        ax.plot([x - 0.14, x + 0.14], [mean, mean], color=ACCENT, linewidth=2.5, zorder=4)
        ax.plot([x, x], [low, high], color=ACCENT, linewidth=2, zorder=2)
        ax.annotate(f"{mean:.1f}%", (x + 0.2, mean), color=INK, fontsize=11, va="center")
    ax.set_xticks(range(len(conditions)), [LABELS[c] for c in conditions], fontsize=11, color=INK)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("LIBERO-Goal success rate (%)", color=INK)
    ax.set_title("Demonstration budget, SmolVLA 450M\n5 seeds, 200 episodes each", color=INK, fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def figure_tasks(rows: list[dict], path: Path) -> None:
    cell = collections.defaultdict(list)
    for row in rows:
        cell[(row["condition"], int(row["task_id"]))].append(int(row["success"]))
    tasks = sorted({int(row["task_id"]) for row in rows})
    a1 = [100 * sum(cell[("a1_sft", t)]) / len(cell[("a1_sft", t)]) for t in tasks]
    a25 = [100 * sum(cell[("a25_sft", t)]) / len(cell[("a25_sft", t)]) for t in tasks]
    order = np.argsort(a1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    y = np.arange(len(tasks))
    for i, idx in enumerate(order):
        ax.plot([a1[idx], a25[idx]], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter([a1[i] for i in order], y, s=46, color=MARK, zorder=3, label="1 demo")
    ax.scatter([a25[i] for i in order], y, s=46, color=ACCENT, zorder=3, label="25 demos")
    ax.set_yticks(y, [f"task {tasks[i]}" for i in order], fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("success rate (%), pooled over 5 seeds", color=INK)
    ax.set_title("Every task gains from more demonstrations", color=INK, fontsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


def strip(ax, scores: dict[str, dict[str, float]], title: str) -> None:
    for x, condition in enumerate(SERIES):
        name, color = SERIES[condition]
        values = [scores[condition][k] for k in sorted(scores[condition])]
        mean, low, high = mean_interval(values)
        ax.scatter([x] * len(values), values, s=46, color=color, zorder=3, alpha=0.9, label=name)
        ax.plot([x - 0.16, x + 0.16], [mean, mean], color=INK, linewidth=2, zorder=4)
        ax.plot([x, x], [low, high], color=INK, linewidth=1.5, zorder=2)
        ax.annotate(f"{mean:.1f}", (x + 0.22, mean), color=INK, fontsize=10, va="center")
    for seed in sorted(scores["a1_sft"]):
        ax.plot([0, 1], [scores["a1_sft"][seed], scores["a1_rl"][seed]], color=GRID, linewidth=1.5, zorder=1)
    ax.set_xticks(range(len(SERIES)), [SERIES[c][0] for c in SERIES], fontsize=10, color=INK)
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(0, 100)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    style(ax)


def paired_lift(scores: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    return mean_interval([scores["a1_rl"][k] - scores["a1_sft"][k] for k in sorted(scores["a1_rl"])])


def figure_cp2(scores: dict[str, dict[str, float]], path: Path) -> None:
    mean, low, high = paired_lift(scores)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE)
    strip(ax, scores, f"RL from one demonstration does not lift SmolVLA 450M\n"
                      f"LIBERO-Goal held out states, 5 seeds; paired lift {mean:+.1f} [{low:+.1f}, {high:+.1f}]")
    ax.set_ylabel("success rate (%)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def block_differences(rows: list[dict], run_id: str) -> list[float]:
    first, blocks = {}, collections.defaultdict(list)
    for row in sorted((r for r in rows if r["run_id"] == run_id), key=lambda r: int(r["round"])):
        key = (row["task_id"], row["init_state"])
        first.setdefault(key, int(row["success"]))
        blocks[int(row["round"]) // 20].append((key, int(row["success"])))
    return [100 * (sum(s for _, s in blocks[b]) - sum(first[k] for k, _ in blocks[b])) / len(blocks[b])
            for b in sorted(blocks)]


def figure_training(rows: list[dict], path: Path) -> None:
    curves = {k: block_differences(rows, f"a1_rl_s{k}") for k in "01234"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    for k, curve in curves.items():
        ax.plot(range(len(curve)), curve, color=GRID if k != "1" else MUTED, linewidth=1.5, zorder=2)
        ax.annotate(f"seed {k}", (len(curve) - 1 + 0.1, curve[-1]), color=MUTED, fontsize=8, va="center")
    mean = [st.mean(c[b] for c in curves.values()) for b in range(8)]
    ax.plot(range(8), mean, color=MARK, linewidth=2.5, marker="o", markersize=5, zorder=3, label="mean of 5 seeds")
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xlim(-0.3, 8.2)
    ax.set_xlabel("block of 20 rounds (one optimizer update each)", color=INK)
    ax.set_ylabel("success minus first visits of the same pairs (points)", color=INK)
    ax.set_title("Training sampler success on matched (task, state) pairs, A1-RL", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    all_rows = load(ROOT / "logs/episodes.csv")
    rows = [r for r in all_rows if r["stage"] == "sft" and r["condition"] in LABELS]
    scores = per_seed(rows)
    figure_budget(scores, OUT / "cp0_budget.png")
    figure_tasks(rows, OUT / "cp0_per_task.png")
    for condition, values in scores.items():
        mean, low, high = mean_interval(list(values.values()))
        print(f"{LABELS[condition]:22s} {mean:5.1f}%  95% CI [{low:.1f}, {high:.1f}]  n={len(values)} seeds")
    goal = per_seed([r for r in all_rows if r["condition"] in SERIES])
    figure_cp2(goal, OUT / "cp2_lift.png")
    figure_training(load(ROOT / "logs/rl_episodes.csv"), OUT / "cp2_training.png")
    mean, low, high = paired_lift(goal)
    print(f"LIBERO-Goal: A1-RL paired lift {mean:+.2f} [{low:+.2f}, {high:+.2f}]")
    print(f"wrote {OUT}: cp0_budget, cp0_per_task, cp2_lift, cp2_training")


if __name__ == "__main__":
    main()
