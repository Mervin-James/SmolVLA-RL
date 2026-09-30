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


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = load(ROOT / "logs/episodes.csv")
    scores = per_seed(rows)
    figure_budget(scores, OUT / "cp0_budget.png")
    figure_tasks(rows, OUT / "cp0_per_task.png")
    for condition, values in scores.items():
        mean, low, high = mean_interval(list(values.values()))
        print(f"{LABELS[condition]:22s} {mean:5.1f}%  95% CI [{low:.1f}, {high:.1f}]  n={len(values)} seeds")
    print(f"wrote {OUT}/cp0_budget.png and {OUT}/cp0_per_task.png")


if __name__ == "__main__":
    main()
