import argparse
import collections
import csv
import json
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["Camera Viewpoints", "Robot Initial States", "Language Instructions",
              "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout"]

PUBLISHED = {
    "OpenVLA": [0.8, 3.5, 23.0, 8.1, 34.8, 15.2, 28.5],
    "WorldVLA": [0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0],
    "UniVLA": [1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9],
    "pi0": [13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9],
    "OpenVLA-OFT": [56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2],
}


def clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    low = stats.beta.ppf(0.025, successes, total - successes + 1) if successes else 0.0
    high = stats.beta.ppf(0.975, successes + 1, total - successes) if successes < total else 1.0
    return 100 * low, 100 * high


def standard_rates(path: Path) -> dict[str, float]:
    by_run = collections.defaultdict(list)
    for row in csv.DictReader(path.open()):
        by_run[row["run_id"]].append(int(row["success"]))
    return {k: 100 * sum(v) / len(v) for k, v in by_run.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus", default="logs/episodes_plus.csv")
    parser.add_argument("--standard", default="logs/episodes.csv")
    parser.add_argument("--sample", default="configs/libero_plus_sample.json")
    args = parser.parse_args()

    rows = list(csv.DictReader((ROOT / args.plus).open()))
    category_of = json.loads((ROOT / args.sample).read_text())["category_of"]
    standard = standard_rates(ROOT / args.standard)

    by_run = collections.defaultdict(list)
    by_run_category = collections.defaultdict(lambda: collections.defaultdict(list))
    by_condition_category = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        success = int(row["success"])
        category = category_of[row["task_id"]]
        by_run[row["run_id"]].append(success)
        by_run_category[row["run_id"]][category].append(success)
        by_condition_category[row["condition"]][category].append(success)

    print(f"{'policy':14s} {'standard':>9s} {'plus':>7s} {'n':>5s} {'95% CI':>16s}")
    for run_id in sorted(by_run):
        v = by_run[run_id]
        low, high = clopper_pearson(sum(v), len(v))
        rate = 100 * sum(v) / len(v)
        print(f"{run_id:14s} {standard.get(run_id, float('nan')):8.1f}% {rate:6.1f}% {len(v):5d}"
              f"   [{low:4.1f}, {high:4.1f}]")

    width = max(len(c) for c in CATEGORIES)
    conditions = sorted(by_condition_category)
    print(f"\n{'category':{width}s} " + " ".join(f"{c:>10s}" for c in conditions)
          + "   " + " ".join(f"{m:>12s}" for m in PUBLISHED))
    for index, category in enumerate(CATEGORIES):
        cells = []
        for condition in conditions:
            v = by_condition_category[condition][category]
            cells.append(f"{100 * sum(v) / len(v):9.1f}%" if v else f"{'-':>10s}")
        reference = " ".join(f"{PUBLISHED[m][index]:11.1f}%" for m in PUBLISHED)
        print(f"{category:{width}s} " + " ".join(cells) + "   " + reference)

    print("\nspread across categories, the discriminating statistic")
    for condition in conditions:
        rates = [100 * sum(v) / len(v) for c in CATEGORIES
                 if (v := by_condition_category[condition][c])]
        print(f"  {condition:10s} min {min(rates):5.1f}%  max {max(rates):5.1f}%  "
              f"range {max(rates) - min(rates):5.1f} points")
    for model, values in PUBLISHED.items():
        print(f"  {model:10s} min {min(values):5.1f}%  max {max(values):5.1f}%  "
              f"range {max(values) - min(values):5.1f} points   (published)")


if __name__ == "__main__":
    main()
