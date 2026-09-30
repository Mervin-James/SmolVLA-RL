import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = "configs/subsets/libero_goal.json"

committed = json.loads(subprocess.run(["git", "show", f"HEAD:{PATH}"], capture_output=True, text=True, cwd=ROOT).stdout)
current = json.loads((ROOT / PATH).read_text())

assert committed["dataset_revision"] == current["dataset_revision"], "dataset revision changed"
assert committed["tasks"] == current["tasks"], "task list changed"
for seed, budgets in committed["episodes"].items():
    for budget, episodes in budgets.items():
        assert current["episodes"][seed][budget] == episodes, f"seed {seed} budget {budget} CHANGED"
print(f"seeds {sorted(committed['episodes'])} unchanged; now {sorted(current['episodes'])}")
