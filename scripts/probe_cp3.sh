set -u
cd "$(dirname "$0")/.."

DEADLINE="${DEADLINE:-55}"
WORKERS="${WORKERS:-8}"
PLUS_VENV="${PLUS_VENV:-$HOME/venvs/vlaplus}"
PLUS_CLONE="${PLUS_CLONE:-$HOME/libero-plus}"
PLUS_CONFIG="${PLUS_CONFIG:-$HOME/.libero-plus}"
SAMPLE="${SAMPLE:-configs/libero_plus_sample.json}"
tmp=$(mktemp -d /tmp/probe_cp3.XXXXXX)
started=$(date +%s)

"$PLUS_VENV/bin/python" scripts/patch_libero_plus.py "$PLUS_CLONE" || exit 1

python3 - "$SAMPLE" "$tmp/sample.json" <<'PY'
import json, sys
sample = json.load(open(sys.argv[1]))
ids = [1770, 1852]
assert all(i in sample["task_ids"] for i in ids)
json.dump({**sample, "n_tasks": len(ids), "task_ids": ids,
           "category_of": {str(i): sample["category_of"][str(i)] for i in ids}}, open(sys.argv[2], "w"))
PY

timeout -s TERM "$DEADLINE" env SAMPLE="$tmp/sample.json" SHARDS=2 CONDITIONS=a1_sft SEEDS=0 EPISODE_LENGTH=10 \
  EPISODES_CSV="$tmp/episodes.csv" EVAL_ROOT="$tmp/eval" LOG_DIR="$tmp/logs" \
  PLUS_VENV="$PLUS_VENV" PLUS_CLONE="$PLUS_CLONE" PLUS_CONFIG="$PLUS_CONFIG" \
  bash scripts/run_cp3.sh > "$tmp/runner.log" 2>&1 &
runner=$!

PYTHONPATH="$PWD/src:$PLUS_CLONE" LIBERO_CONFIG_PATH="$PLUS_CONFIG" MUJOCO_GL=egl LP_NUM_THREADS=1 \
  timeout -s TERM "$DEADLINE" "$PLUS_VENV/bin/python" -m smolvla_rl.eval.probe_envs \
  --sample "$SAMPLE" --workers "$WORKERS" --deadline "$((DEADLINE - 5))"
envs=$?

wait "$runner"; runner_status=$?
rows=$( [ -f "$tmp/episodes.csv" ] && tail -n +2 "$tmp/episodes.csv" | wc -l || echo 0)
if [ "$runner_status" -eq 0 ] && [ "$rows" -eq 2 ]; then
  echo "end to end: fog 1770 and motion blur 1852 through policy, shards, collect and summary: ok ($rows rows)"
  pipeline=0
else
  echo "end to end: FAILED (runner status $runner_status, $rows of 2 rows), runner log:"
  tr '\r' '\n' < "$tmp/runner.log" | tail -n 30
  pipeline=1
fi

echo "probe took $(($(date +%s) - started)) s, files in $tmp"
[ "$envs" -eq 0 ] && [ "$pipeline" -eq 0 ] && echo "PROBE PASSED" || { echo "PROBE FAILED"; exit 1; }
