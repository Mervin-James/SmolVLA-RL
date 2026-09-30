set -u
cd "$(dirname "$0")/.."

STEPS="${STEPS:-8000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONDITIONS="${CONDITIONS:-a1_sft a25_sft}"
SHARDS="${SHARDS:-5}"
SAMPLE="${SAMPLE:-configs/libero_plus_sample.json}"
EPISODES_CSV="${EPISODES_CSV:-logs/episodes_plus.csv}"
OUTPUTS="${OUTPUTS:-outputs/cp0}"
EVAL_ROOT="${EVAL_ROOT:-outputs/cp3_eval}"
LOG_DIR="${LOG_DIR:-logs}"
EPISODE_LENGTH="${EPISODE_LENGTH:-}"
PLUS_VENV="${PLUS_VENV:-$HOME/venvs/vlaplus}"
PLUS_CLONE="${PLUS_CLONE:-$HOME/libero-plus}"
PLUS_CONFIG="${PLUS_CONFIG:-$HOME/.libero-plus}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL_ARGS="${EVAL_ARGS:-}"
STAGE="${STAGE:-sft}"
BUDGET="${BUDGET:-}"
pids=""
tick=""
trap 'kill $pids $tick 2>/dev/null; wait; printf "\n"; say "interrupted, shards stopped"; exit 130' INT TERM

if pgrep -f lerobot-eval >/dev/null; then
  echo "lerobot-eval is already running, stop it before starting CP3:"
  pgrep -af lerobot-eval
  exit 1
fi

TOTAL=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['task_ids']))" "$SAMPLE")

say () { echo "[$(date +%F' '%H:%M:%S)] $*"; }
run_dir () { ls -dt "$OUTPUTS/${1}_s${2}/"*/ 2>/dev/null | head -1; }
budget_of () { sed -n 's/^budget:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "configs/conditions/$1.yaml"; }

ticker () {
  run_id=$1; out=$2; start=$3; shift 3
  nap=""
  trap 'kill $nap 2>/dev/null; exit 0' TERM
  while :; do
    alive=0
    for pid in "$@"; do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done
    [ "$alive" -gt 0 ] || break
    n=$(find "$out" -name '*.mp4' 2>/dev/null | wc -l)
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    printf '\r[%s] %s ~%3d/%d variants, %d/%d shards, %s, VRAM %s MiB, RAM free %s MiB  ' \
      "$(date +%H:%M:%S)" "$run_id" "$n" "$TOTAL" "$alive" "$SHARDS" \
      "$(rate "$n" "$(($(date +%s) - start))")" "${vram:-0}" "$(free -m | awk '/^Mem:/ {print $7}')"
    sleep 15 & nap=$!
    wait "$nap"
  done
}

rate () {
  python3 -c "
n, t, total = $1, $2, $TOTAL
print('measuring' if n < 3 else f'{t/n:.1f} s/variant, ETA {(total-n)*t/n/60:.0f} min, {total*t/n/3600:.1f} h/policy')"
}

evaluate () {
  condition=$1; seed=$2; run_id="${condition}_s${seed}"
  if [ -f "$EPISODES_CSV" ] && grep -q "^${run_id}," "$EPISODES_CSV"; then
    say "$run_id already evaluated on LIBERO-Plus"; return 0
  fi
  if [ -n "$CHECKPOINT" ]; then
    ckpt=$(printf '%s' "$CHECKPOINT" | sed "s/{condition}/$condition/g; s/{seed}/$seed/g")
  else
    dir=$(run_dir "$condition" "$seed")
    ckpt="$dir/checkpoints/$(printf %06d "$STEPS")/pretrained_model"
  fi
  [ -d "$ckpt" ] || { say "missing checkpoint for $run_id"; return 0; }
  out="$EVAL_ROOT/$run_id"
  rm -rf "$out" "$LOG_DIR"/cp3_eval_${run_id}_shard*.log
  say "$run_id evaluating $TOTAL LIBERO-Plus variants across $SHARDS processes"
  started=$(date +%s)
  pids=""
  for i in $(seq 0 $((SHARDS - 1))); do
    PYTHONPATH="$PWD/src:$PLUS_CLONE" LIBERO_CONFIG_PATH="$PLUS_CONFIG" MUJOCO_GL=egl LP_NUM_THREADS=1 \
      "$PLUS_VENV/bin/python" -m smolvla_rl.eval.run "$ckpt" --output-dir "$out/shard$i" \
      --env-type libero_plus --task-ids-file "$SAMPLE" --shard "$i" --num-shards "$SHARDS" \
      --episodes 1 --batch-size 1 --parallel-tasks 1 --seed 1000 ${EPISODE_LENGTH:+--episode-length "$EPISODE_LENGTH"} $EVAL_ARGS \
      > "$LOG_DIR/cp3_eval_${run_id}_shard$i.log" 2>&1 &
    pids="$pids $!"
  done
  ticker "$run_id" "$out" "$started" $pids &
  tick=$!
  remaining="$pids"; status=0
  while [ -n "$remaining" ]; do
    wait -n -p finished $remaining; status=$?
    remaining=$(printf '%s\n' $remaining | grep -vx "$finished" | tr '\n' ' ')
    remaining="${remaining% }"
    [ "$status" -eq 0 ] || break
  done
  kill $pids $tick 2>/dev/null; wait 2>/dev/null; tick=""; printf '\n'
  if [ "$status" -ne 0 ]; then
    say "$run_id eval FAILED: a shard exited with status $status, the other shards were stopped, CP3 halted"
    tail -n 25 "$LOG_DIR"/cp3_eval_${run_id}_shard*.log
    exit 1
  fi
  PYTHONPATH="$PWD/src" "$PLUS_VENV/bin/python" -m smolvla_rl.eval.collect "$out"/shard*/ \
    --run-id "$run_id" --condition "$condition" --budget "${BUDGET:-$(budget_of "$condition")}" \
    --seed "$seed" --step "$STEPS" --stage "$STAGE" --distribution libero_plus --out "$EPISODES_CSV" \
    --expect-task-ids "$SAMPLE" || { say "$run_id collect FAILED, CP3 halted"; exit 1; }
}

mkdir -p "$LOG_DIR"
for condition in $CONDITIONS; do
  for seed in $SEEDS; do evaluate "$condition" "$seed"; done
done

say "CP3 done"
[ -f "$EPISODES_CSV" ] && EPISODES_CSV="$EPISODES_CSV" SAMPLE="$SAMPLE" python3 - <<'PY'
import csv, collections, json, os, pathlib
rows = list(csv.DictReader(open(os.environ["EPISODES_CSV"])))
sample = json.loads(pathlib.Path(os.environ["SAMPLE"]).read_text())["category_of"]
by_run = collections.defaultdict(list)
by_cat = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by_run[(r["condition"], r["seed"])].append(int(r["success"]))
    by_cat[r["condition"]][sample.get(r["task_id"], "?")].append(int(r["success"]))
print(f"{'run':16s} {'n':>5s} {'success %':>10s}")
for key in sorted(by_run):
    v = by_run[key]
    print(f"{key[0]+' s'+key[1]:16s} {len(v):5d} {100*sum(v)/len(v):10.1f}")
print()
cats = sorted({c for d in by_cat.values() for c in d})
print(f"{'category':24s} " + " ".join(f"{c:>9s}" for c in sorted(by_cat)))
for cat in cats:
    cells = []
    for cond in sorted(by_cat):
        v = by_cat[cond][cat]
        cells.append(f"{100*sum(v)/len(v):9.1f}" if v else f"{'-':>9s}")
    print(f"{cat:24s} " + " ".join(cells))
PY
