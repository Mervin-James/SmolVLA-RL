set -u
cd "$(dirname "$0")/.."

STEPS="${STEPS:-8000}"
BATCH="${BATCH:-32}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_BATCH="${EVAL_BATCH:-10}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONDITIONS="${CONDITIONS:-a1_sft a25_sft}"
EPISODES_CSV="${EPISODES_CSV:-logs/episodes.csv}"
OUTPUTS="${OUTPUTS:-outputs/cp0}"
export OUTPUTS

say () { echo "[$(date +%F' '%H:%M:%S)] $*"; }
run_dir () { ls -dt "$OUTPUTS/${1}_s${2}/"*/ 2>/dev/null | head -1; }
budget_of () { sed -n 's/^budget:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "configs/conditions/$1.yaml"; }

train () {
  condition=$1; seed=$2; dir=$(run_dir "$condition" "$seed")
  if [ -n "$dir" ] && [ -d "$dir/checkpoints/$(printf %06d "$STEPS")" ]; then
    say "$condition seed $seed already trained"; return 0
  fi
  if [ -n "$dir" ] && [ -d "$dir/checkpoints/last" ]; then
    say "$condition seed $seed resuming"
    lerobot-train --config_path="$dir/checkpoints/last/pretrained_model/" --resume=true \
      >> "logs/cp0_train_${condition}_s${seed}.log" 2>&1
  else
    say "$condition seed $seed training $STEPS steps"
    python -m smolvla_rl.sft.train "$condition" --seed "$seed" --steps "$STEPS" \
      --batch-size "$BATCH" --save-freq "$SAVE_FREQ" \
      >> "logs/cp0_train_${condition}_s${seed}.log" 2>&1
  fi
  dir=$(run_dir "$condition" "$seed")
  [ -d "$dir/checkpoints/$(printf %06d "$STEPS")" ] || { say "$condition seed $seed FAILED to reach $STEPS"; return 1; }
}

evaluate () {
  condition=$1; seed=$2; run_id="${condition}_s${seed}"
  if [ -f "$EPISODES_CSV" ] && grep -q "^${run_id}," "$EPISODES_CSV"; then
    say "$run_id already evaluated"; return 0
  fi
  dir=$(run_dir "$condition" "$seed")
  ckpt="$dir/checkpoints/$(printf %06d "$STEPS")/pretrained_model"
  budget=$(budget_of "$condition") || { say "cannot read budget for $condition"; return 1; }
  out="$OUTPUTS/eval/$run_id"
  say "$run_id evaluating ${EVAL_EPISODES} episodes per task"
  python -m smolvla_rl.eval.run "$ckpt" --output-dir "$out" --episodes "$EVAL_EPISODES" \
    --batch-size "$EVAL_BATCH" --parallel-tasks 1 --seed 1000 > "logs/cp0_eval_${run_id}.log" 2>&1 || { say "$run_id eval FAILED"; return 0; }
  python -m smolvla_rl.eval.collect "$out" --run-id "$run_id" --condition "$condition" \
    --budget "$budget" --seed "$seed" --step "$STEPS" --out "$EPISODES_CSV"
}

mkdir -p logs
for condition in $CONDITIONS; do
  for seed in $SEEDS; do
    train "$condition" "$seed" && evaluate "$condition" "$seed"
  done
done

say "CP0 done"
[ -f "$EPISODES_CSV" ] || { say "no episodes recorded"; exit 0; }
python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("logs/episodes.csv")))
by = collections.defaultdict(list)
for r in rows:
    by[(r["condition"], r["seed"])].append(int(r["success"]))
print(f"{'run':16s} {'n':>5s} {'success %':>10s}")
for key in sorted(by):
    v = by[key]
    print(f"{key[0]+' s'+key[1]:16s} {len(v):5d} {100*sum(v)/len(v):10.1f}")
PY
