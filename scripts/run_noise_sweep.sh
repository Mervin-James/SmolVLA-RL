set -u
cd "$(dirname "$0")/.."

STEPS="${STEPS:-8000}"
EPISODES="${EPISODES:-20}"
BATCH="${BATCH:-10}"
OUTPUTS="${OUTPUTS:-outputs/cp0}"
EPISODES_CSV="${EPISODES_CSV:-logs/episodes_noise.csv}"
RUNS="${RUNS:-a25_sft:0:flow_ode:0.0:single a25_sft:0:flow_sde:0.5:single a1_sft:0:flow_sde:0.5:single \
a25_sft:0:flow_sde:0.3:single a25_sft:0:flow_sde:0.8:single a1_sft:0:flow_sde:0.3:single \
a1_sft:0:flow_sde:0.8:single a25_sft:0:flow_sde:0.5:joint}"

say () { echo "[$(date +%F' '%H:%M:%S)] $*"; }
run_dir () { ls -dt "$OUTPUTS/${1}_s${2}/"*/ 2>/dev/null | head -1; }
budget_of () { sed -n 's/^budget:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "configs/conditions/$1.yaml"; }

child=""
trap 'kill $child 2>/dev/null; wait; say "interrupted"; exit 130' INT TERM

mkdir -p logs
for spec in $RUNS; do
  IFS=: read -r condition seed method level mode <<< "$spec"
  tag="${method#flow_}${level}_${mode}"
  run_id="${condition}_s${seed}_${tag}"
  if [ -f "$EPISODES_CSV" ] && grep -q "^${run_id}," "$EPISODES_CSV"; then
    say "$run_id already evaluated"; continue
  fi
  ckpt="$(run_dir "$condition" "$seed")checkpoints/$(printf %06d "$STEPS")/pretrained_model"
  [ -d "$ckpt" ] || { say "missing checkpoint for $run_id, halted"; exit 1; }
  out="outputs/noise_sweep/$run_id"
  rm -rf "$out"
  say "$run_id: ${EPISODES} episodes per task, sampler $method level $level $mode"
  started=$(date +%s)
  python -m smolvla_rl.eval.run "$ckpt" --output-dir "$out" --episodes "$EPISODES" --batch-size "$BATCH" \
    --parallel-tasks 1 --seed 1000 --noise-method "$method" --noise-level "$level" \
    ${mode:+$( [ "$mode" = joint ] && echo --noise-joint )} > "logs/noise_${run_id}.log" 2>&1 &
  child=$!
  wait "$child" || { say "$run_id eval FAILED, halted"; tail -n 25 "logs/noise_${run_id}.log"; exit 1; }
  child=""
  python -m smolvla_rl.eval.collect "$out" --run-id "$run_id" --condition "$condition" \
    --budget "$(budget_of "$condition")" --seed "$seed" --step "$STEPS" \
    --distribution "libero_goal_${tag}" --out "$EPISODES_CSV" || { say "$run_id collect FAILED, halted"; exit 1; }
  say "$run_id done in $(( ($(date +%s) - started) / 60 )) min"
done
say "noise sweep done"
