set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint=$1; run_id=$2; condition=$3; budget=$4; seed=$5; step=$6
out="outputs/rl/eval/$run_id"
python -m smolvla_rl.eval.run "$checkpoint" --output-dir "$out" --episodes 20 --batch-size 10 \
  --parallel-tasks 1 --seed 1000 --stored-dtype 2>&1 | tee -a "logs/rl_eval_${run_id}.log"
python -m smolvla_rl.eval.collect "$out" --run-id "$run_id" --condition "$condition" --budget "$budget" \
  --seed "$seed" --step "$step" --stage rl
