set -o pipefail
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1}"

for k in $SEEDS; do
  run=a1_rl_s$k
  python -m smolvla_rl.rl.train --condition a1_sft --seed "$k" --noise-joint --noise-level 1.5 --rounds 160 \
    --rounds-per-update 20 --lr 2.5e-7 --save-every 40 --run-id "$run" < /dev/null 2>&1 | tee -a "logs/rl_cp2_$run.log" \
    || { echo "stopped: training $run failed"; exit 1; }
  bash scripts/eval_rl.sh "outputs/rl/$run/round_160" "$run" a1_rl 1 "$k" 160 < /dev/null \
    || { echo "stopped: evaluating $run failed"; exit 1; }
done
