#!/usr/bin/env bash
# Short experiment on piece endings (scenario 2). Run from the repo root on a GPU machine:
#   bash scripts/run_karar.sh
# Trains the countdown model in both token orders (pitch-duration and duration-pitch) for the
# Transformer and the LSTM, generates whole pieces and saves their endings for inspection.
# Existing checkpoints are reused. Restartable.
set -euo pipefail
SEED=${SEED:-1}

train () {
  NAME=$(python scripts/03_train_baseline.py "$@" --print_name)
  if [ -f "runs/$NAME/best.pt" ] && grep -q '"few_shot"' "runs/$NAME/metrics.json" 2>/dev/null; then
    echo "skip training $NAME"
  else
    python scripts/03_train_baseline.py "$@"
  fi
}

endings () {   # run-name
  [ -f "runs/$1/gen_prompt0_full_constrained_samples.jsonl" ] || python scripts/04_evaluate_generation.py "runs/$1" --prompt 0 --full --constrain --save_samples
  [ -f "runs/$1/gen_prompt0_full_samples.jsonl" ] || python scripts/04_evaluate_generation.py "runs/$1" --prompt 0 --full --save_samples
  [ -f "runs/$1/gen_prompt16.json" ] || python scripts/04_evaluate_generation.py "runs/$1" --prompt 16
}

T="--model transformer --pitch koma53 --pos --ctx 768 --meter --rem --seed $SEED"
L="--model lstm --pitch koma53 --dropout 0.5 --pos --meter --rem --seed $SEED"
train $T;            endings "$NAME"
train $T --order dp; endings "$NAME"
train $L;            endings "$NAME"
train $L --order dp; endings "$NAME"
python scripts/05_summarize_runs.py
