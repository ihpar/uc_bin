#!/usr/bin/env bash
# Experiment grid for the preliminary study. Run on a GPU machine from the repo root:
#   SEEDS="1" bash scripts/run_baselines.sh    # single-seed pass (do this first)
#   bash scripts/run_baselines.sh              # seeds 1 2 3
#   SWEEP=0 bash scripts/run_baselines.sh      # skip the regularisation sweep
# Finished steps are skipped, so the script can be restarted at any time.
set -euo pipefail
SEEDS=${SEEDS:-"1 2 3"}
SWEEP=${SWEEP:-1}

train () {   # all arguments are passed to the training script; echoes nothing, sets $NAME
  NAME=$(python scripts/03_train_baseline.py "$@" --print_name)
  if [ -f "runs/$NAME/best.pt" ] && grep -q '"few_shot"' "runs/$NAME/metrics.json" 2>/dev/null; then
    echo "skip training $NAME"
  else
    python scripts/03_train_baseline.py "$@"
  fi
}

gen () {     # run-name, then arguments of the generation evaluation
  local name="$1"; shift
  local tag="gen_prompt"; local prompt=16 full="" con=""
  for arg in "$@"; do
    case "$arg" in --full) full="_full";; --constrain) con="_constrained";; esac
  done
  [[ " $* " == *" --prompt 0 "* ]] && prompt=0
  if [ -f "runs/$name/${tag}${prompt}${full}${con}.json" ]; then echo "skip ${tag}${prompt}${full}${con} $name"; else
    python scripts/04_evaluate_generation.py "runs/$name" "$@"
  fi
}

evaluate_all () {   # the two usage scenarios, each with free and usul-constrained decoding
  gen "$1" --prompt 16
  gen "$1" --prompt 16 --constrain
  gen "$1" --prompt 0 --full
  gen "$1" --prompt 0 --full --constrain
}

for s in $SEEDS; do
  for pitch in tet12 aeu koma53 koma53_rel; do                         # A. pitch encoding
    train --model transformer --pitch "$pitch" --seed "$s"; evaluate_all "$NAME"
  done
  train --model lstm --pitch koma53_rel --seed "$s"; evaluate_all "$NAME"          # B. architecture
  train --model transformer --pitch koma53_rel --cond none --seed "$s"; evaluate_all "$NAME"   # C. conditioning
  train --model transformer --pitch koma53 --pos --seed "$s"; evaluate_all "$NAME"  # D. usul position tokens
  train --model transformer --pitch koma53 --pos --ctx 768 --seed "$s"; evaluate_all "$NAME"   #    same musical context as the runs without POS
done

if [ "$SWEEP" = "1" ]; then                                              # E. regularisation sweep (seed 1, likelihood only)
  for do in 0.2 0.3; do train --model transformer --pitch koma53 --dropout "$do" --seed 1; done
  for do in 0.1 0.2; do train --model transformer --pitch koma53 --layers 4 --d_model 192 --dropout "$do" --seed 1; done
  train --model transformer --pitch koma53 --layers 4 --d_model 128 --seed 1
  for do in 0.3 0.5; do train --model lstm --pitch koma53_rel --dropout "$do" --seed 1; done
fi
python scripts/05_summarize_runs.py
