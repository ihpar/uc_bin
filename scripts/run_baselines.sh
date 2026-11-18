#!/usr/bin/env bash
# Baseline grid for the preliminary study. Run on a GPU machine from the repo root:
#   bash scripts/run_baselines.sh            # 3 seeds
#   SEEDS="1" bash scripts/run_baselines.sh  # quick single-seed pass
# Finished runs are skipped, so the script can be restarted.
set -euo pipefail
SEEDS=${SEEDS:-"1 2 3"}

run () {   # model pitch cond seed
  local name="$1-$2-$3-s$4"
  if [ -f "runs/$name/metrics.json" ]; then echo "skip $name"; else
    python scripts/03_train_baseline.py --model "$1" --pitch "$2" --cond "$3" --seed "$4"
  fi
  [ -f "runs/$name/generation_metrics.json" ] || python scripts/04_evaluate_generation.py "runs/$name"
}

for s in $SEEDS; do
  for pitch in tet12 aeu koma53 koma53_rel; do      # A. pitch-encoding ablation (Transformer)
    run transformer "$pitch" prefix "$s"
  done
  run lstm koma53_rel prefix "$s"                    # B. architecture baseline
  run transformer koma53_rel none "$s"               # C. conditioning ablation
done
python scripts/05_summarize_runs.py
