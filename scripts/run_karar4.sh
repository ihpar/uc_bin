#!/usr/bin/env bash
# Combining the two countdown models for composition from scratch (generation only).
#   bash scripts/run_karar4.sh
# The Transformer and the LSTM are blended at every step; the weight of the LSTM is given
# for the body and for the last 4 measures. All runs: constrained decoding, loop penalty 2,
# ending selection with 8 candidates. Checkpoints are reused when present.
set -euo pipefail
SEED=${SEED:-1}
T="--model transformer --pitch koma53 --pos --ctx 768 --meter --rem --seed $SEED"
L="--model lstm --pitch koma53 --dropout 0.5 --pos --meter --rem --seed $SEED"
TN=$(python scripts/03_train_baseline.py $T --print_name); [ -f "runs/$TN/best.pt" ] || python scripts/03_train_baseline.py $T
LN=$(python scripts/03_train_baseline.py $L --print_name); [ -f "runs/$LN/best.pt" ] || python scripts/03_train_baseline.py $L

mixrun () {   # LSTM weight in the body, LSTM weight in the ending
  local suf="_mix${1}_${2}_sel8_k4_cp2"
  [ -f "runs/$TN/gen_prompt0_full_constrained${suf}.json" ] || python scripts/04_evaluate_generation.py "runs/$TN" \
      --prompt 0 --full --constrain --copy_penalty 2 --select 8 --select_measures 4 \
      --mix "runs/$LN" --mix_body "$1" --mix_end "$2" --mix_measures 4 --suffix "$suf" --save_samples
}
mixrun 0 1        # hand-off: Transformer body, LSTM ending
mixrun 0.3 1      # mostly Transformer body, LSTM ending
mixrun 0.5 0.5    # even blend throughout
mixrun 0.7 0.7    # mostly LSTM throughout
# the same loop penalty on the LSTM alone, so the comparison is like for like
[ -f "runs/$LN/gen_prompt0_full_constrained_sel8_k4_cp2.json" ] || python scripts/04_evaluate_generation.py "runs/$LN" \
    --prompt 0 --full --constrain --copy_penalty 2 --select 8 --select_measures 4 --suffix _sel8_k4_cp2 --save_samples

python - <<'PY'
import glob, json, os
keys = ("ends_on_karar", "repeated_measures", "scale_overlap", "off_scale_rate", "copy_rate", "final_note_dur", "eos_rate", "length_ratio")
for f in sorted(glob.glob("runs/*rem*-s1/gen_prompt0_full_constrained*sel8_k4*.json")):
    j = json.load(open(f)); g = j["generated"]
    print(os.path.basename(os.path.dirname(f))[:11].ljust(11), os.path.basename(f)[28:-5].ljust(26), {k: round(v, 3) for k, v in j["selection"].items()})
    print(" " * 38, {k: round(g[k], 3) for k in keys if g.get(k) is not None})
    print(" " * 38, "contour", g["pitch_by_tenth"])
print("real", {k: round(j["reference"][k], 3) for k in keys if j["reference"].get(k) is not None}, j["reference"]["pitch_by_tenth"])
PY
