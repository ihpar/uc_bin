#!/usr/bin/env bash
# Second look at piece endings: generation only, on the two pitch-first countdown models.
#   bash scripts/run_karar2.sh
# Tests (a) the corrected ending rule of constrained decoding (the last measure may stop
# early) and (b) a penalty on literally repeating a recent measure. Checkpoints are reused
# when present, otherwise the two models are retrained (about 10 minutes).
set -euo pipefail
SEED=${SEED:-1}
T="--model transformer --pitch koma53 --pos --ctx 768 --meter --rem --seed $SEED"
L="--model lstm --pitch koma53 --dropout 0.5 --pos --meter --rem --seed $SEED"
for ARGS in "$T" "$L"; do
  NAME=$(python scripts/03_train_baseline.py $ARGS --print_name)
  [ -f "runs/$NAME/best.pt" ] || python scripts/03_train_baseline.py $ARGS
  for CP in 0 2 4; do
    SUF="_end$([ "$CP" = "0" ] || echo "_cp$CP")"
    [ -f "runs/$NAME/gen_prompt0_full_constrained${SUF}.json" ] || python scripts/04_evaluate_generation.py "runs/$NAME" --prompt 0 --full --constrain --copy_penalty "$CP" --suffix "$SUF" --save_samples
    [ "$CP" = "0" ] || [ -f "runs/$NAME/gen_prompt0_full${SUF}.json" ] || python scripts/04_evaluate_generation.py "runs/$NAME" --prompt 0 --full --copy_penalty "$CP" --suffix "$SUF" --save_samples
  done
done
python - <<'PY'
import glob, json, os
keys = ("ends_on_karar", "karar_in_last_measure", "repeated_measures", "scale_overlap", "bar_validity_meter", "eos_rate", "length_ratio", "copy_rate")
for f in sorted(glob.glob("runs/*rem*-s1/gen_prompt0_full*.json")):
    j = json.load(open(f)); g = j["generated"]
    print(os.path.basename(os.path.dirname(f))[:28].ljust(28), os.path.basename(f)[11:-5].ljust(28),
          {k: round(g[k], 3) for k in keys if g.get(k) is not None})
    if g.get("pitch_by_tenth"): print(" " * 58, "contour", g["pitch_by_tenth"])
print("real", {k: round(j["reference"][k], 3) for k in keys if j["reference"].get(k) is not None}, j["reference"].get("pitch_by_tenth"))
PY
