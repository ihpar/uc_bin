#!/usr/bin/env bash
# Ending selection: generation only, on the two pitch-first countdown models.
#   bash scripts/run_karar3.sh
# The piece is generated up to its last K measures; N endings are sampled and the most
# probable one that ends on the karar is kept. Checkpoints are reused when present,
# otherwise the two models are retrained (about 10 minutes).
set -euo pipefail
SEED=${SEED:-1}
T="--model transformer --pitch koma53 --pos --ctx 768 --meter --rem --seed $SEED"
L="--model lstm --pitch koma53 --dropout 0.5 --pos --meter --rem --seed $SEED"

sel () {   # run-name, N, K, copy penalty, extra flags
  local name="$1" n="$2" k="$3" cp="$4"; shift 4
  local suf="_sel${n}_k${k}$([ "$cp" = "0" ] || echo "_cp$cp")"
  local tag="gen_prompt0_full$([[ " $* " == *" --constrain "* ]] && echo _constrained)${suf}"
  [ -f "runs/$name/$tag.json" ] || python scripts/04_evaluate_generation.py "runs/$name" --prompt 0 --full "$@" \
      --select "$n" --select_measures "$k" --copy_penalty "$cp" --suffix "$suf" --save_samples
}

for ARGS in "$T" "$L"; do
  NAME=$(python scripts/03_train_baseline.py $ARGS --print_name)
  [ -f "runs/$NAME/best.pt" ] || python scripts/03_train_baseline.py $ARGS
  CP=0; [[ "$ARGS" == *transformer* ]] && CP=2       # the Transformer needs the loop penalty
  sel "$NAME" 8 4 "$CP" --constrain
  sel "$NAME" 16 8 "$CP" --constrain
  sel "$NAME" 8 4 "$CP"
done
python - <<'PY'
import glob, json, os
keys = ("ends_on_karar", "karar_in_last_measure", "final_note_dur", "repeated_measures", "scale_overlap", "bar_validity_meter", "eos_rate", "length_ratio")
for f in sorted(glob.glob("runs/*rem*-s1/gen_prompt0_full*_sel*.json")):
    j = json.load(open(f)); g = j["generated"]
    print(os.path.basename(os.path.dirname(f))[:12].ljust(12), os.path.basename(f)[11:-5].ljust(30), j["selection"])
    print(" " * 43, {k: round(g[k], 3) for k in keys if g.get(k) is not None})
    print(" " * 43, "contour", g["pitch_by_tenth"])
print("real", {k: round(j["reference"][k], 3) for k in keys if j["reference"].get(k) is not None}, j["reference"]["pitch_by_tenth"])
PY
