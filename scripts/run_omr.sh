#!/usr/bin/env bash
# OMR preliminary study: automatic labels -> symbol detector -> real scans and photos.
#   bash scripts/run_omr.sh
# 1. exports the detection dataset to omr_data/ (about 10 minutes, CPU only)
# 2. trains the detector on clean + synthetically degraded tiles ("mixed")
# 3. trains the same detector on clean tiles only ("clean"), to measure what the
#    degradation buys on real captures. Skip with CLEAN=0.
# Safe to stop and restart: finished steps are skipped and training resumes.
# Results to push: runs/omr/*/results.json and runs/omr/*/figs/
set -euo pipefail
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-16}
PAGES=${PAGES:-500}
CLEAN=${CLEAN:-1}
TAG=${TAG:-}          # suffix for the run names, so that an earlier run is not overwritten

python -c "import ultralytics" 2>/dev/null || pip install -q ultralytics
python -c "import cv2" 2>/dev/null || { pip uninstall -y -q opencv-python; pip install -q opencv-python-headless; }
python -c "import pymupdf, fontTools" 2>/dev/null || pip install -q pymupdf fonttools

[ -f omr_data/meta.json ] || python scripts/07_export_omr_dataset.py --train_pages "$PAGES"
python scripts/08_train_omr.py --data mixed --name "mixed$TAG" --epochs "$EPOCHS" --batch "$BATCH"
if [ "$CLEAN" = "1" ]; then
  python scripts/08_train_omr.py --data clean --name "clean$TAG" --epochs "$EPOCHS" --batch "$BATCH"
fi

python - <<'PY'
import glob, json
for f in sorted(glob.glob("runs/omr/*/results.json")):
    r = json.load(open(f))
    print(f"\n{r['name']}  ({r.get('epochs_run', '?')} epochs, {r['train_minutes']} min)")
    for tag, s in r["sets"].items():
        g = s["groups"].get("accidentals", {})
        print(f"  {tag:8s} mAP50 {s['map50']:.3f}  recall {s['weighted_recall']:.3f}  precision {s['weighted_precision']:.3f}"
              f"  accidentals recall {g.get('recall', float('nan')):.3f}")
PY
