#!/usr/bin/env bash
# Speed of the two services on this machine (GPU if there is one, then CPU).
#   bash scripts/run_speed.sh
# 1. composition: seconds per token and per request      -> reports/results/generation_speed_<cpu|cuda>.json
# 2. recognition: seconds per page for symbol detection  -> reports/results/omr_speed_<cpu|cuda>.json
# Step 2 needs a trained detector. If none is found under omr_work/, a short one
# (10 epochs, about 10 minutes of export plus 10 of training on an RTX 5090) is
# trained under the name "speedcheck"; its accuracy is not reported anywhere.
# Results to push: the two json files.
set -euo pipefail
python -c "import ultralytics" 2>/dev/null || pip install -q ultralytics
python -c "import cv2" 2>/dev/null || { pip uninstall -y -q opencv-python; pip install -q opencv-python-headless; }
python -c "import pymupdf, fontTools" 2>/dev/null || pip install -q pymupdf fonttools

python scripts/09_generation_speed.py

if ! ls omr_work/*/weights/best.pt >/dev/null 2>&1; then
  [ -f omr_data/meta.json ] || python scripts/07_export_omr_dataset.py --train_pages "${PAGES:-500}"
  python scripts/08_train_omr.py --data mixed --name speedcheck --epochs "${EPOCHS:-10}" --batch "${BATCH:-16}"
fi
python scripts/10_omr_speed.py --cpu_threads "${CPU_THREADS:-4}"
