# OMR preliminary study: symbol detection

Run of 22 November 2026 (`scripts/run_omr.sh`, commit a39ae7a, one seed, RTX 5090, about 30 minutes in total). Source files: `runs/omr/*/results.json`.

## Setup

- Labels are automatic: symbol class and box are read from the SymbTr PDFs (embedded Mus2 font). No manual annotation.
- 500 training pages (2,930 tiles of 1024 px at 200 dpi), 60 validation pages, 196 test pages; pieces do not cross sets.
- Detector: YOLO11s (Ultralytics, AGPL-3.0), 30 epochs, no architecture changes.
- `clean`: trained on rendered tiles. `mixed`: the same tiles plus a synthetically degraded copy (blur, shadow, low contrast, noise, JPEG).
- Real captures: 15 test pages printed, then scanned (300 dpi) and photographed by hand with a phone. Labels carried over by registration (`src/register.py`, error about 1 px on scans, 1 to 2 px on photos).
- Recall and precision are at IoU 0.5, at the confidence threshold chosen per class by the evaluation tool, weighted by how often each class occurs.

## Results

| Test set | Model | Recall | Precision | mAP50 | mAP50-95 | Accidentals recall | Boxes |
|---|---|---|---|---|---|---|---|
| Clean renders (196 test pages) | clean | 99.7% | 99.6% | 0.953 | 0.873 | 99.5% | 79,540 |
| Clean renders (196 test pages) | mixed | 99.8% | 99.7% | 0.975 | 0.920 | 99.7% | 79,540 |
| Flatbed scans (15 printed pages) | clean | 98.4% | 98.9% | 0.965 | 0.842 | 99.5% | 7,320 |
| Flatbed scans (15 printed pages) | mixed | 98.9% | 99.0% | 0.969 | 0.833 | 99.5% | 7,320 |
| Phone photos (15 printed pages) | clean | 20.4% | 93.4% | 0.327 | 0.248 | 11.0% | 11,688 |
| Phone photos (15 printed pages) | mixed | 97.6% | 98.6% | 0.914 | 0.662 | 95.7% | 11,688 |

mAP is the unweighted mean over classes, so it is pulled down by classes with a handful of examples.

### Accidentals, `mixed` model (recall / precision, number of boxes)

| Font code | Training boxes | Clean test | Scanner | Phone |
|---|---|---|---|---|
| 0x58 | 6,378 | 1.00 / 1.00 (n=2284) | 1.00 / 0.99 (n=182) | 0.99 / 0.99 (n=244) |
| 0x50 | 2,858 | 1.00 / 1.00 (n=1298) | 0.99 / 0.99 (n=147) | 0.95 / 1.00 (n=211) |
| 0x6E | 3,818 | 1.00 / 1.00 (n=1046) | 1.00 / 0.99 (n=58) | 0.93 / 0.98 (n=120) |
| 0x51 | 2,652 | 1.00 / 1.00 (n=947) | 0.99 / 0.99 (n=124) | 0.93 / 0.99 (n=150) |
| 0x57 | 741 | 0.99 / 0.97 (n=158) | 1.00 / 0.99 (n=33) | 0.95 / 0.99 (n=40) |
| 0x59 | 336 | 1.00 / 1.00 (n=166) | 1.00 / 0.96 (n=12) | 1.00 / 0.96 (n=15) |
| 0x55 | 295 | 0.98 / 0.98 (n=62) | 1.00 / 0.96 (n=6) | 1.00 / 0.81 (n=12) |
| 0x52 | 309 | 1.00 / 0.98 (n=121) | 1.00 / 0.96 (n=28) | 0.95 / 0.99 (n=37) |
| 0x56 | 108 | 1.00 / 1.00 (n=15) | 1.00 / 0.93 (n=4) | 1.00 / 0.95 (n=5) |
| 0x04 | 28 | not in set | not in set | not in set |
| 0x03 | 14 | not in set | not in set | not in set |

Code names are provisional (see `reports/figures/glyph_atlas.png`); the grouping is in `scripts/08_train_omr.py`.

## Findings

1. On this engraving style, symbol detection is close to solved with a standard detector and automatic labels: 99.8% recall on clean pages, 98.9% on scans, 97.6% on hand-held phone photos.
2. Comma accidentals are not a weak point for detection here: 99.7% / 99.5% / 95.7% recall (clean / scanner / phone).
3. Synthetic degradation is what makes photos work: phone recall goes from 20.4% (clean training) to 97.6% (mixed training). Scans are fine either way (98.4% vs 98.9%).
4. On six pages from the group without automatic labels, the detector produces plausible detections (`runs/omr/mixed/figs/nofont__*.jpg`); this is visual evidence only.

## Limits

- One engraving style (Mus2), one printer, one scanner, one phone, 15 captured pages. Other editions and handwritten scores are untested.
- Detection only. Pitch (staff position), duration, and reading order are not produced; that is the notation assembly stage.
- 61 of the 85 symbol codes occur in the test pages; 5 rare accidental codes have 28 or fewer training boxes or none and are not evaluated.
- Weakest classes on photos: augmentation dots (0xAB, 0xE2: about 92 to 94% recall), double dots and trills with very few examples.
- Boxes of very small symbols were enlarged to 10 px, and phone labels carry 1 to 2 px of alignment error, so mAP50-95 on photos (0.66) understates localisation quality.
- One seed; the validation split has the known near-duplicate issue (see `preliminary_results.md`), which does not affect the test and capture sets.

## Second run, 23 November 2026: a second engraving style

Run `mixed_v2` (`TAG=_v2 CLEAN=0 bash scripts/run_omr.sh`, commit 418000f, one seed, 24.5 minutes). Same settings as `mixed`, but the class list now has 102 codes, and the dataset has a new test set `cff`: 200 pages typeset with an older Mus2 version (2007), which differ in glyph design and layout and were not used for training.

| Test set | Recall | Precision | Accidentals recall | Boxes |
|---|---|---|---|---|
| Clean renders, training style (196 pages) | 99.8% | 99.6% | 99.7% | 79,540 |
| Clean renders, second style, unseen in training (200 pages) | 97.9% | 98.2% | 99.7% | 85,791 |
| Flatbed scans (15 pages) | 99.0% | 99.2% | 99.9% | 7,320 |
| Phone photos (15 pages) | 97.1% | 98.8% | 95.3% | 11,688 |

- On the second style, 911 boxes (1.1%) belong to 17 symbol classes that never occur in the training style (the largest is 0xE0 with 637 boxes); the detector cannot find these by construction. On the classes it has seen, recall is 99.0% and precision 99.3%.
- Accidental precision on the second style is 96.3%; the 5-comma flat (0x50) has precision 0.77, so some other symbol is being read as that flat.
- mAP50 on the second style is 0.68 because the unseen classes count as zero.
- Run-to-run variation: scans and photos differ from the first run by up to 0.5 points (photos 97.6% then 97.1%).
- Both styles are Mus2 output. Scores from other publishers are still untested, and the second style was tested on clean renders only.
