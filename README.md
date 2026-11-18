# uc_bin

Preliminary work for a TÜBİTAK 3501 resubmission: makam-aware symbolic music modelling and optical music recognition on SymbTr v3.

## Contents

| Path | Description |
|---|---|
| `symbtr_v3/` | SymbTr v3 txt, PDF, mu2 and MusicXML files (CC BY 4.0, Karaosmanoğlu; Zenodo 10.5281/zenodo.15470412) |
| `src/symbtr.py` | txt parser |
| `src/pdf_symbols.py` | automatic symbol bounding boxes from the PDFs |
| `scripts/01_data_report.py` | distributions and piece-level split → `reports/data_report.md`, `data/splits.csv` |
| `scripts/02_pdf_symbol_inventory.py` | symbol inventory and glyph atlas → `reports/pdf_symbol_report.md` |
| `src/tokenizer.py`, `src/dataset.py`, `src/models.py` | event tokenization (4 pitch encodings), data pipeline, Transformer and LSTM language models |
| `scripts/03_train_baseline.py` | trains one model, reports held-out bits per note → `runs/<name>/metrics.json` |
| `scripts/04_evaluate_generation.py` | generation metrics for a run, with reference values from real music |
| `scripts/run_baselines.sh` | full baseline grid (pitch-encoding, architecture and conditioning ablations) |
| `scripts/05_summarize_runs.py` | collects all runs → `reports/baseline_results.md` |

## Usage

```
pip install -r requirements.txt
python scripts/01_data_report.py
python scripts/02_pdf_symbol_inventory.py
```

## Baseline experiments (GPU)

```
python scripts/03_train_baseline.py --smoke     # 20-second pipeline check, CPU is fine
bash scripts/run_baselines.sh                   # 18 runs (6 configurations x 3 seeds)
SEEDS="1" bash scripts/run_baselines.sh         # single-seed pass first
```

Sequence layout: `BOS MAKAM:x USUL:y FORM:z ( [BAR] P:<pitch> D:<num>/<den> )* EOS`.
Pitch encodings: `tet12` (12-TET, lossy), `aeu` (AEU note names), `koma53` (absolute 53-comma),
`koma53_rel` (53-comma relative to the makam's karar). Only rows with `Kod = 9` (regular notes
and rests) are used; ornament rows are ignored in the baseline.
