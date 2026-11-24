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
| `src/generation.py` | batched sampling with optional usul-constrained decoding |
| `scripts/04_evaluate_generation.py` | generation metrics for both usage scenarios, with reference values from real music |
| `scripts/run_baselines.sh` | full experiment grid and regularisation sweep |
| `scripts/05_summarize_runs.py` | collects the main grid → `reports/baseline_results.md` |
| `scripts/run_karar.sh` … `run_karar4.sh` | follow-up experiments on piece endings (token order, loop penalty, ending selection, blending) |
| `scripts/06_collect_results.py` | flattens every result file into `reports/results/*.csv` with its origin |
| `src/register.py` | registers a scan or photo of a printed page to the PDF, to carry the automatic labels over |
| `scripts/07_export_omr_dataset.py` | symbol-detection dataset from the labelled PDFs and from `data/scans` → `omr_data/` |
| `scripts/08_train_omr.py` | trains and evaluates the baseline symbol detector → `runs/omr/<name>/results.json` |
| `scripts/run_omr.sh` | the OMR preliminary study end to end |
| `scripts/run_speed.sh` | speed of composition and of page recognition on this machine (GPU, then CPU) |
| `feasibility/vexflow/` | feasibility test of the VexFlow engraving library for the editor (own README) |

## Usage

```
pip install -r requirements.txt
python scripts/01_data_report.py
python scripts/02_pdf_symbol_inventory.py
```

## Baseline experiments (GPU)

```
python scripts/03_train_baseline.py --smoke     # 20-second pipeline check, CPU is fine
SEEDS="1" bash scripts/run_baselines.sh         # single-seed pass first
bash scripts/run_baselines.sh                   # seeds 1 2 3
```

Sequence layout: `BOS MAKAM:x USUL:y FORM:z [METER:n/d] ( [BAR [REM:k]] [POS:k] [TIE] P:<pitch> D:<num>/<den> )* EOS`.

- Pitch encodings: `tet12` (12-TET, lossy), `aeu` (AEU note names), `koma53` (absolute 53-comma),
  `koma53_rel` (53-comma relative to the makam's karar).
- `--pos` adds POS tokens: duration elapsed in the current measure on an eighth-note grid.
- `--meter` adds the notated measure length of the piece to the prefix; `--rem` adds a countdown of
  the measures remaining after every bar line, which also lets the user choose the piece length.
- Every row that occupies time is used as a note or rest: regular notes (`Kod = 9`) and notes stored
  under ornament codes (tremolo, trill, mordent, glissando, ...). The ornament itself is not modelled
  yet. Grace notes without their own duration and marker rows are skipped.
- A note held across a bar line is split there into tied parts (`TIE`), as in a printed score. With
  both steps, 97% of the measures of real pieces match the piece's meter.

The grid in `run_baselines.sh` covers pitch encoding, architecture, conditioning, POS tokens, meter
and countdown
(`SWEEP=1` adds a regularisation sweep). First-round results are archived in `runs/archive_v1/`
and `reports/baseline_results_v1.md`. Every main run is evaluated for the two usage scenarios of the editor, each
with free and usul-constrained decoding (`src/generation.py`):

1. **Autocomplete:** continue a test piece from its first 16 notes.
2. **Composition from scratch:** only makam, usul and form are given; the model writes a whole piece.

Likelihood is also reported on the `few_shot` pool, i.e. makams that never occur in training.
Checkpoints (`runs/*/best.pt`) are git-ignored; do not commit them.

## Results

- `reports/preliminary_results.md`: findings for the proposal, with an audit section listing known issues.
- `reports/results/`: all numbers as CSV (`likelihood`, `generation`, `reference`, `replicates`).
- `reports/baseline_results.md`: automatic tables of the main grid; `_v1`, `_v2`: archived rounds.
- `runs/README.md`: what each result file is and which session produced it.
- `reports/data_report.md`, `reports/pdf_symbol_report.md`: data statistics and the OMR labelling feasibility study.
