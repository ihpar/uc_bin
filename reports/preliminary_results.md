# Preliminary results: makam-aware symbolic music modelling

Summary of the third experiment round for use in the proposal. Full tables: `baseline_results.md`. Earlier rounds, kept for the record with their known issues: `baseline_results_v1.md`, `baseline_results_v2.md`.

## Setup

- **Data:** SymbTr v3, the 38 makams with at least 20 pieces (2,631 pieces, 87% of the collection). Piece-level split within each makam: 2,070 train, 320 validation, 241 test. Pieces sharing title and composer stay in one set.
- **Events:** every row that occupies time is a note or rest (1.14M sounding notes); notes held across a bar line are split into tied parts. After both steps 97% of real measures match the meter of their piece.
- **Models:** decoder-only Transformer (6 layers, d=256, 4 heads, about 5M parameters) and a 2-layer LSTM (hidden 512, about 4M parameters). AdamW, learning rate 3e-4 with warm-up and cosine decay, early stopping on validation likelihood.
- **Scenarios:** (1) autocomplete: continue a test piece from its first 16 notes for 64 notes; (2) composition from scratch: only makam, usul, form (and meter, target length where the model supports them) are given and the model writes a whole piece.
- **Caveat:** one training run per configuration. Differences of a few hundredths of a bit or a few percentage points are within run-to-run variation; the findings below rest on much larger gaps.

## Findings

### 1. A 12-tone representation cannot express makam pitch

| Pitch encoding | Off-scale notes (scenario 1) | Scale overlap |
|---|---|---|
| 12-TET | 13.0% | 0.733 |
| AEU note names | 5.8% | 0.817 |
| 53-comma (Koma53) | 0.6% | 0.866 |
| *Real music* | 1.4% | 0.908 |
| *Real music rounded to 12-TET* | 14.4% | 0.762 |

12-TET has a slightly better likelihood (1.99 against 2.07 bits per note) only because its target is coarser. A karar-relative variant of Koma53 was tested in round 2 and gave no advantage, including on makams unseen in training.

### 2. A Transformer needs explicit position in the usul cycle

| Model (Koma53, scenario 1, free decoding) | Valid measures | Test bits/note |
|---|---|---|
| Transformer | 56.1% | 2.07 |
| Transformer + position tokens | 96.0% | 1.95 |
| Transformer + position tokens + meter | 98.5% | 1.98 |
| LSTM | 96.8% | 3.10 |
| *Real music* | 99.5% | |

Position tokens (duration elapsed in the measure) give the best likelihood of all runs and close the measure-length gap. The LSTM keeps count without them. Usul-constrained decoding reaches 100% for any model when the meter is known, with no loss in pitch metrics.

### 3. Conditioning is what makes composition from scratch possible

Without makam/usul/form labels, scenario 2 output has a scale overlap of 0.61 and 18% off-scale notes; with them 0.83 and 0.9%. In scenario 1 the labels matter little because the prompt reveals the makam.

### 4. Length can be controlled; cadence cannot yet

| Model (scenario 2, free decoding) | Ended by model | Length / target | Ends on karar |
|---|---|---|---|
| Transformer | 40% | 2.97 | 34% |
| Transformer + position, meter, countdown | 93% | 0.99 | 21% |
| LSTM | 85% | 1.98 | 68% |
| LSTM + position, meter, countdown | 95% | 0.99 | 49% |
| *Real music* | | 1.00 | 97% |

Without a countdown of the remaining measures the models do not know when to stop. With it they end at the requested length, but they do not prepare the ending: only 21–49% of pieces finish on the karar, against 97% of real pieces. The baseline has no notion of long-range form.

### 5. Capacity is not the bottleneck

In the round-2 sweep, Transformers from 0.9M to 5M parameters and dropout from 0.1 to 0.3 all landed within 0.06 bits of each other. All runs stop improving on validation after about 5,000 steps. Gains have to come from representation, structure and data, not from model size.

### 6. Likelihood and the generation metrics disagree about the LSTM

The Transformer predicts held-out music far better (2.0 against 3.1 bits per note), yet on the coarse generation metrics above the LSTM is on par, and better at ending pieces. The LSTM was tuned only for dropout, so part of the likelihood gap may be tuning. The generation metrics check pitch inventory, measure length and endings; they do not measure melodic quality. Separating the models needs finer metrics and expert listening tests.

## Open problems that the project addresses

1. **Form and cadence:** ending on the karar, phrase structure, seyir. Candidate methods: section and phrase tokens from the SymbTr annotations, longer countdown horizon, auxiliary objectives.
2. **Usul changes inside a piece** (4.4% of pieces): a meter token at the change point.
3. **Ornaments:** currently the ornamented note is kept and the ornament type is dropped.
4. **Evaluation:** metrics for melodic quality, 3–5 seeds with paired tests, blind expert evaluation.
5. **Rare makams:** outside the scope of composition from scratch; autocomplete degrades gracefully there (2.5 against 2.1 bits per note).
