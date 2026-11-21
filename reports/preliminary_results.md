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
| Transformer + position, meter, countdown | 93% | 0.98 | 22% |
| LSTM | 85% | 1.98 | 68% |
| LSTM + position, meter, countdown | 96% | 0.99 | 52% |
| *Real music* | | 1.00 | 97% |

Without a countdown of the remaining measures the models do not know when to stop. With it they end at the requested length, but they do not prepare the ending. Models that choose their own ending land on the karar in about 80–85% of the pieces they end; models that are told when to end do not steer there in time.

Three follow-up experiments on the countdown models (all 241 test pieces, scenario 2):

| Hypothesis | Test | Result |
|---|---|---|
| The model picks the pitch before it knows the note is the last one | Write the duration before the pitch | No change (Transformer 20%, LSTM 49%) |
| The decoding constraint forces a full final measure, while real pieces mostly end on a short one | Let the last measure stop early | Small gain under constraints (Transformer 17% to 27%, LSTM 40% to 45%) |
| The Transformer loops: 39% of its measures repeat one of the two before (real music 3%, LSTM 0.3%) | Penalise literal repetition of a recent measure | Loops disappear (1–6%) and scale overlap rises slightly, but karar endings stay at 24–28% |

So the failure is not a decoding artefact. Two observations describe it:

- **Where pieces end.** The Transformer's most frequent final pitches are the karar (27%), the fifth above it (15%), the fourth (10%) and the octave (8%): notes on which inner phrases come to rest. It produces a phrase ending, not the final one.
- **Seyir.** Mean pitch in commas above the karar, per tenth of the piece:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Real music | 27.7 | 25.2 | 26.6 | 27.1 | 26.0 | 28.1 | 29.4 | 31.2 | 26.9 | 19.1 |
| LSTM (countdown, constrained) | 25.9 | 24.9 | 25.4 | 26.6 | 24.4 | 25.3 | 26.1 | 26.1 | 26.4 | 20.3 |
| Transformer (countdown, constrained) | 24.4 | 24.4 | 25.5 | 25.6 | 25.5 | 24.8 | 25.0 | 24.8 | 24.8 | 24.3 |

Real pieces rise around three quarters of the way through and descend to the karar. The LSTM reproduces the final descent but not the rise; the Transformer is flat. Neither model has a notion of the large-scale melodic path of a makam.

**Ending selection.** A decoding-time remedy in the spirit of the usul constraint: the piece is generated up to its last measures, several endings are sampled, and the most probable one whose last sounding note is the karar is kept. Every ending is the model's own output.

| Model (countdown, constrained) | Endings sampled, length | Single ending on karar | Pieces with a qualifying ending | Ends on karar |
|---|---|---|---|---|
| LSTM | 8, last 4 measures | 50% | 95% | 93% |
| LSTM | 16, last 8 measures | 51% | 96% | 93% |
| Transformer (loop penalty 2) | 8, last 4 measures | 28% | 51% | 48% |
| Transformer (loop penalty 2) | 16, last 8 measures | 24% | 58% | 54% |
| *Real music* | | | | 97% |

With the LSTM, selection brings karar endings close to real music with no loss in scale overlap (0.89), measure validity or length. With the Transformer it does not: a quarter of its sampled endings reach the karar, but they are concentrated in half of the pieces. If attempts were independent, 8 samples would succeed for about 90% of pieces; the observed 51% means that for the other half the piece is, by its last measures, in a state from which the model does not cadence at all. Selection cannot repair a body that has not been heading home, which again points to large-scale form.

### 5. Capacity is not the bottleneck

In the round-2 sweep, Transformers from 0.9M to 5M parameters and dropout from 0.1 to 0.3 all landed within 0.06 bits of each other. All runs stop improving on validation after about 5,000 steps. Gains have to come from representation, structure and data, not from model size.

### 6. Likelihood and the generation metrics disagree about the LSTM

The Transformer predicts held-out music far better (2.0 against 3.1 bits per note), yet on the coarse generation metrics above the LSTM is on par, and clearly better at whole pieces: it does not loop, it descends at the end, and with ending selection it reaches the karar in 93% of pieces against 48–54%. The LSTM was tuned only for dropout, so part of the likelihood gap may be tuning. The generation metrics check pitch inventory, measure length and endings; they do not measure melodic quality. Separating the models needs finer metrics and expert listening tests.

## Open problems that the project addresses

1. **Form and seyir:** ending on the karar, the rise and descent across a piece, phrase structure. Local conditioning and decoding rules do not solve it (finding 4). Candidate methods: section and phrase tokens from the SymbTr annotations, position-in-piece conditioning, a planned pitch contour as a control signal, auxiliary objectives.
2. **Usul changes inside a piece** (4.4% of pieces): a meter token at the change point.
3. **Ornaments:** currently the ornamented note is kept and the ornament type is dropped.
4. **Evaluation:** metrics for melodic quality, 3–5 seeds with paired tests, blind expert evaluation.
5. **Rare makams:** outside the scope of composition from scratch; autocomplete degrades gracefully there (2.5 against 2.1 bits per note).
