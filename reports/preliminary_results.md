# Preliminary results: makam-aware symbolic music modelling

Summary of the third experiment round and its follow-ups, for use in the proposal. Every number below was checked against the result files on 2026-11-21 (see "Audit" at the end). Flat tables with the origin of every file: `results/` (`likelihood.csv`, `generation.csv`, `reference.csv`, `replicates.csv`, built by `scripts/06_collect_results.py`). Automatic summary of the main grid: `baseline_results.md`. Earlier rounds, kept for the record with their known issues: `baseline_results_v1.md`, `baseline_results_v2.md`.

## Setup

- **Data:** SymbTr v3. The evaluation targets the 38 makams with at least 20 pieces (87% of the collection). Models are trained on 2,067 pieces, validated on 320 and tested on 241; 240 of the 241 test pieces belong to the 38 makams and every one of the 38 is represented. Scenario 1 uses the 239 test pieces with at least 80 notes. See "Audit" for two flaws of this split that do not affect the test set.
- **Events:** every row that occupies time is a note or rest (1.14M sounding notes); notes held across a bar line are split into tied parts. After both steps 97% of real measures match the meter of their piece.
- **Models:** decoder-only Transformer (6 layers, d=256, 4 heads, about 5M parameters) and a 2-layer LSTM (hidden 512, about 4M parameters). AdamW, learning rate 3e-4 with warm-up and cosine decay, early stopping on validation likelihood.
- **Scenarios:** (1) autocomplete: continue a test piece from its first 16 notes for 64 notes; (2) composition from scratch: only makam, usul, form (and meter, target length where the model supports them) are given and the model writes a whole piece.
- **Caveat:** one random seed per configuration. The two countdown models were retrained four times with that seed on different machines: test likelihood varied by 0.02 bits per note (Transformer 1.984–2.005, LSTM 3.165–3.183). Variation across seeds is not measured and is probably larger. With 241 test pieces a percentage near 50% has a standard error of about 3 points. The findings below rest on much larger gaps.

## Findings

### 1. A 12-tone representation cannot express makam pitch

| Pitch encoding | Off-scale notes (scenario 1) | Scale overlap |
|---|---|---|
| 12-TET | 13.0% | 0.733 |
| AEU note names | 5.8% | 0.817 |
| 53-comma (Koma53) | 0.6% | 0.866 |
| *Real music* | 1.4% | 0.908 |
| *Real music rounded to 12-TET* | 14.4% | 0.762 |

12-TET has a slightly better likelihood (1.99 against 2.07 bits per note) only because its target is coarser. A karar-relative variant of Koma53 was tested in round 1 (`baseline_results_v1.md`) and gave no advantage: 2.27 against 2.26 bits per note on the test set, and 2.86 against 2.67 on the pool of rare makams.

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

Without a countdown of the remaining measures the models do not know when to stop. With it they end at the requested length, but they do not prepare the ending. Models that choose their own ending do better on the pieces they manage to end: assuming pieces cut off at the token limit end on the karar only by chance (0–15%), the rate among self-ended pieces is roughly 65–85% for the Transformer and 78–80% for the LSTM. This is an estimate; the endings of those runs were not saved. Models that are told when to end do not steer to the karar in time.

Three follow-up experiments on the countdown models (all 241 test pieces, scenario 2):

| Hypothesis | Test | Result |
|---|---|---|
| The model picks the pitch before it knows the note is the last one | Write the duration before the pitch | No change (Transformer 20%, LSTM 49%) |
| The decoding constraint forces a full final measure, while real pieces mostly end on a short one | Let the last measure stop early | Small gain under constraints (Transformer 17% to 27%, LSTM 40% to 45%) |
| The Transformer loops: 39% of its measures repeat one of the two before (real music 3%, LSTM 0.3%) | Penalise literal repetition of a recent measure | Loops disappear (1–6%) and scale overlap rises slightly, but karar endings stay at 24–28% |

So the failure is not a decoding artefact. Two observations describe it:

- **Where pieces end.** The Transformer's most frequent final pitches are the karar (27%), the fifth above it (15%), the fourth (12%) and the octave (8%): notes on which inner phrases come to rest. It produces a phrase ending, not the final one.
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

With the LSTM, selection brings karar endings close to real music with no loss in scale overlap (0.89), measure validity or length. With the Transformer alone it does not. A quarter of its sampled endings reach the karar, but the eight endings sampled for one generated body succeed or fail together (51% of bodies get at least one; independent attempts would give about 93%). Which pieces fail is not fixed: two generations of the same test set with this model agree little more than chance on which pieces end well, and there is no clear pattern by makam or form. Success does fall with piece length: 70% for target pieces that fit in the 768-token window, 53% up to 1,500 tokens, 38% up to 3,000. The LSTM stays at 89–98% across those lengths.

**Combining the two models.** Both models use the same tokens, so their next-token predictions can be blended at every step (log-linear), with one weight for the body and one for the last four measures. All rows: constrained decoding, loop penalty 2, selection among 8 endings.

| LSTM share in body / ending | Single ending on karar | Ends on karar | Repeated measures | Scale overlap | Off-scale |
|---|---|---|---|---|---|
| 0 / 1 (Transformer body, LSTM ending) | 48% | 93% | 3.4% | 0.873 | 0.7% |
| 0.3 / 1 | 50% | 93% | 0.8% | 0.891 | 0.3% |
| 0.5 / 0.5 | 40% | 89% | 0.5% | 0.893 | 0.3% |
| 0.7 / 0.7 | 47% | 89% | 0.3% | 0.897 | 0.6% |
| LSTM alone | 50% | 91% | 0.1% | 0.880 | 0.8% |
| Transformer alone (earlier checkpoint) | 28% | 48% | 5.9% | 0.886 | 0.9% |
| *Real music* | | 97% | 3.2% | 0.922 | 2.2% |

The hand-off row settles where the Transformer fails. The LSTM cadences from a body written entirely by the Transformer as well as from its own (48% against 50% per attempt). The Transformer's bodies are therefore not the obstacle; what it lacks is the ability to write the ending itself, increasingly so in long pieces. The "Transformer alone" row comes from a different trained copy of the model than the other rows. Differences of a few points between the rows with an LSTM ending are within the variation expected from about 240 pieces.

In every combination the mean pitch falls to 19–21 commas above the karar in the last tenth (real music: 19.1), but none shows the rise around three quarters of the piece (real music: 31.2; models: 24–26). The ending is repaired; the seyir across the piece is not.

### 5. Capacity is not the bottleneck

In the round-1 sweep (`baseline_results_v1.md`), Transformers from 0.9M to 5M parameters and dropout from 0.1 to 0.3 all landed within 0.06 bits of each other (2.20–2.26). In the current round every run reaches its best validation score between step 4,500 and 6,000. Gains have to come from representation, structure and data, not from model size.

### 6. Likelihood and the generation metrics disagree about the LSTM

The Transformer predicts held-out music far better (2.0 against 3.1 bits per note), yet on the coarse generation metrics above the LSTM is on par, and clearly better at whole pieces: it does not loop, it descends at the end, and with ending selection it reaches the karar in 93% of pieces against 48–54%. The LSTM was tuned only for dropout, so part of the likelihood gap may be tuning. Its advantage on whole pieces is not explained by longer memory: during generation both models are re-run on a truncated window (768 tokens for the Transformer, 512 for the LSTM), with the conditioning tokens kept in front. The generation metrics check pitch inventory, measure length and endings; they do not measure melodic quality. Separating the models needs finer metrics and expert listening tests.

## Open problems that the project addresses

1. **Form and seyir:** ending on the karar, the rise and descent across a piece, phrase structure. Local conditioning and decoding rules do not solve it (finding 4). Candidate methods: section and phrase tokens from the SymbTr annotations, position-in-piece conditioning, a planned pitch contour as a control signal, auxiliary objectives.
2. **Usul changes inside a piece** (4.4% of pieces): a meter token at the change point.
3. **Ornaments:** currently the ornamented note is kept and the ornament type is dropped.
4. **Evaluation:** metrics for melodic quality, 3–5 seeds with paired tests, blind expert evaluation.
5. **Rare makams:** outside the scope of composition from scratch. Prediction is moderately worse there (2.5 against 2.1 bits per note for the plain Koma53 Transformer). The collection also holds about 160 short pieces of form `seyir`, notated demonstrations of the melodic path of individual makams, which are a resource for problem 1.

## Audit (2026-11-21)

All tables above were recomputed from the JSON files in `runs/` and compared with this text; the likelihood evaluation was checked to score every test token exactly once (242,508 tokens, 112,436 note events for the plain encoding). Corrections made to earlier versions of this report: the karar-relative test and the capacity sweep belong to round 1, not round 2; the share of self-ended pieces that end on the karar was overstated as "80–85%"; one percentage in the list of final pitches; the description of how Transformer failures are distributed over pieces.

**Known issues that remain**

1. **Validation set contains near-duplicates of training pieces.** Pieces were grouped by title and composer to keep versions of one composition together. 72 instrumental interludes (`aranagme`) all carry the title "1" and no composer, were treated as one group and all went to validation; 56 of the 320 validation pieces share at least half of their 8-note sequences with a single training piece. Validation is used only for early stopping, so the test numbers are not affected, but stopping points may be slightly late. The test set is clean: one of 241 pieces has a near-duplicate in training (`hicaz_uzzal--zeybek--aksak----izmir`).
2. **Makam stratification is imperfect for the same reason.** Groups of generically titled pieces by one author span many makams and were assigned as a block. Training therefore includes 55 files from makams outside the 38 (81 makam labels in total), and the `few_shot` pool is not "unseen makams": 149 of its 369 files have a makam absent from training, the rest have a makam with a handful of training pieces. Read `few_shot` as "rare makams".
3. **Mixed checkpoints.** New GPU sessions retrain the models, so the files of the two countdown runs come from four copies (A–D in `results/generation.csv`): scenario-1 files from A, base scenario-2 files and the loop/ending-rule tests from B, ending selection from C, blending from D. `metrics.json` in those two directories is from D. Likelihood differs by at most 0.02 bits between copies (`results/replicates.csv`); generation percentages of different copies should be compared with a margin of a few points.
4. **Token limit in scenario 2.** Generation stops after 3,000 tokens. Ten of the 241 test pieces are longer than that in the countdown encoding, and success on them is lower (60–70% karar endings for the LSTM pipeline against 89–98% otherwise). Models without a countdown mostly run into the limit, so their scenario-2 metrics describe 3,000-token excerpts.
5. **Scenario-2 "constrained" rows of models without a countdown** (in `baseline_results.md`) are not meaningful: the round-3 rule allowed an ending only on a completed measure, and without a countdown these models rarely end (3–20% of pieces).
6. **Measure boundaries come from the txt offsets.** 97% of real measures match the meter of their piece; the rest are mostly pieces with an usul change, for which the single meter token is wrong after the change.
7. **Metrics are coarse.** They check pitch inventory, measure length, repetition, length and the final note. They do not measure melodic quality, and selection-based numbers (ending selection, usul constraint) are correct by construction.

Before the experiments for publication: rebuild the split with content-based duplicate detection and stratification independent of the grouping, raise the token limit, save all generated pieces, and train 3–5 seeds.
