# Result files

One directory per trained configuration (`<model>-<pitch>-<conditioning>[-options]-s<seed>`).
Checkpoints (`best.pt`) are not in the repository.

| File | Content |
|---|---|
| `config.json`, `vocab.json` | training arguments, vocabulary and karar table |
| `metrics.json` | held-out likelihood (validation, test, rare-makam pool) and training history |
| `gen_prompt16[_constrained].json` | scenario 1, autocomplete |
| `gen_prompt0_full[_constrained]*.json` | scenario 2, composition from scratch |
| `*_samples.jsonl` | last three measures of every generated and real piece, plus 20 full generated pieces |

Suffixes of scenario-2 files: `_end` corrected ending rule (last measure may stop early; default in
all later files), `_cpN` loop penalty N, `_selN_kK` ending selection with N candidates over the
last K measures, `_mixA_B` blend with the LSTM (its weight in the body and in the last 4 measures).

## Sessions

Each GPU session pushed its files in one commit. A new pod retrains the models, so files in the
same directory can come from different copies of a model (same seed, slightly different weights).
`reports/results/generation.csv` lists session and copy for every file.

| Session | Commit | Model copy | What was run |
|---|---|---|---|
| round3 | `42dbf58` | A | full grid on the corrected data (both scenarios, free and constrained) |
| karar | `39426c1` | B | duration-first order; base scenario-2 files of the countdown models regenerated with saved endings |
| karar2 | `fe7220f` | B | loop penalty, corrected ending rule |
| karar3 | `db10801` | C | ending selection |
| karar4 | `7430494` | D | Transformer + LSTM blending |

`archive_v1` and `archive_v2` hold earlier rounds with known issues described in
`reports/baseline_results_v1.md` and `reports/baseline_results_v2.md`.
