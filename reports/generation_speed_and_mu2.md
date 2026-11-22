# Composition speed and the mu2 format

Checks of 22 November 2026 for the requirements table.

## Composition speed

`scripts/09_generation_speed.py`, results in `reports/results/generation_speed_cpu.json`.
Measured on a 2-core cloud CPU (Intel Xeon 2.1 GHz), no GPU. Speed depends on the
architecture, context length and batch size, not on the weight values, so the two
models were built from their saved configuration and vocabulary with random weights.

Real music needs 27 tokens per measure (median; 44 at the 90th percentile). A
median test piece has 52 measures and 1,441 tokens.

| Request | Current research code | With cached decoding |
|---|---|---|
| Autocomplete: 3 alternatives of 8 measures (Transformer) | 84 s (137 s for dense measures) | 3.8 s (5.9 s) |
| From scratch: median piece with the Transformer, then 8 candidate endings with the LSTM | 212 s | 12 s |

- The current code re-runs the model over the whole context (up to 768 tokens) for
  every new token: 127 ms per token for one sequence, 388 ms for three.
- Cached decoding keeps the model state between tokens (key/value cache for the
  Transformer, hidden state for the LSTM). The step functions in the script give the
  same outputs as the full pass (difference below 2e-6) and take 3 ms (one sequence)
  to 7 ms (three) per token for the Transformer at a 700-token context, and 2 to 3 ms
  for the LSTM. Sampling and constraint checks add 3 to 9 ms per token.
- The "cached" column is computed from these measured parts. Cached decoding is not
  yet integrated in `src/generation.py`; when the sequence outgrows the context
  window the cache has to be rebuilt, which happens once every 64 tokens at most.
- Not measured on a GPU.

## mu2 files

All 3,000 mu2 files are tab-separated text with 10 columns (3 rows have 11) in the
same encoding as the txt files, and use the same row codes. Some fields carry
surrounding spaces and need trimming.

What mu2 has that the txt files do not:

- the key signature of the piece (code 50, for example `B4b4/F5#4/C5#4` for hicaz)
- repeats and voltas, stored folded (markers such as `(`, `(1`, `(2`, `[`, `:`; codes 21 and 35)
- beam grouping per measure (code 14, for example `22221`)
- two lyric lines, tempo, lyricist, genre

Comparison of the note sequences (code, pitch name, duration) with the txt files:

| Group | Files |
|---|---|
| Identical to the txt file | 647 |
| Different, file has repeat marks (txt is the unfolded version) | 2,258 |
| Different, no repeat marks (tied notes are separate rows in mu2, merged in txt) | 95 |

So reading mu2 is easy; a faithful import additionally needs repeat expansion for
playback, which is not written yet.

Accidentals: mu2 note names and key signatures use numbered commas, not only the
eight AEU signs. Counts over all note rows: #4 57,681; b1 33,727; b5 33,702;
b4 31,547; b2 24,607; #3 20,207; b3 12,455; #5 5,701; #1 2,268; #2 594; others 23.
The txt files carry both spellings (Nota53 and NotaAE). The editor therefore needs
a decision on showing AEU signs, numbered commas, or both.
