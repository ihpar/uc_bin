# VexFlow feasibility test

Question: can the score editor be built on the open-source engraving library
VexFlow, or does it need its own drawing engine?

Tested with VexFlow 5.9.0 (MIT licence) and its Bravura font on two setups:
Firefox 157 on a 2020 MacBook Air (M1, 16 GB), 24 November 2026, numbers in
`results_firefox.json`; and headless Chrome 153 on a 2-core cloud machine,
22 November 2026, numbers in `results.json`. The table below and the figures in
the text are from the MacBook run; the cloud run is about twice as slow.
Drawing, selection, playback and sound were also checked by hand in Firefox.
Pictures are in `screenshots/` (Chrome run). Firefox rounds its timer to 1 ms.

## Result

| Requirement | Result |
|---|---|
| AEU comma accidentals on notes | All eight glyphs draw (`screenshots/accidentals.png`). VexFlow has short codes for only four; the other four, including the koma flat (the most frequent in SymbTr), work by passing the SMuFL glyph directly. |
| Numbered comma accidentals (Mus2 style) | Work: a flat or sharp followed by the comma count, on notes and in key signatures (`screenshots/numbered_commas.png`). The page switches between the two notations; every note keeps both spellings, as SymbTr does. Placement of the digit needs tuning. |
| Makam key signatures | Work through the custom ordered key signature (`KeySignature.setOrdered`), read from the mu2 files. Spacing of the signs needs tuning. |
| Usul time signatures | Work (8/8, 10/8, 16/8, 64/4 tested). |
| Long usuls (a measure wider than a line) | Not handled by VexFlow. Solved here by cutting the measure at note boundaries and drawing dotted bar lines. |
| Click to select a note | Works: every note is an SVG group; one click handler on the score finds it. |
| Edit and redraw | Works with a local redraw: about 5 ms per edit (median), 28 ms worst case. |
| Playback cursor and note highlight | Work: during playback only a class and a position change, nothing is redrawn (`screenshots/playback.png`). |
| Microtonal playback | Works with the browser's audio engine: frequency = 440 Hz x 2^(commas from A4 / 53), so one comma is 22.64 cents, as in Mus2. Playback uses the numbered spelling. Measured pitch error of the synthesized sounds is below 0.05 cents (a comma is 22.6 cents). Two synthesized sounds are selectable; sampled instruments would plug in the same way. |
| Long pieces | Work when only the systems in view are drawn: a 6,273-symbol piece opens in 0.2 s; a 2,443-symbol piece in 0.10 s. |

| Piece | Symbols | Systems | Draw everything | Draw systems in view | One system | Edit (median / max) | Jump scroll (median) |
|---|---|---|---|---|---|---|---|
| hüseyni türkü | 299 | 11 | 64 ms | 33 ms | 2.6 ms | 4 / 5 ms | 1 ms |
| hicaz şarkı | 453 | 14 | 67 ms | 50 ms | 3.1 ms | 5 / 28 ms | 3 ms |
| muhayyerkürdi saz semaisi | 543 | 19 | 83 ms | 50 ms | 2.8 ms | 4 / 6 ms | 4 ms |
| rast kâr-ı nâtık | 2,443 | 81 | 321 ms | 99 ms | 2.5 ms | 4 / 6 ms | 20 ms |
| mahur şarkı | 2,896 | 93 | 398 ms | 123 ms | 2.8 ms | 4 / 13 ms | 26 ms |
| segâh kâr | 6,273 | 202 | 830 ms | 201 ms | 2.7 ms | 5 / 25 ms | 27 ms |

Times are from the start of loading until the browser has painted. "Edit" covers
moving a note, changing its accidental, inserting and removing a note; the
slowest of the four is shown.

## What VexFlow does and what the editor has to own

VexFlow draws one measure well: note heads, stems, beams, ties, tuplets, dots,
accidentals, clef, key and time signature, with correct spacing inside the
measure. Everything around that was written for this test (about 300 lines in
`app.js`) and remains the editor's own code:

- which accidentals to show, given the key signature and earlier notes in the measure
- splitting durations into tied notes, and long usul measures into segments
- line breaking and stretching measures to the line width
- one SVG per system, and drawing only the systems in view
- selection, keyboard editing, and re-flow after an edit

## Not tested

Lyrics, repeats and voltas (in drawing and in playback order), ornaments and grace
notes, section labels, sampled instruments, page layout for printing and PDF export, touch input, Safari, and
undo. The opening time of long pieces is dominated by measuring every measure
once (0.05 to 0.15 s); that can be cached or done in the background.

## Run it

```
cd feasibility/vexflow
npm install
python make_piece.py <SymbTr file name without .txt>    # optional: add a piece
python build.py                                          # -> dist/index.html (open in a browser)
node bench.mjs                                           # -> results.json, screenshots/
node bench.mjs <path to chrome or firefox> <output name> # another browser, another results file
```
