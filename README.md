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

## Usage

```
pip install pymupdf fonttools matplotlib
python scripts/01_data_report.py
python scripts/02_pdf_symbol_inventory.py
```
