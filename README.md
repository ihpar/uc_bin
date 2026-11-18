# uc_bin

TÜBİTAK 3501 yeniden başvurusu için ön çalışma deposu: SymbTr v3 üzerinde makam duyarlı sembolik modelleme ve optik müzik tanıma.

## İçerik

| Yol | Açıklama |
|---|---|
| `symbtr_v3/` | SymbTr v3 txt ve PDF dosyaları (CC BY 4.0, Karaosmanoğlu; Zenodo 10.5281/zenodo.15470412) |
| `src/symbtr.py` | txt ayrıştırıcı |
| `src/pdf_semboller.py` | PDF'ten otomatik sembol kutusu çıkarımı |
| `scripts/01_veri_raporu.py` | Dağılımlar ve eser bazlı bölme → `reports/veri_raporu.md`, `data/bolmeler.csv` |
| `scripts/02_pdf_sembol_envanteri.py` | Sembol envanteri ve atlas → `reports/pdf_sembol_raporu.md` |

## Çalıştırma

```
pip install pymupdf fonttools matplotlib
python scripts/01_veri_raporu.py
python scripts/02_pdf_sembol_envanteri.py
```
