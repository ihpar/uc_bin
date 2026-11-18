"""Runs automatic symbol extraction on all SymbTr PDFs; writes the inventory, a glyph
atlas and an example overlay image. Run: python scripts/02_pdf_symbol_inventory.py"""
import os, sys, collections, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pymupdf
from src.pdf_symbols import page_symbols, page_drawings, font_kind
from src.symbtr import PDF_DIR

ROOT = os.path.join(os.path.dirname(__file__), "..")
files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
codes = collections.Counter()
example = {}                      # code -> (file, page, bbox)
n_pages = n_sym = unmapped = 0
full_files = full_pages = full_syms = 0
draw = collections.Counter(); files_with = collections.Counter(); sizes = collections.Counter()
kinds, kind_pages = collections.Counter(), collections.Counter()
rows = []
for f in files:
    doc = pymupdf.open(os.path.join(PDF_DIR, f))
    fc = collections.Counter(); got = False; un = 0
    kind = font_kind(doc); kinds[kind] += 1; kind_pages[kind] += len(doc)
    for i, page in enumerate(doc):
        n_pages += 1
        syms = page_symbols(doc, page)
        if syms: got = True
        for s in syms:
            if s["bbox"] is None: unmapped += 1; un += 1; continue
            codes[s["code"]] += 1; fc[s["code"]] += 1; n_sym += 1
            sizes[round(s["size"])] += 1
            if s["code"] not in example or (example[s["code"]][0] != f and codes[s["code"]] == 50):
                example[s["code"]] = (f, i, s["bbox"])
        for d in page_drawings(page): draw[d["kind"]] += 1
    if got and un == 0:
        full_files += 1; full_pages += len(doc); full_syms += sum(fc.values())
    for c in fc: files_with[c] += 1
    rows.append((f, len(doc), sum(fc.values()), un, kind))
    doc.close()

# --- glyph atlas: one example crop per code
atlas = pymupdf.open(); cols, cw, chh = 10, 56, 74
order = [c for c, _ in codes.most_common()]
pg = atlas.new_page(width=cols * cw + 10, height=(len(order) // cols + 1) * chh + 10)
cache = {}
for k, c in enumerate(order):
    f, i, bb = example[c]
    if f not in cache: cache[f] = pymupdf.open(os.path.join(PDF_DIR, f))
    r = pymupdf.Rect(bb) + (-3, -3, 3, 3)
    x, y = 5 + (k % cols) * cw, 5 + (k // cols) * chh
    sc = min((cw - 8) / r.width, (chh - 26) / r.height, 2.5)
    tgt = pymupdf.Rect(x + 4, y + 2, x + 4 + r.width * sc, y + 2 + r.height * sc)
    pg.show_pdf_page(tgt, cache[f], i, clip=r)
    pg.insert_text((x + 2, y + chh - 14), f"0x{c:02X}", fontsize=7)
    pg.insert_text((x + 2, y + chh - 6), f"{codes[c]:,}", fontsize=6, color=(.4, .4, .4))
pg.get_pixmap(dpi=170).save(os.path.join(ROOT, "reports", "figures", "glyph_atlas.png"))

# --- example overlay
ex = "hicaz--sarki--semai--degdi_saclarima--bekirof.pdf"
doc = pymupdf.open(os.path.join(PDF_DIR, ex)); page = doc[0]; sh = page.new_shape()
for s in page_symbols(doc, page):
    if s["bbox"]: sh.draw_rect(pymupdf.Rect(s["bbox"]))
sh.finish(color=(1, 0, 0), width=0.4); sh.commit()
page.get_pixmap(dpi=120, clip=pymupdf.Rect(0, 90, 595, 470)).save(os.path.join(ROOT, "reports", "figures", "example_boxes.png"))

with open(os.path.join(ROOT, "reports", "pdf_symbol_inventory.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["code_hex", "count", "n_files"])
    for c, v in codes.most_common(): w.writerow([f"0x{c:02X}", v, files_with[c]])
with open(os.path.join(ROOT, "data", "pdf_page_counts.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["file", "pages", "symbols", "unmapped", "embedding"]); w.writerows(rows)
summary = dict(files=len(files), pages=n_pages, symbols=n_sym, distinct_codes=len(codes),
               fully_extracted=dict(files=full_files, pages=full_pages, symbols=full_syms),
               embedding_files=dict(kinds), embedding_pages=dict(kind_pages),
               unmapped_chars=unmapped, drawings=dict(draw), font_sizes=dict(sizes.most_common(6)))
json.dump(summary, open(os.path.join(ROOT, "reports", "pdf_symbol_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
