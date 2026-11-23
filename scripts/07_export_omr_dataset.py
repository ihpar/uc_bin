"""Export a symbol-detection dataset (YOLO format) from the SymbTr PDFs.

    python scripts/07_export_omr_dataset.py            # -> omr_data/
    python scripts/07_export_omr_dataset.py --smoke    # a few pages, for testing

Labels are automatic: every music symbol is a character of the embedded Mus2
font, so its class (font code) and bounding box are read from the PDF. Only
files in which every symbol could be mapped are used.

Pages are rendered at 200 dpi and cut into overlapping 1024-pixel tiles, since
many symbols are only a few pixels wide on a full page. Sets written:

    train          clean renders
    train_deg      the same tiles with synthetic print/scan/photo degradation
    val, val_deg   for model selection
    test           clean renders of test-split pieces
    cff            clean renders in a second engraving (older Mus2 version), never
                   used for training: a test of generalisation across styles
    scanner, phone real captures of 15 printed test pages (data/scans), with the
                   labels carried over by registration (src/register.py)

Pieces are assigned by data/splits.csv, so no piece is in two sets.
"""
from __future__ import annotations
import argparse, collections, csv, json, os, random, shutil, sys

import cv2
import numpy as np
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pdf_symbols import page_symbols           # noqa: E402
from register import register                  # noqa: E402
from symbtr import PDF_DIR                     # noqa: E402

DPI = 200
ZOOM = DPI / 72
TILE = 1024
MIN_SIDE = 10         # pixels; smaller boxes (dots) are grown to this size, so that a
                      # registration error of 1-2 pixels on real captures does not void the match
MIN_VISIBLE = 0.5     # a box cut by the tile edge is kept if this share is inside


# ---------------------------------------------------------------- pages
def labelled_files(embedding="truetype"):
    """Files whose symbols were all mapped to glyph boxes. `embedding` selects the
    engraving: "truetype" = current Mus2 font, "cff" = files typeset with an older
    Mus2 version in 2007 (different glyph designs and layout)."""
    out = []
    with open(os.path.join(ROOT, "data", "pdf_page_counts.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["embedding"] == embedding and r["unmapped"] == "0" and int(r["symbols"]) > 0:
                out.append((r["file"], int(r["pages"])))
    return out


def piece_split():
    with open(os.path.join(ROOT, "data", "splits.csv"), encoding="utf-8") as fh:
        return {r["fname"][:-4]: r["split"] for r in csv.DictReader(fh)}


def render(fname, page):
    doc = pymupdf.open(os.path.join(PDF_DIR, fname)); p = doc[page]
    pm = p.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w).copy()
    boxes = [(s["code"], *[v * ZOOM for v in s["bbox"]]) for s in page_symbols(doc, p) if s["bbox"]]
    return img, boxes


def tile_origins(size):
    if size <= TILE:
        return [0]
    n = int(np.ceil((size - TILE) / (TILE * 0.75))) + 1
    return [int(round(i * (size - TILE) / (n - 1))) for i in range(n)]


def tiles(img, boxes):
    """Yield (tx, ty, tile image, [(code, x0, y0, x1, y1) in tile pixels])."""
    h, w = img.shape[:2]
    for ty in tile_origins(h):
        for tx in tile_origins(w):
            t = img[ty:ty + TILE, tx:tx + TILE]
            if t.shape != (TILE, TILE):
                t = cv2.copyMakeBorder(t, 0, TILE - t.shape[0], 0, TILE - t.shape[1],
                                       cv2.BORDER_CONSTANT, value=255)
            lab = []
            for c, x0, y0, x1, y1 in boxes:
                if x1 - x0 < MIN_SIDE: m = (x0 + x1) / 2; x0, x1 = m - MIN_SIDE / 2, m + MIN_SIDE / 2
                if y1 - y0 < MIN_SIDE: m = (y0 + y1) / 2; y0, y1 = m - MIN_SIDE / 2, m + MIN_SIDE / 2
                a0, b0 = max(x0 - tx, 0), max(y0 - ty, 0)
                a1, b1 = min(x1 - tx, TILE), min(y1 - ty, TILE)
                if a1 <= a0 or b1 <= b0:
                    continue
                if (a1 - a0) * (b1 - b0) < MIN_VISIBLE * (x1 - x0) * (y1 - y0):
                    continue
                lab.append((c, a0, b0, a1, b1))
            yield tx, ty, t, lab


# ---------------------------------------------------------------- degradation
def degrade(img, rng):
    """Synthetic print + capture: uneven light, low contrast, blur, noise, JPEG."""
    h, w = img.shape
    x = img.astype(np.float32) / 255.0
    if rng.random() < 0.4:                                  # ink spread
        x = cv2.erode(x, np.ones((2, 2), np.uint8))
    sigma = rng.choice([0, rng.uniform(0.4, 1.2), rng.uniform(1.0, 2.2)])
    if sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    if rng.random() < 0.25:                                 # motion blur
        n = rng.choice([3, 5, 7]); k = np.zeros((n, n), np.float32)
        if rng.random() < 0.5: k[n // 2, :] = 1.0 / n
        else: k[:, n // 2] = 1.0 / n
        x = cv2.filter2D(x, -1, k)
    paper, ink = rng.uniform(0.55, 0.97), rng.uniform(0.02, 0.30)
    x = ink + (paper - ink) * x
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)                         # light falling off to one side
    ramp = ((xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)) / max(h, w)
    x = x * (1 + rng.uniform(0, 0.35) * ramp)
    if rng.random() < 0.5:                                  # soft shadow
        cx, cy = rng.uniform(0, w), rng.uniform(0, h); r = rng.uniform(0.25, 0.7) * w
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        x = x * (1 - rng.uniform(0.1, 0.4) * np.clip(1 - d / r, 0, 1))
    x = x + rng.normal(0, rng.uniform(0.005, 0.04), x.shape).astype(np.float32)
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------- writing
class Writer:
    def __init__(self, out, classes):
        self.out, self.cid = out, {c: i for i, c in enumerate(classes or [])}
        self.count = collections.defaultdict(collections.Counter)
        self.n = collections.Counter()

    def add(self, split, name, img, lab, quality=95):
        d = os.path.join(self.out, split)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "labels"), exist_ok=True)
        cv2.imwrite(os.path.join(d, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        with open(os.path.join(d, "labels", name + ".txt"), "w") as fh:
            for c, x0, y0, x1, y1 in lab:
                fh.write(f"{self.cid[c]} {(x0 + x1) / 2 / TILE:.6f} {(y0 + y1) / 2 / TILE:.6f} "
                         f"{(x1 - x0) / TILE:.6f} {(y1 - y0) / TILE:.6f}\n")
                self.count[split][c] += 1
        self.n[split] += 1


def _page_job(job):
    out, cid, split, deg_split, f, p, seed = job
    wr = Writer(out, None); wr.cid = cid
    rng = np.random.default_rng(seed)
    img, boxes = render(f, p)
    stem = f"{os.path.splitext(f)[0]}__p{p + 1}"
    for tx, ty, t, lab in tiles(img, boxes):
        name = f"{stem}__{tx}_{ty}"
        wr.add(split, name, t, lab)
        if deg_split:
            wr.add(deg_split, name, degrade(t, rng), lab, quality=92)
    return {k: dict(v) for k, v in wr.count.items()}, dict(wr.n)


def export_pages(wr, split, pages, deg_split=None, seed=0, workers=1):
    jobs = [(wr.out, wr.cid, split, deg_split, f, p, seed * 100003 + k) for k, (f, p) in enumerate(pages)]
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        for k, (count, n) in enumerate(pool.imap_unordered(_page_job, jobs, chunksize=4)):
            for s_, c in count.items(): wr.count[s_].update(c)
            wr.n.update(n)
            if (k + 1) % 100 == 0:
                print(f"  {split}: {k + 1}/{len(pages)} pages", flush=True)


def export_captures(wr, kind, report):
    """Real scans / photos of the printed test pages, labels carried over by registration."""
    d = os.path.join(ROOT, "data", "scans")
    with open(os.path.join(d, "print_list.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        n = int(r["image"])
        path = next((os.path.join(d, kind, f"{s}{e}") for s in (f"{n:02d}", str(n))
                     for e in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
                     if os.path.exists(os.path.join(d, kind, f"{s}{e}"))), None)
        if path is None:
            continue
        cap = cv2.imread(path, cv2.IMREAD_GRAYSCALE)        # EXIF orientation is applied
        ref, boxes = render(r["pdf"], int(r["page"]) - 1)
        m, info = register(ref, cap)
        s = 1.0 / m.zoom                                    # bring the capture to the 200 dpi scale
        cap = cv2.resize(cap, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = cap.shape
        moved = []
        for c, x0, y0, x1, y1 in boxes:
            q = m([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * s
            a, b = q.min(0), q.max(0)
            if a[0] < 0 or a[1] < 0 or b[0] > w or b[1] > h:
                continue                                    # symbol outside the photo
            moved.append((c, a[0], a[1], b[0], b[1]))
        for tx, ty, t, lab in tiles(cap, moved):
            wr.add(kind, f"{kind}{n:02d}__{tx}_{ty}", t, lab)
        info.update(image=os.path.basename(path), symbols=len(boxes), kept=len(moved))
        report.append(dict(kind=kind, **info))
        print(f"  {kind} {n:02d}: {len(moved)}/{len(boxes)} symbols, "
              f"registration residual {info['residual']:.1f} px", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "omr_data"))
    ap.add_argument("--train_pages", type=int, default=700)
    ap.add_argument("--val_pages", type=int, default=60)
    ap.add_argument("--cff_pages", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.train_pages, a.val_pages, a.cff_pages = 6, 2, 3
    random.seed(a.seed)

    split = piece_split()
    pages = collections.defaultdict(list)
    for f, n in labelled_files():
        s = split.get(os.path.splitext(f)[0], "train")
        s = "train" if s == "few_shot" else s               # rare makams: fine for symbol detection
        pages[s] += [(f, p) for p in range(n)]
    for s in pages:
        pages[s].sort(); random.shuffle(pages[s])
    total = {s: len(v) for s, v in pages.items()}
    if a.smoke:
        pages["test"] = pages["test"][:3]

    # classes: every font code in the symbol inventory, most frequent first
    with open(os.path.join(ROOT, "reports", "pdf_symbol_inventory.csv"), encoding="utf-8") as fh:
        classes = [int(r["code_hex"], 16) for r in csv.DictReader(fh)]

    # training pages: first the pages that contain rare classes, then random ones
    train = pages["train"]
    if len(train) > a.train_pages and not a.smoke:
        print("scanning training pages for rare symbols ...", flush=True)
        codes = {}
        for f, n in {(f, None) for f, _ in train}:
            doc = pymupdf.open(os.path.join(PDF_DIR, f))
            for p in range(len(doc)):
                codes[(f, p)] = {s["code"] for s in page_symbols(doc, doc[p]) if s["bbox"]}
        have, chosen = collections.Counter(), []
        rare_first = sorted(classes, key=lambda c: sum(c in v for v in codes.values()))
        for c in rare_first:
            for pg in train:
                if have[c] >= 40:
                    break
                if pg not in chosen and c in codes.get(pg, ()):
                    chosen.append(pg)
                    for k in codes[pg]: have[k] += 1
            if len(chosen) >= a.train_pages // 2:
                break
        rest = [pg for pg in train if pg not in set(chosen)]
        train = chosen + rest[:a.train_pages - len(chosen)]
    else:
        train = train[:a.train_pages]

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    wr = Writer(a.out, classes)
    export_pages(wr, "train", train, "train_deg", seed=a.seed, workers=a.workers)
    export_pages(wr, "val", pages["val"][:a.val_pages], "val_deg", seed=a.seed + 1, workers=a.workers)
    export_pages(wr, "test", pages["test"], workers=a.workers)
    # second engraving: pieces of the test split first, then others, up to --cff_pages
    cff = [(f, p) for f, n in labelled_files("cff") for p in range(n)]
    cff.sort(key=lambda fp: (split.get(os.path.splitext(fp[0])[0]) != "test", fp))
    export_pages(wr, "cff", cff[:a.cff_pages], workers=a.workers)
    reg = []
    for kind in ("scanner", "phone"):
        export_captures(wr, kind, reg)

    names = {i: f"g{c:02X}" for i, c in enumerate(classes)}

    def yaml(name, train_dirs, val_dirs):
        with open(os.path.join(a.out, name + ".yaml"), "w") as fh:
            fh.write(f"path: {a.out}\ntrain:\n" + "".join(f"  - {d}/images\n" for d in train_dirs))
            fh.write("val:\n" + "".join(f"  - {d}/images\n" for d in val_dirs))
            fh.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items()))

    yaml("clean", ["train"], ["val"])
    yaml("mixed", ["train", "train_deg"], ["val", "val_deg"])
    for s in ("test", "cff", "scanner", "phone", "val_deg"):
        if wr.n[s]:
            yaml("eval_" + s, ["train"], [s])
    meta = dict(dpi=DPI, tile=TILE, min_side=MIN_SIDE, min_visible=MIN_VISIBLE, seed=a.seed,
                labelled_pages_available=total,
                pages=dict(train=len(train), val=min(a.val_pages, len(pages["val"])), test=len(pages["test"])),
                tiles=dict(wr.n), classes=[f"g{c:02X}" for c in classes],
                boxes={s: {f"g{c:02X}": n for c, n in v.items()} for s, v in wr.count.items()},
                registration=reg)
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("tiles:", dict(wr.n))
    print("boxes:", {s: sum(v.values()) for s, v in wr.count.items()})


if __name__ == "__main__":
    main()
