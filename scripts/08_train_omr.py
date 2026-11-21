"""Train and evaluate the symbol detector of the OMR preliminary study.

    python scripts/08_train_omr.py --data mixed --name mixed
    python scripts/08_train_omr.py --data clean --name clean

Needs the dataset from scripts/07_export_omr_dataset.py. The detector is an
off-the-shelf YOLO model (Ultralytics, AGPL-3.0) used as a baseline; nothing in
it is specific to makam notation. Results are written to runs/omr/<name>/:

    results.json   per test set (clean renders, scanner, phone): precision, recall,
                   mAP for all classes and per class, and the same for the groups
                   of classes that matter for makam music
    figs/          example predictions, including pages without automatic labels

Weights and Ultralytics' own output go to omr_work/ (not committed).
"""
from __future__ import annotations
import argparse, csv, glob, json, os, random, sys, time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Provisional grouping by font code (to be confirmed with the music advisor).
GROUPS = {
    "accidentals": "58 50 6E 51 57 59 55 52 56 04 03 C7 5C D8",
    "noteheads_and_notes": "78 2A 2B 6F 54 29 25 27 2C 24 06 28 08 07",
    "rests": "47 46 48 45 44 49 0E",
    "augmentation_dot": "AB E2 E1 DF",
}
GROUPS = {k: {"g" + c for c in v.split()} for k, v in GROUPS.items()}


def evaluate(model, yaml, a, tag):
    m = model.val(data=yaml, imgsz=a.imgsz, batch=a.batch, conf=0.001, iou=0.6, plots=False,
                  project=os.path.join(ROOT, "omr_work"), name=f"{a.name}_val_{tag}", exist_ok=True,
                  verbose=False, workers=a.workers)
    b = m.box
    per = {}
    for k, ci in enumerate(b.ap_class_index):
        per[model.names[int(ci)]] = dict(precision=float(b.p[k]), recall=float(b.r[k]),
                                         ap50=float(b.ap50[k]), ap=float(b.ap[k]))
    return dict(precision=float(b.mp), recall=float(b.mr), map50=float(b.map50), map=float(b.map),
                classes_present=len(per), per_class=per)


def add_counts(res, meta, split):
    n = meta["boxes"].get(split, {})
    for c, d in res["per_class"].items():
        d["boxes"] = n.get(c, 0)
    tot = sum(d["boxes"] for d in res["per_class"].values()) or 1
    # micro view: classes weighted by how often they occur
    res["weighted_recall"] = sum(d["recall"] * d["boxes"] for d in res["per_class"].values()) / tot
    res["weighted_precision"] = sum(d["precision"] * d["boxes"] for d in res["per_class"].values()) / tot
    res["boxes"] = tot
    res["groups"] = {}
    for g, codes in GROUPS.items():
        sel = [d for c, d in res["per_class"].items() if c in codes]
        k = sum(d["boxes"] for d in sel)
        if k:
            res["groups"][g] = dict(classes=len(sel), boxes=k,
                                    recall=sum(d["recall"] * d["boxes"] for d in sel) / k,
                                    precision=sum(d["precision"] * d["boxes"] for d in sel) / k,
                                    ap50_mean=float(np.mean([d["ap50"] for d in sel])))
    return res


def draw(model, img, a, path):
    r = model.predict(img, imgsz=a.imgsz, conf=0.25, verbose=False)[0]
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 220), 1)
        cv2.putText(vis, model.names[int(c)][1:], (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 0, 0), 1)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(r.boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mixed", choices=["clean", "mixed"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "omr_data"))
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()
    a.name = a.name or a.data
    from ultralytics import YOLO

    work = os.path.join(ROOT, "omr_work")
    out = os.path.join(ROOT, "runs", "omr", a.name)
    os.makedirs(os.path.join(out, "figs"), exist_ok=True)
    best = os.path.join(work, a.name, "weights", "best.pt")
    with open(os.path.join(a.dataset, "meta.json")) as fh:
        meta = json.load(fh)

    t0 = time.time()
    if not (a.eval_only or os.path.exists(os.path.join(work, a.name, "done"))):
        last = os.path.join(work, a.name, "weights", "last.pt")
        if os.path.exists(last):                             # interrupted run: continue
            YOLO(last).train(resume=True)
        else:
            YOLO(a.weights).train(
                data=os.path.join(a.dataset, a.data + ".yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                workers=a.workers, patience=a.patience, seed=a.seed, fraction=a.fraction,
                project=work, name=a.name, exist_ok=True, plots=False,
                # notation is not mirror-symmetric and has a fixed orientation and scale
                fliplr=0.0, flipud=0.0, degrees=3.0, perspective=0.0002, scale=0.25, translate=0.1,
                mosaic=0.5, close_mosaic=5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, erasing=0.0)
        open(os.path.join(work, a.name, "done"), "w").close()
    train_minutes = (time.time() - t0) / 60

    model = YOLO(best)
    res = dict(name=a.name, data=a.data, weights=a.weights, imgsz=a.imgsz, epochs=a.epochs, seed=a.seed,
               train_minutes=round(train_minutes, 1), dataset={k: meta[k] for k in ("dpi", "tile", "pages", "tiles")},
               train_boxes=meta["boxes"].get("train", {}), sets={})
    csvp = os.path.join(work, a.name, "results.csv")
    if os.path.exists(csvp):
        with open(csvp) as fh:
            res["epochs_run"] = sum(1 for _ in csv.DictReader(fh))
    for tag in ("test", "val_deg", "scanner", "phone"):
        y = os.path.join(a.dataset, f"eval_{tag}.yaml")
        if os.path.exists(y):
            res["sets"][tag] = add_counts(evaluate(model, y, a, tag), meta, tag)
            s = res["sets"][tag]
            print(f"{tag:8s} mAP50 {s['map50']:.3f}  mAP {s['map']:.3f}  weighted recall {s['weighted_recall']:.3f} "
                  f"precision {s['weighted_precision']:.3f}  ({s['boxes']} boxes, {s['classes_present']} classes)", flush=True)

    # figures: one tile of each real capture type, and pages that have no automatic labels
    rnd = random.Random(0)
    for tag in ("scanner", "phone"):
        fs = sorted(glob.glob(os.path.join(a.dataset, tag, "images", "*.jpg")))
        for f in rnd.sample(fs, min(3, len(fs))):
            draw(model, cv2.imread(f), a, os.path.join(out, "figs", os.path.basename(f)))
    try:
        import pymupdf
        from symbtr import PDF_DIR
        with open(os.path.join(ROOT, "data", "pdf_page_counts.csv"), encoding="utf-8") as fh:
            nofont = [r["file"] for r in csv.DictReader(fh) if r["embedding"] == "no_font"]
        res["unlabelled_examples"] = []
        for f in rnd.sample(nofont, min(6, len(nofont))):
            pm = pymupdf.open(os.path.join(PDF_DIR, f))[0].get_pixmap(dpi=meta["dpi"], colorspace=pymupdf.csGRAY)
            img = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
            tile = np.ascontiguousarray(img[300:300 + meta["tile"], 100:100 + meta["tile"]])
            n = draw(model, tile, a, os.path.join(out, "figs", "nofont__" + os.path.splitext(f)[0][:60] + ".jpg"))
            res["unlabelled_examples"].append(dict(file=f, detections=n))
    except Exception as e:                                   # figures are optional
        print("skipped unlabelled examples:", e)

    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.join(out, "results.json"))


if __name__ == "__main__":
    main()
