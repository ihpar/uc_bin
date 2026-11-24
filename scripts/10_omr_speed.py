"""How long does recognising one page take?

    python scripts/10_omr_speed.py [--weights path/to/best.pt]   # -> reports/results/omr_speed_<cpu|cuda>.json

Pages are rendered at 200 dpi and cut into overlapping 1024-pixel tiles, as in
training (an A4 page gives 6 tiles). The script times symbol detection for a
whole page: all tiles in one batch, including non-maximum suppression. It does
not time the later steps (merging tiles, building notes), which run on the CPU
and are not written yet.

On a GPU machine it measures the GPU first and then the CPU with a limited
number of threads (--cpu_threads), as a stand-in for a modest server.

Weights: the detector trained by scripts/run_omr.sh (omr_work/<run>/weights/best.pt).
Speed depends on the number of detections, so use trained weights; with the
untrained yolo11s.pt the numbers are only a lower bound.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, platform, statistics, sys, time
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("export", os.path.join(ROOT, "scripts", "07_export_omr_dataset.py"))
E7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E7)


def page_tiles(img):
    T = E7.TILE
    out = []
    for ty in E7.tile_origins(img.shape[0]):
        for tx in E7.tile_origins(img.shape[1]):
            t = img[ty:ty + T, tx:tx + T]
            if t.shape != (T, T):
                t = np.pad(t, ((0, T - t.shape[0]), (0, T - t.shape[1])), constant_values=255)
            out.append(np.stack([t] * 3, -1))
    return out


def timed(model, pages, device):
    model.predict(pages[0], imgsz=E7.TILE, conf=0.25, device=device, verbose=False)   # warm-up
    ts, dets = [], []
    for tiles in pages:
        t0 = time.perf_counter()
        r = model.predict(tiles, imgsz=E7.TILE, conf=0.25, device=device, verbose=False)
        if device != "cpu": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0); dets.append(sum(len(x.boxes) for x in r))
    return dict(pages=len(pages), tiles_per_page=statistics.median(len(p) for p in pages),
                detections_per_page=statistics.median(dets),
                seconds_per_page=dict(median=statistics.median(ts), max=max(ts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--cpu_pages", type=int, default=5)
    ap.add_argument("--cpu_threads", type=int, default=4)
    a = ap.parse_args()
    from ultralytics import YOLO
    if a.weights is None:
        found = sorted(glob.glob(os.path.join(ROOT, "omr_work", "*", "weights", "best.pt")), key=os.path.getmtime)
        a.weights = found[-1] if found else "yolo11s.pt"
    trained = os.path.basename(a.weights) != "yolo11s.pt"
    if not trained:
        print("WARNING: no trained detector found; timing the untrained yolo11s.pt (lower bound only)")
    model = YOLO(a.weights)

    files = sorted(os.listdir(E7.PDF_DIR))
    step = max(1, len(files) // a.pages)
    pages = [page_tiles(E7.render(f, 0)[0]) for f in files[::step][:a.pages]]

    gpu = torch.cuda.is_available()
    res = dict(weights=os.path.relpath(a.weights, ROOT) if trained else a.weights, trained_weights=trained,
               classes=len(model.names), torch=torch.__version__, dpi=E7.DPI, tile=E7.TILE,
               gpu=torch.cuda.get_device_name(0) if gpu else None, cpu=platform.processor() or platform.machine())
    if gpu:
        res["cuda"] = timed(model, pages, 0)
    torch.set_num_threads(a.cpu_threads)
    res["cpu"] = dict(threads=a.cpu_threads, **timed(model, pages[:a.cpu_pages], "cpu"))
    out = os.path.join(ROOT, "reports", "results", f"omr_speed_{'cuda' if gpu else 'cpu'}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for k in ("cuda", "cpu"):
        if k in res:
            print(f"{k:20s} {res[k]['seconds_per_page']['median']:.3f} s per page "
                  f"({res[k]['tiles_per_page']} tiles, {res[k]['detections_per_page']} detections)")
    print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
