"""Register a photo or scan of a printed page to the rendered PDF page.

The mapping from page coordinates to image coordinates is a homography plus a
smooth correction (cubic polynomial), which absorbs paper curl and lens
distortion. It is used to carry the automatic symbol boxes over to real
captures, so that they can be scored without manual labelling."""
from __future__ import annotations
import cv2
import numpy as np

MATCH_WIDTH = 1600      # both images are resized to this width for matching


def _features(img, n=12000):
    s = MATCH_WIDTH / img.shape[1]
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    kp, des = cv2.SIFT_create(n).detectAndCompute(small, None)
    return np.float32([k.pt for k in kp]) / s, des


def _poly(xy, scale):
    x, y = xy[:, 0] / scale, xy[:, 1] / scale
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y,
                     x ** 3, x * x * y, x * y * y, y ** 3], 1)


class Mapping:
    """Page (reference image) pixel coordinates -> capture pixel coordinates."""

    def __init__(self, H, coef, scale):
        self.H, self.coef, self.scale = H, coef, scale

    def __call__(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        out = cv2.perspectiveTransform(xy[None], self.H)[0]
        if self.coef is not None:
            out = out + _poly(xy, self.scale) @ self.coef
        return out

    @property
    def zoom(self):
        """Mean magnification of the capture relative to the reference."""
        return float(np.sqrt(abs(np.linalg.det(self.H[:2, :2] / self.H[2, 2]))))


def register(ref, img, radius=14.0, holdout=0.2, seed=0):
    """Return (Mapping, info). `ref` and `img` are greyscale arrays.
    info['residual'] is measured on matches that were not used for the fit."""
    p1, d1 = _features(ref); p2, d2 = _features(img)
    knn = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in knn if a.distance < 0.75 * b.distance]
    a = np.float64([p1[g.queryIdx] for g in good]); b = np.float64([p2[g.trainIdx] for g in good])
    H, _ = cv2.findHomography(a, b, cv2.RANSAC, 6.0)
    if H is None:
        raise RuntimeError("registration failed")
    # guided matching: nearest descriptor among key points close to the predicted position
    pred = cv2.perspectiveTransform(p1[None].astype(np.float64), H)[0]
    order = np.argsort(p2[:, 0]); xs = p2[order, 0]
    A, B = [], []
    for i, q in enumerate(pred):
        lo, hi = np.searchsorted(xs, [q[0] - radius, q[0] + radius])
        cand = order[lo:hi]
        cand = cand[np.abs(p2[cand, 1] - q[1]) < radius]
        if len(cand) == 0:
            continue
        dist = np.linalg.norm(d2[cand] - d1[i], axis=1)
        j = np.argsort(dist)
        if dist[j[0]] < 220 and (len(j) == 1 or dist[j[0]] < 0.8 * dist[j[1]]):
            A.append(p1[i]); B.append(p2[cand[j[0]]])
    A, B = np.float64(A), np.float64(B)
    rng = np.random.default_rng(seed); test = rng.random(len(A)) < holdout
    scale = float(max(ref.shape))
    res = B - cv2.perspectiveTransform(A[None], H)[0]
    keep = ~test
    coef = None
    for _ in range(3):                      # robust least squares
        coef, *_ = np.linalg.lstsq(_poly(A[keep], scale), res[keep], rcond=None)
        err = np.linalg.norm(res - _poly(A, scale) @ coef, axis=1)
        keep = ~test & (err < max(3.0, 2.5 * np.median(err[~test])))
    m = Mapping(H, coef, scale)
    err = np.linalg.norm(m(A[test]) - B[test], axis=1)
    err0 = np.linalg.norm(res[test], axis=1)
    return m, dict(matches=int(len(A)), zoom=m.zoom,
                   residual=float(np.median(err)), residual_p90=float(np.percentile(err, 90)),
                   residual_homography=float(np.median(err0)))
