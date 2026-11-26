"""Trained refinement head (stage 3b).

DDRNet was trained on Cityscapes; on other domains (CamVid, our campus) it
confuses classes in systematic ways (grey sky -> building, dry soil ->
building). The head is a gradient-boosted classifier trained on labelled
CamVid pixels. Per pixel it sees:
  - DDRNet's 19 class scores               (what the network thinks)
  - colour: L, a, b, chroma                 (sky is bright and smooth, etc.)
  - vertical position y / H                 (ground is low, sky is high)
  - texture: local std of L (2 scales), edge strength
  - neighbourhood colour (15x15 mean L, a, b)
and outputs probabilities for the 6 categories directly.

Features are computed at a fixed working size (short side 360 px, CamVid's
size), so photos of any resolution are described the same way.
"""
import cv2
import numpy as np

FEAT_SHORT = 360
FEATURE_NAMES = ([f"score_{i}" for i in range(19)] +
                 ["L", "a", "b", "chroma", "y", "std7", "std21", "edge", "mL15", "ma15", "mb15"])


def _resize_short(x, short, interp):
    h, w = x.shape[:2]
    s = short / min(h, w)
    return cv2.resize(x, (round(w * s), round(h * s)), interpolation=interp)


def features(bgr, scores):
    """bgr: HxWx3 uint8, scores: HxWx19 (float or uint8), same size.
    Returns (F, (h, w)) with F of shape (h*w, 30) at the working size."""
    img = _resize_short(bgr, FEAT_SHORT, cv2.INTER_AREA)
    sc = _resize_short(scores.astype(np.float32), FEAT_SHORT, cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)
    y = np.repeat((np.arange(h, dtype=np.float32) / h)[:, None], w, axis=1)

    def lstd(k):
        m = cv2.blur(L, (k, k))
        return np.sqrt(np.maximum(cv2.blur(L * L, (k, k)) - m * m, 0))

    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)
    edge = cv2.blur(np.sqrt(gx * gx + gy * gy), (5, 5))
    m15 = cv2.blur(lab, (15, 15))
    F = np.dstack([sc, L, a, b, chroma, y, lstd(7), lstd(21), edge, m15]).reshape(-1, 30)
    return F.astype(np.float32), (h, w)


class Head:
    def __init__(self, path):
        import joblib
        self.model = joblib.load(path)

    def predict_proba(self, bgr, scores):
        """Category probabilities (H, W, 6) at the photo's own size."""
        F, (h, w) = features(bgr, scores)
        P = self.model.predict_proba(F).reshape(h, w, -1).astype(np.float32)
        H, W = bgr.shape[:2]
        return cv2.resize(P, (W, H), interpolation=cv2.INTER_LINEAR)

    def predict(self, bgr, scores):
        return self.predict_proba(bgr, scores).argmax(axis=2).astype(np.uint8)
