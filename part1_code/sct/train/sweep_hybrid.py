"""Tune the hybrid rule on CamVid VAL: use the head only where DDRNet's margin < tau."""
import joblib
from common import *
from head import features
m = joblib.load(HEAD_PATH)
data = []
for n in names("val"):
    img, gt, sc = load("val", n)
    F, (h, w) = features(img, sc)
    Fs = F.reshape(h, w, -1)[::2, ::2].reshape(-1, F.shape[1])           # stride 2 for speed
    P = m.predict_proba(Fs).reshape((h + 1) // 2, (w + 1) // 2, -1).astype(np.float32)
    P = cv2.resize(P, (gt.shape[1], gt.shape[0]))
    lab = sc.argmax(2).astype(np.uint8)
    base = CS_TO_CAT[ground_prior(sc, lab)]
    top2 = np.partition(sc, -2, axis=2)[:, :, -2:]
    margin = top2[:, :, 1] - top2[:, :, 0]
    data.append((gt, base, P.argmax(2).astype(np.uint8), margin))
for tau in [0, 10, 20, 30, 45, 60, 90, 1e9]:
    c = Confusion()
    for gt, base, head, margin in data:
        c.add(gt, np.where(margin < tau, head, base))
    c.report(f"tau={tau:>5g}")
