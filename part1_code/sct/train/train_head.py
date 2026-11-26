"""Train the refinement head on CamVid train and choose its settings on val.
(Held-out test numbers come from eval_test.py.)

Usage: python train_head.py            (writes models/head.joblib)
"""
import time

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from common import *  # noqa: F403
from head import features

rng = np.random.default_rng(0)
PER_CLASS = 400            # pixels sampled per class per image (class-balanced sampling)


def sample(split, per_class=PER_CLASS):
    X, Y = [], []
    for n in names(split):
        img, gt, sc = load(split, n)
        F, (h, w) = features(img, sc)
        g = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST).ravel()
        for c in range(NCAT):
            idx = np.flatnonzero(g == c)
            if idx.size:
                pick = rng.choice(idx, min(per_class, idx.size), replace=False)
                X.append(F[pick])
                Y.append(np.full(pick.size, c, np.uint8))
    return np.concatenate(X), np.concatenate(Y)


def evaluate(model, split, title):
    c_base, c_prior, c_head = Confusion(), Confusion(), Confusion()
    for n in names(split):
        img, gt, sc = load(split, n)
        lab = sc.argmax(2).astype(np.uint8)
        c_base.add(gt, CS_TO_CAT[lab])
        c_prior.add(gt, CS_TO_CAT[ground_prior(sc, lab)])
        F, (h, w) = features(img, sc)
        P = model.predict_proba(F).reshape(h, w, -1).astype(np.float32)
        P = cv2.resize(P, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        c_head.add(gt, P.argmax(2).astype(np.uint8))
    print(f"--- {title} ---")
    return (c_base.report("DDRNet only         "), c_prior.report("DDRNet + prior      "),
            c_head.report("DDRNet + trained head"))


if __name__ == "__main__":
    t = time.time()
    Xtr, Ytr = sample("train")
    print(f"train pixels {len(Ytr):,}  ({time.time() - t:.0f}s)", flush=True)
    best, best_miou = None, -1
    for lr, leaves in [(0.1, 31), (0.1, 63)]:
        t = time.time()
        m = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, max_iter=300,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=15, random_state=0)
        m.fit(Xtr, Ytr)
        print(f"lr={lr} leaves={leaves}: {m.n_iter_} trees, {time.time() - t:.0f}s", flush=True)
        iou = evaluate(m, "val", f"VAL lr={lr} leaves={leaves}")[2]
        if iou.mean() > best_miou:
            best, best_miou = m, iou.mean()
    joblib.dump(best, HEAD_PATH, compress=3)
    print(f"chosen on val: lr={best.learning_rate} leaves={best.max_leaf_nodes} (val mIoU {best_miou * 100:.1f})")
    print("Next: held-out test evaluation with  python eval_test.py 0 117 ; python eval_test.py 117 233 ; python eval_test.py report")
