"""Held-out CamVid TEST: DDRNet / +prior / +head / hybrid(tau=30).
Chunked (each part is resumable): python eval_test.py 0 117 ; python eval_test.py 117 233 ;
then python eval_test.py report"""
import sys, joblib
from common import *
from head import features
m = joblib.load(HEAD_PATH)
if sys.argv[1] == "report":
    import glob
    C = sum(np.load(f) for f in glob.glob(os.path.join(CACHE, "conf_*_*.npy")))
    for k, t in enumerate(["DDRNet only", "DDRNet + ground prior", "Full trained head", "Hybrid (tau=30)"]):
        c = Confusion(); c.m = C[k]; c.report(f"{t:22s}")
    sys.exit(0)
a, b = int(sys.argv[1]), int(sys.argv[2])
C = np.zeros((4, NCAT, NCAT), np.int64)
for n in names("test")[a:b]:
    img, gt, sc = load("test", n)
    F, (h, w) = features(img, sc)
    Fs = F.reshape(h, w, -1)[::2, ::2].reshape(-1, F.shape[1])
    P = cv2.resize(m.predict_proba(Fs).reshape((h + 1) // 2, (w + 1) // 2, -1).astype(np.float32), (gt.shape[1], gt.shape[0]))
    lab = sc.argmax(2).astype(np.uint8)
    base = CS_TO_CAT[ground_prior(sc, lab)]
    top2 = np.partition(sc, -2, axis=2)[:, :, -2:]
    head = P.argmax(2).astype(np.uint8)
    for k, pred in enumerate([CS_TO_CAT[lab], base, head, np.where(top2[..., 1] - top2[..., 0] < 30, head, base)]):
        c = Confusion(); c.add(gt, pred); C[k] += c.m
os.makedirs(CACHE, exist_ok=True)
np.save(os.path.join(CACHE, f"conf_{a}_{b}.npy"), C)
print("saved", a, b, flush=True)
