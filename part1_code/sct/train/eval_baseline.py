"""mIoU of the current (untrained) pipeline on CamVid test, with/without ground prior and cleanup."""
from common import *
from semantics import CategoryMap
cm = CategoryMap()
C = {k: Confusion() for k in ["raw", "prior", "prior+cleanup"]}
for n in names("test"):
    img, gt, sc = load("test", n)
    lab = sc.argmax(2).astype(np.uint8)
    C["raw"].add(gt, CS_TO_CAT[lab])
    p = CS_TO_CAT[ground_prior(sc, lab)]
    C["prior"].add(gt, p)
    C["prior+cleanup"].add(gt, cm.cleanup(p))
for k, c in C.items():
    c.report(f"{k:14s}")
