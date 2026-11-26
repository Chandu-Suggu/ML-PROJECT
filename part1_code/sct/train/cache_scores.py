"""Run DDRNet once over a CamVid split and cache its 19 class scores per pixel.

Scores are stored as uint8 (the model's own output is uint8; flip-averaged
values are rounded), one .npy per image, shape (H, W, 19).
Usage: python cache_scores.py <camvid_dir> <split> <out_dir>
   e.g. python cache_scores.py ../data/CamVid train ../data/cache
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segmenter import Segmenter  # noqa: E402

camvid, split, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(out, split), exist_ok=True)
seg = Segmenter(os.path.join(os.path.dirname(__file__), "..", "models", "ddrnet23_slim.onnx"))
names = sorted(os.listdir(os.path.join(camvid, split)))
t0 = time.time()
for i, n in enumerate(names):
    dst = os.path.join(out, split, n.replace(".png", ".npy"))
    if os.path.exists(dst):
        continue
    img = cv2.imread(os.path.join(camvid, split, n))
    sc = seg.scores(img, tta=True)
    np.save(dst, np.clip(np.rint(sc), 0, 255).astype(np.uint8))
    if i % 25 == 0:
        print(f"{split} {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)
print(f"{split} done {len(names)} in {time.time() - t0:.0f}s", flush=True)
