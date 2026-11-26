"""Shared helpers for training / evaluation on CamVid.

Category ids follow categories.json order:
0 Road & paths, 1 Sky, 2 Trees & greenery, 3 People, 4 Buildings, 5 Foreign objects.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from semantics import CategoryMap  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Where CamVid and the cached DDRNet scores live (override with environment variables)
CAMVID = os.environ.get("CAMVID_DIR", os.path.join(ROOT, "data", "CamVid"))
CACHE = os.environ.get("CACHE_DIR", os.path.join(ROOT, "data", "cache"))
HEAD_PATH = os.path.join(ROOT, "models", "head.joblib")
IGNORE = 255
NCAT = 6
CAT_NAMES = ["Road & paths", "Sky", "Trees & greenery", "People", "Buildings", "Foreign objects"]

# CamVid (SegNet version): 0 Sky, 1 Building, 2 Pole, 3 Road, 4 Pavement, 5 Tree,
# 6 SignSymbol, 7 Fence, 8 Car, 9 Pedestrian, 10 Bicyclist, 11 void
CAMVID_TO_CAT = np.full(256, IGNORE, np.uint8)
for cv, cat in {0: 1, 1: 4, 2: 5, 3: 0, 4: 0, 5: 2, 6: 5, 7: 4, 8: 5, 9: 3, 10: 3}.items():
    CAMVID_TO_CAT[cv] = cat

_cm = CategoryMap()
CS_TO_CAT = _cm.lut  # Cityscapes train id -> category


def names(split):
    return sorted(os.listdir(os.path.join(CAMVID, split)))


def load(split, name):
    img = cv2.imread(os.path.join(CAMVID, split, name))
    gt = CAMVID_TO_CAT[cv2.imread(os.path.join(CAMVID, split + "annot", name), 0)]
    sc = np.load(os.path.join(CACHE, split, name.replace(".png", ".npy"))).astype(np.float32)
    return img, gt, sc


def ground_prior(sc, labels, gap=60.0):
    h = labels.shape[0]
    ground = np.array([0, 1, 9])
    gsc = sc[:, :, ground]
    lower = (np.arange(h) >= h // 2)[:, None]
    fix = (labels == 2) & lower & (sc[:, :, 2] - gsc.max(axis=2) < gap)
    out = labels.copy()
    out[fix] = ground[gsc.argmax(axis=2)][fix]
    return out


class Confusion:
    def __init__(self, n=NCAT):
        self.m = np.zeros((n, n), np.int64)
        self.n = n

    def add(self, gt, pred):
        v = gt != IGNORE
        self.m += np.bincount(gt[v].astype(np.int64) * self.n + pred[v], minlength=self.n * self.n).reshape(self.n, self.n)

    def iou(self):
        tp = np.diag(self.m)
        return tp / np.maximum(self.m.sum(0) + self.m.sum(1) - tp, 1)

    def report(self, title):
        iou = self.iou()
        acc = np.diag(self.m).sum() / self.m.sum()
        print(f"{title}: mIoU {iou.mean() * 100:.1f}  pixel acc {acc * 100:.1f}  | " +
              "  ".join(f"{n.split()[0]} {v * 100:.1f}" for n, v in zip(CAT_NAMES, iou)))
        return iou
