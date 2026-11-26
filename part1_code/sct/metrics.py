"""Evaluation metrics from the poster's plan that can be computed without
extra pretrained networks.

outside_mask_l1   mean |output - input| (0-255) on pixels that were NOT selected.
                  Reported two ways: beyond the soft-edge band (strict: should
                  be exactly 0) and on all unselected pixels (includes the band).
in_mask_change    mean |output - input| inside the selection: proves the edit
                  actually happened (a do-nothing method scores 0 outside AND inside).
layout_miou       re-segment the output and compare its category map with the
                  input's: does the scene layout survive the translation?
boundary_f        F-score of Canny edges of input vs output (2 px tolerance):
                  are object boundaries preserved?
FID and LPIPS need Inception / AlexNet weights, which are not available here.
"""
import cv2
import numpy as np


def outside_mask_l1(inp, out, hard, band_px):
    diff = np.abs(inp.astype(np.float32) - out.astype(np.float32)).mean(axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    far = cv2.dilate(hard.astype(np.uint8), k) == 0
    unsel = ~hard
    return (float(diff[far].mean()) if far.any() else 0.0,
            float(diff[unsel].mean()) if unsel.any() else 0.0)


def in_mask_change(inp, out, hard):
    diff = np.abs(inp.astype(np.float32) - out.astype(np.float32)).mean(axis=2)
    return float(diff[hard].mean()) if hard.any() else 0.0


def layout_miou(cat_a, cat_b, n=6):
    ious = []
    for c in range(n):
        a, b = cat_a == c, cat_b == c
        u = (a | b).sum()
        if u > 0.001 * cat_a.size:
            ious.append((a & b).sum() / u)
    return float(np.mean(ious))


def _edges(rgb):
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    v = np.median(g)
    return cv2.Canny(g, int(max(0, 0.66 * v)), int(min(255, 1.33 * v))) > 0


def boundary_f(inp, out, tol=2):
    ea, eb = _edges(inp), _edges(out)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    prec = (eb & da).sum() / max(eb.sum(), 1)
    rec = (ea & db).sum() / max(ea.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))
