"""Stages 4-7: group classes, clean up the map, and render the outputs.

Stage 4  group   : 19 Cityscapes labels -> your categories (categories.json)
                   (+ optional: low-confidence pixels -> "Foreign objects")
Stage 5  cleanup : tiny isolated regions are absorbed by their surroundings
Stage 6  render  : flat colour mask, overlay, and the *translated* image
                   (each region recoloured to its category colour while
                   keeping the photo's light, shadow and texture)
Stage 7  report  : side-by-side panel with legend + % coverage per category
"""
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from segmenter import CITYSCAPES_CLASSES


# ---------------- Stage 4: grouping ----------------
class CategoryMap:
    def __init__(self, path=None):
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "categories.json")
        cfg = json.load(open(path))
        self.cfg = cfg
        self.names = [c["name"] for c in cfg["categories"]]
        self.colors = np.array([c["color"] for c in cfg["categories"]], np.uint8)  # RGB
        idx = {n: i for i, n in enumerate(self.names)}
        fallback = idx[cfg["fallback"]]
        self.lut = np.full(256, fallback, np.uint8)
        for i, c in enumerate(cfg["categories"]):
            for cls in c["classes"]:
                self.lut[CITYSCAPES_CLASSES.index(cls)] = i
        self.uncertain_id = idx.get(cfg.get("uncertain_to", ""), fallback)
        self.min_frac = float(cfg.get("min_region_fraction", 0.0002))
        self.protected = {idx[n] for n in cfg.get("protect_from_cleanup", []) if n in idx}

    def group(self, labels19, margin=None, unknown_margin=0.0):
        cat = self.lut[labels19]
        if margin is not None and unknown_margin > 0:
            cat = cat.copy()
            cat[margin < unknown_margin] = self.uncertain_id
        return cat

    # ---------------- Stage 5: cleanup ----------------
    def cleanup(self, cat):
        """Absorb connected regions smaller than min_region_fraction of the
        image into the nearest surrounding category (protected categories,
        e.g. distant people, are kept even if small)."""
        h, w = cat.shape
        min_area = max(1, int(self.min_frac * h * w))
        hole = np.zeros((h, w), np.uint8)
        for k in range(len(self.names)):
            if k in self.protected:
                continue
            n, comp, stats, _ = cv2.connectedComponentsWithStats((cat == k).astype(np.uint8), connectivity=8)
            small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
            if small.size:
                hole[np.isin(comp, small)] = 1
        if not hole.any():
            return cat
        # nearest non-hole pixel for every hole pixel (zeros = kept pixels;
        # each kept pixel gets its own id, holes receive the id of the closest)
        _, nearest = cv2.distanceTransformWithLabels(
            hole, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        keep = hole == 0
        ys, xs = np.nonzero(keep)
        lut = np.zeros(nearest.max() + 1, np.uint8)
        lut[nearest[ys, xs]] = cat[ys, xs]
        out = cat.copy()
        out[hole == 1] = lut[nearest[hole == 1]]
        return out

    def sky_connectivity(self, cat, sky_id=1, top_frac=0.15):
        """Real sky is connected to the upper part of the frame. Sky regions that
        never reach the top `top_frac` of the image (e.g. a white cabinet or a
        pale wall read as sky) are given to the category that surrounds them."""
        h = cat.shape[0]
        n, comp = cv2.connectedComponents((cat == sky_id).astype(np.uint8), connectivity=8)
        if n <= 1:
            return cat
        top_ids = set(np.unique(comp[: max(1, int(top_frac * h))])) - {0}
        out = cat.copy()
        k = np.ones((9, 9), np.uint8)
        for i in range(1, n):
            if i in top_ids:
                continue
            blob = (comp == i).astype(np.uint8)
            ring = (cv2.dilate(blob, k) > 0) & (blob == 0)
            vals = cat[ring]
            vals = vals[vals != sky_id]
            out[blob > 0] = np.bincount(vals, minlength=len(self.names)).argmax() if vals.size else out[blob > 0]
        return out

    def building_support(self, cat, bld_id=4, sky_id=1, low_frac=0.4):
        """Buildings stand up against the sky; flat dry ground does not. A
        'building' blob that touches no sky and lies mainly in the lower part
        of the photo (centre below `low_frac` of the height) is given to the
        non-building category that surrounds it."""
        h = cat.shape[0]
        n, comp = cv2.connectedComponents((cat == bld_id).astype(np.uint8), connectivity=8)
        if n <= 1:
            return cat
        sky = cat == sky_id
        out = cat.copy()
        k = np.ones((7, 7), np.uint8)
        for i in range(1, n):
            blob = (comp == i).astype(np.uint8)
            ys = np.nonzero(blob)[0]
            ring = (cv2.dilate(blob, k) > 0) & (blob == 0)
            if sky[ring].any() or ys.mean() < low_frac * h:
                continue
            vals = cat[ring]
            vals = vals[vals != bld_id]
            if vals.size:
                out[blob > 0] = np.bincount(vals, minlength=len(self.names)).argmax()
        return out

    # ---------------- Stage 6: rendering ----------------
    def mask_rgb(self, cat):
        return self.colors[cat]

    def overlay(self, rgb, cat, alpha=0.55):
        m = self.mask_rgb(cat).astype(np.float32)
        out = rgb.astype(np.float32) * (1 - alpha) + m * alpha
        # thin dark outlines between categories make regions easier to read
        edges = cv2.morphologyEx(cat, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        out[edges] *= 0.35
        return out.clip(0, 255).astype(np.uint8)

    def translate(self, rgb, cat, keep_texture=0.65):
        """Semantic-controlled translation: every region takes its category's
        colour (chroma) while keeping the photo's lightness pattern, so shading
        and texture survive. keep_texture=0 -> flat mask, 1 -> photo lightness."""
        lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_col = cv2.cvtColor(self.mask_rgb(cat), cv2.COLOR_RGB2LAB).astype(np.float32)
        L_img, L_col = lab_img[..., 0], lab_col[..., 0]
        # normalise photo lightness per category so a dark road and a bright
        # sky both show texture around their class colour's own lightness
        L_out = np.empty_like(L_img)
        for k in np.unique(cat):
            r = cat == k
            v = L_img[r]
            detail = (v - v.mean()) / (v.std() + 1e-6) * 28.0
            L_out[r] = L_col[r] + keep_texture * detail
        out = np.dstack([L_out.clip(0, 255), lab_col[..., 1], lab_col[..., 2]]).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

    # ---------------- Stage 7: report ----------------
    def coverage(self, cat):
        n = cat.size
        return {name: round(float((cat == i).sum()) * 100.0 / n, 2) for i, name in enumerate(self.names)}


def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_panel(rgb, mask, overlay, translated, catmap, coverage, title=""):
    """2x2 grid (original / mask / overlay / translated) + legend with %."""
    h, w = rgb.shape[:2]
    tile_w = 640
    tile_h = int(round(h * tile_w / w))
    tiles = [cv2.resize(x, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
             for x in (rgb, mask, overlay, translated)]
    labels = ["Original", "Semantic mask", "Overlay", "Translated (texture kept)"]
    pad, head, leg_h = 12, 44, 46 + 34 * ((len(catmap.names) + 2) // 3)
    W = 2 * tile_w + 3 * pad
    H = head + 2 * (tile_h + 30) + 2 * pad + leg_h
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 10), title or "Semantic Controlled Image Translation", fill=(20, 20, 20), font=_font(22))
    f = _font(16)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x = pad + (i % 2) * (tile_w + pad)
        y = head + (i // 2) * (tile_h + 30 + pad)
        d.text((x, y), lab, fill=(40, 40, 40), font=f)
        canvas.paste(Image.fromarray(t), (x, y + 24))
    y0 = H - leg_h + 8
    d.text((pad, y0), "Legend  (% of image)", fill=(20, 20, 20), font=f)
    for i, name in enumerate(catmap.names):
        x = pad + (i % 3) * (W // 3)
        y = y0 + 30 + (i // 3) * 34
        d.rectangle([x, y, x + 26, y + 22], fill=tuple(int(c) for c in catmap.colors[i]), outline=(60, 60, 60))
        d.text((x + 36, y + 2), f"{name}: {coverage[name]:.1f}%", fill=(30, 30, 30), font=f)
    return np.array(canvas)
