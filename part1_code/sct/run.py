"""Semantic Controlled Image Translation - command-line runner.

Usage
  python run.py --input test_images/            # a folder
  python run.py --input my_photo.jpg            # one photo
  python run.py --input photos/ --unknown-margin 8 --no-tta

For every photo it writes into --output:
  <name>_mask.png        flat colour semantic map (your 6 categories)
  <name>_overlay.jpg     mask blended on the photo, with region outlines
  <name>_translated.jpg  semantic-controlled translation (colour per class, texture kept)
  <name>_classes19.png   raw 19-class Cityscapes map (for debugging)
  <name>_panel.jpg       all of the above side by side + legend with % coverage
  summary.json           coverage % per category for every photo
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from segmenter import Segmenter
from semantics import CategoryMap, make_panel

CITYSCAPES_PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32)], np.uint8)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def load_photo(path, max_side):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)          # applies EXIF rotation
    if bgr is None:
        raise ValueError(f"cannot read {path}")
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def process(path, seg, cmap, args):
    t0 = time.time()
    bgr = load_photo(path, args.max_side)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    zooms = tuple(float(z) for z in args.zooms.split(","))
    labels19, margin = seg.segment(bgr, tta=not args.no_tta, zooms=zooms,
                                   detail=args.detail, ground_prior=args.ground_prior)                  # stages 1-3
    cat = cmap.group(labels19, margin, args.unknown_margin)                 # stage 4
    if not args.no_cleanup:
        cat = cmap.cleanup(cat)                                             # stage 5
    mask = cmap.mask_rgb(cat)                                               # stage 6
    over = cmap.overlay(rgb, cat)
    trans = cmap.translate(rgb, cat, keep_texture=args.texture)
    cov = cmap.coverage(cat)                                                # stage 7
    name = os.path.splitext(os.path.basename(path))[0]
    panel = make_panel(rgb, mask, over, trans, cmap, cov, title=f"Semantic Controlled Image Translation - {name}")

    o = os.path.join(args.output, name)
    cv2.imwrite(o + "_mask.png", mask[:, :, ::-1])
    cv2.imwrite(o + "_overlay.jpg", over[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(o + "_translated.jpg", trans[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(o + "_classes19.png", CITYSCAPES_PALETTE[labels19][:, :, ::-1])
    cv2.imwrite(o + "_panel.jpg", panel[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    dt = time.time() - t0
    print(f"[{name}] {bgr.shape[1]}x{bgr.shape[0]}  {dt:.1f}s  " +
          "  ".join(f"{k}={v:.1f}%" for k, v in cov.items()))
    return name, cov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--model", default=None, help="default: models/ddrnet23_slim.onnx")
    ap.add_argument("--config", default=None, help="default: categories.json")
    ap.add_argument("--max-side", type=int, default=2048, help="downscale photos larger than this")
    ap.add_argument("--no-tta", action="store_true", help="disable flip averaging (2x faster)")
    ap.add_argument("--zooms", default="1.0",
                    help="comma-separated zoom levels (<=1) averaged together, e.g. '1.0,0.6'")
    ap.add_argument("--detail", type=int, default=0,
                    help="also average a softened copy with this short side (e.g. 512); 0 = off")
    ap.add_argument("--no-cleanup", action="store_true", help="keep tiny speckle regions")
    ap.add_argument("--ground-prior", type=float, default=60.0,
                    help="relabel 'building' in the lower half as ground when a ground class "
                         "scores within this much (fixes sandy soil read as building); 0 = off")
    ap.add_argument("--unknown-margin", type=float, default=0.0,
                    help="pixels whose top-2 class scores differ by less than this become "
                         "'Foreign objects' (0 = off; try 5-15)")
    ap.add_argument("--texture", type=float, default=0.65, help="0 = flat colours, 1 = strong photo texture")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths = ([os.path.join(args.input, f) for f in sorted(os.listdir(args.input)) if f.lower().endswith(EXTS)]
             if os.path.isdir(args.input) else [args.input])
    seg, cmap = Segmenter(args.model), CategoryMap(args.config)
    summary = dict(process(p, seg, cmap, args) for p in paths)
    json.dump(summary, open(os.path.join(args.output, "summary.json"), "w"), indent=2)
    print(f"done: {len(paths)} image(s) -> {args.output}/")


if __name__ == "__main__":
    main()
