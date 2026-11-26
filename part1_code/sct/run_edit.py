"""Region-selective semantic translation - command-line tool (poster version).

Examples
  python run_edit.py --input campus/test_7.jpg --edit sky=sunset
  python run_edit.py --input campus/ --edit trees=autumn --edit road=wet
  python run_edit.py --input photo.jpg --edit buildings=#3A6EA5 --mode ddrnet
  python run_edit.py --input photo.jpg --edit sky=reference --ref sunset.jpg

Regions: road, sky, trees, people, buildings, foreign
Styles:  sky = sunset | night | overcast | clear
         trees = autumn | lush | winter
         road = wet | snow
         any region = #RRGGBB (recolour, texture kept) | reference (needs --ref)

Writes, per photo, into --output:
  <name>_edited.jpg     the translated photo (only selected regions change)
  <name>_global.jpg     baseline: the same style applied to the whole image
  <name>_alpha.png      the soft edit mask
  <name>_compare.jpg    input | mask | ours | global baseline, with metrics
  metrics.json          poster metrics for ours vs the baseline
"""
import argparse
import json
import os

import cv2
import numpy as np

from edit import parse_edits, translate_regions
from metrics import boundary_f, in_mask_change, layout_miou, outside_mask_l1
from segmenter import Segmenter
from semantics import CategoryMap

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Labeller:
    """Photo -> 6-category map.

    mode "ddrnet": DDRNet + ground prior (no training)
    mode "head":   trained refinement head everywhere
    mode "hybrid": trained head only where DDRNet is unsure (score margin < tau)
    All modes then apply cleanup and the sky connectivity rule."""

    def __init__(self, mode="hybrid", tau=30.0):
        self.seg, self.cm, self.mode, self.tau = Segmenter(), CategoryMap(), mode, tau
        self.head = None
        if mode != "ddrnet":
            from head import Head
            self.head = Head(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "head.joblib"))

    def __call__(self, bgr):
        sc = self.seg.scores(bgr, tta=True)
        top2 = np.partition(sc, -2, axis=2)[:, :, -2:]
        margin = top2[..., 1] - top2[..., 0]
        lab = self.seg_labels(sc)
        cat = self.cm.lut[lab]
        if self.mode == "head":
            cat = self.head.predict(bgr, sc)
        elif self.mode == "hybrid":
            cat = np.where(margin < self.tau, self.head.predict(bgr, sc), cat)
        return self.cm.building_support(self.cm.sky_connectivity(self.cm.cleanup(cat)))

    @staticmethod
    def seg_labels(sc, gap=60.0):
        labels = sc.argmax(axis=2).astype(np.uint8)
        h = labels.shape[0]
        ground = np.array([0, 1, 9])
        gsc = sc[:, :, ground]
        lower = (np.arange(h) >= h // 2)[:, None]
        fix = (labels == 2) & lower & (sc[:, :, 2] - gsc.max(axis=2) < gap)
        labels[fix] = ground[gsc.argmax(axis=2)][fix]
        return labels


def label_text(img, text):
    img = img.copy()
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def process(path, labeller, edits, args, ref):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    s = min(1.0, args.max_side / max(h, w))
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cat = labeller(bgr)
    ours, alpha, hard = translate_regions(rgb, cat, edits, ref=ref)
    glob, _, _ = translate_regions(rgb, cat, edits, ref=ref, global_baseline=True)
    band = max(3, int(0.02 * min(rgb.shape[:2])))

    def score(out):
        far, allun = outside_mask_l1(rgb, out, hard, band)
        cat_out = labeller(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        unsel = ~hard
        return {"outside_L1_beyond_band": round(far, 3), "outside_L1_all_unselected": round(allun, 3),
                "in_region_change": round(in_mask_change(rgb, out, hard), 2),
                "layout_mIoU": round(layout_miou(cat, cat_out), 4),
                "layout_mIoU_unselected": round(layout_miou(np.where(unsel, cat, 255), np.where(unsel, cat_out, 255)), 4),
                "boundary_F": round(boundary_f(rgb, out), 4)}

    m = {"selected_fraction": round(float(hard.mean()), 4), "ours": score(ours), "global_baseline": score(glob)}
    name = os.path.splitext(os.path.basename(path))[0]
    o = os.path.join(args.output, name)
    cv2.imwrite(o + "_edited.jpg", ours[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
    cv2.imwrite(o + "_global.jpg", glob[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(o + "_alpha.png", (alpha * 255).astype(np.uint8))
    a3 = np.repeat((alpha * 255).astype(np.uint8)[..., None], 3, 2)
    tiles = [label_text(bgr, "Input"), label_text(a3, "Edit mask"),
             label_text(ours[:, :, ::-1], f"Ours  outside-L1 {m['ours']['outside_L1_all_unselected']:.2f}"),
             label_text(glob[:, :, ::-1], f"Global  outside-L1 {m['global_baseline']['outside_L1_all_unselected']:.1f}")]
    th = 520
    tiles = [cv2.resize(t, (round(t.shape[1] * th / t.shape[0]), th), interpolation=cv2.INTER_AREA) for t in tiles]
    cv2.imwrite(o + "_compare.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[{name}] selected {m['selected_fraction'] * 100:.1f}%  ours: outside-L1 {m['ours']['outside_L1_all_unselected']:.2f}, "
          f"in-region {m['ours']['in_region_change']:.1f}, boundary-F {m['ours']['boundary_F']:.3f}  | global: outside-L1 "
          f"{m['global_baseline']['outside_L1_all_unselected']:.1f}, boundary-F {m['global_baseline']['boundary_F']:.3f}")
    return name, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--edit", action="append", required=True, help="region=style, repeatable")
    ap.add_argument("--output", default="out_edit")
    ap.add_argument("--ref", help="reference photo for style 'reference'")
    ap.add_argument("--mode", default="hybrid", choices=["ddrnet", "head", "hybrid"],
                    help="labelling: DDRNet+prior, trained head, or hybrid (default)")
    ap.add_argument("--max-side", type=int, default=1600)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    edits = parse_edits(args.edit)
    ref = cv2.cvtColor(cv2.imread(args.ref), cv2.COLOR_BGR2RGB) if args.ref else None
    paths = ([os.path.join(args.input, f) for f in sorted(os.listdir(args.input)) if f.lower().endswith(EXTS)]
             if os.path.isdir(args.input) else [args.input])
    labeller = Labeller(mode=args.mode)
    res = dict(process(p, labeller, edits, args, ref) for p in paths)
    json.dump({"edits": args.edit, "mode": args.mode, "results": res},
              open(os.path.join(args.output, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
