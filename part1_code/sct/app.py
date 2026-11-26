"""Semantic-Controlled Image Translation - interactive app.

Run:  python app.py
It prints a guide, asks for a photo, shows which regions it found, then asks
which region(s) to change and how. Only the chosen regions are translated;
every other pixel stays exactly as in the photo.
"""
import os
import sys

import cv2
import numpy as np

from edit import CATS, STYLES, translate_regions
from metrics import boundary_f, in_mask_change, outside_mask_l1

REGION_INFO = {
    "sky": "Sky",
    "trees": "Trees & greenery (trees, bushes, grass)",
    "road": "Road & paths (roads, footpaths, pavements)",
    "buildings": "Buildings (buildings, walls, fences)",
    "people": "People (pedestrians, riders)",
    "foreign": "Foreign objects (vehicles, bikes, poles, signs)",
}
STYLE_INFO = {
    "sunset": "orange-purple evening sky, clouds kept",
    "night": "dark night sky with faint stars",
    "overcast": "grey cloudy sky",
    "clear": "bright clear blue sky",
    "autumn": "orange-red autumn leaves",
    "lush": "deep, saturated green",
    "winter": "pale, frosted look",
    "wet": "darker, glossy rain-wet surface",
    "snow": "snow-covered surface",
}
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GUIDE = r"""
======================================================================
        SEMANTIC-CONTROLLED IMAGE TRANSLATION  -  USER GUIDE
======================================================================
What it does
  You give a photo, choose which PART of the scene to change, and pick
  a new look for it. Only that part changes; the rest of the photo is
  kept exactly as it was.

Parts (regions) you can change
  1. Sky ............... sunset | night | overcast | clear
  2. Trees & greenery .. autumn | lush | winter
  3. Road & paths ...... wet | snow
  4. Buildings ......... recolour only
  5. People ............ recolour only
  6. Foreign objects ... recolour only (vehicles, bikes, poles, signs)
  Only regions actually found in your photo are offered.
  Every region can also be:
     - recoloured to any colour (you type a hex code, e.g. #3A6EA5),
       the texture and shading of the photo are kept
     - matched to a reference photo (its colours are copied)

Limitations (please read)
  - The changes are COLOUR and TONE changes. Shapes are never changed:
    it cannot add or remove objects, or turn a road into grass.
  - Works best on outdoor, eye-level street or campus photos taken in
    daylight. Indoor, night, aerial or very close-up photos may be
    segmented wrongly.
  - Objects the model does not know (benches, mirrors, animals) may be
    counted as a nearby region.
  - Very thin details (single leaves, wires) may keep a faint trace of
    their old colour at the edges.
  - Dry, sandy fields on the far horizon can be counted as Buildings.

Tips for good results
  - Use a clear, well-lit photo where the region you want is large.
  - JPG or PNG; large photos are reduced to 1600 px automatically.
  - After the photo is analysed, open the "_regions.jpg" preview to see
    what was detected before you choose.

Outputs (saved in the output folder)
  <name>_edited.jpg     THE RESULT: only your chosen regions changed
  <name>_regions.jpg    what was detected in the photo
  <name>_mask.png       where the change was applied (white = changed)
  <name>_compare.jpg    input | mask | result, side by side
======================================================================
"""


def ask(prompt, valid=None, allow_empty=False):
    """Read a line; re-ask until it is valid. 'q' quits at any prompt."""
    while True:
        try:
            ans = input(prompt).strip()
        except EOFError:
            print("\nNo more input. Goodbye.")
            sys.exit(0)
        if ans.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            sys.exit(0)
        if not ans and allow_empty:
            return ""
        if valid is None or ans.lower() in valid:
            return ans
        print(f"  Please enter one of: {', '.join(sorted(valid))}  (or q to quit)")


def ask_photo():
    while True:
        p = ask("\nPath to your photo (or q to quit): ").strip('"').strip("'")
        if os.path.isfile(p) and p.lower().endswith(EXTS):
            img = cv2.imread(p)
            if img is not None:
                return p, img
        print("  Could not open that file. Give the path to a .jpg/.png photo.")


def ask_hex():
    while True:
        h = ask("  Colour as hex code, e.g. #3A6EA5: ")
        h = h if h.startswith("#") else "#" + h
        if len(h) == 7 and all(c in "0123456789abcdefABCDEF" for c in h[1:]):
            return h.upper()
        print("  That is not a hex colour. Use 6 hex digits like #FF8800.")


def ask_reference():
    while True:
        p = ask("  Path to the reference photo: ").strip('"').strip("'")
        img = cv2.imread(p) if os.path.isfile(p) else None
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("  Could not open that file.")


def choose_edits(present):
    """Ask which regions to change and the style for each. Returns (edits, ref)."""
    names = [r for r in CATS if r in present]
    print("\nWhich part do you want to change?")
    for i, r in enumerate(names, 1):
        print(f"  {i}. {REGION_INFO[r]:58s} {present[r]:5.1f}% of photo")
    print("  (choose one number, or several separated by commas, e.g. 1,3)")
    valid = {str(i) for i in range(1, len(names) + 1)}
    while True:
        picks = [x.strip() for x in ask("Your choice: ").split(",") if x.strip()]
        if picks and all(x in valid for x in picks):
            break
        print(f"  Enter numbers between 1 and {len(names)}.")
    edits, ref = [], None
    for x in dict.fromkeys(picks):
        region = names[int(x) - 1]
        opts = sorted(STYLES.get(region, [])) + ["colour", "reference"]
        print(f"\nNew look for {REGION_INFO[region].split(' (')[0]}:")
        for i, o in enumerate(opts, 1):
            desc = STYLE_INFO.get(o, "any colour you choose, texture kept" if o == "colour"
                                  else "copy the colours of another photo")
            print(f"  {i}. {o:10s} - {desc}")
        k = int(ask("Your choice: ", {str(i) for i in range(1, len(opts) + 1)})) - 1
        style = opts[k]
        if style == "colour":
            style = ask_hex()
        elif style == "reference":
            ref = ask_reference()
        edits.append((region, {CATS[region]}, style))
    return edits, ref


def describe(edits):
    return ", ".join(f"{r} -> {s}" for r, _, s in edits)


def main():
    print(GUIDE)
    out_dir = ask("Output folder [press Enter for 'results']: ", allow_empty=True) or "results"
    os.makedirs(out_dir, exist_ok=True)

    print("\nLoading the model ...")
    from run_edit import Labeller
    from semantics import CategoryMap
    labeller, cm = Labeller(mode="hybrid"), CategoryMap()
    print("Ready.")

    while True:
        path, bgr = ask_photo()
        h, w = bgr.shape[:2]
        s = min(1.0, 1600 / max(h, w))
        if s < 1:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = os.path.splitext(os.path.basename(path))[0]
        base = os.path.join(out_dir, name)

        print("Analysing the photo ...")
        cat = labeller(bgr)
        cv2.imwrite(base + "_regions.jpg", cm.overlay(rgb, cat)[:, :, ::-1])
        cov = {r: float((cat == i).mean() * 100) for r, i in CATS.items()}
        present = {r: v for r, v in cov.items() if v >= 0.2}
        print(f"Regions found (preview saved to {base}_regions.jpg):")
        for r in CATS:
            if r in present:
                print(f"  - {REGION_INFO[r].split(' (')[0]:18s} {present[r]:5.1f}%")
        missing = [REGION_INFO[r].split(' (')[0] for r in CATS if r not in present]
        if missing:
            print(f"  (not found, so not offered: {', '.join(missing)})")

        while True:
            edits, ref = choose_edits(present)
            print(f"\nApplying: {describe(edits)} ...")
            out, alpha, hard = translate_regions(rgb, cat, edits, ref=ref)
            tag = "_".join(f"{r}-{s.lstrip('#')}" for r, _, s in edits)
            cv2.imwrite(f"{base}_{tag}_edited.jpg", out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(f"{base}_{tag}_mask.png", (alpha * 255).astype(np.uint8))
            th = 480
            tiles = [bgr, np.repeat((alpha * 255).astype(np.uint8)[..., None], 3, 2), out[:, :, ::-1]]
            tiles = [cv2.resize(t, (round(t.shape[1] * th / t.shape[0]), th)) for t in tiles]
            cv2.imwrite(f"{base}_{tag}_compare.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])

            band = max(3, int(0.02 * min(rgb.shape[:2])))
            far, allun = outside_mask_l1(rgb, out, hard, band)
            print("Done.")
            print(f"  Result:   {base}_{tag}_edited.jpg")
            print(f"  Compare:  {base}_{tag}_compare.jpg")
            print(f"  Changed area: {hard.mean() * 100:.1f}% of the photo, "
                  f"average change there {in_mask_change(rgb, out, hard):.0f}/255")
            print(f"  Rest of the photo: average change {allun:.2f}/255 "
                  f"({far:.2f} away from the edges), edges kept {boundary_f(rgb, out) * 100:.0f}%")

            nxt = ask("\nWhat next?  1 = another change on this photo,  2 = new photo,  q = quit: ", {"1", "2"})
            if nxt == "2":
                break


if __name__ == "__main__":
    main()
