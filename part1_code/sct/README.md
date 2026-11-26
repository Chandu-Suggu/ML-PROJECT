# Semantic Controlled Image Translation

## Quick start: the interactive app
```
pip install -r requirements.txt
python app.py
```
The app prints a user guide (what can be changed, limitations, tips), asks
for an output folder and a photo, shows which regions it found (with a
`_regions.jpg` preview), then asks which part(s) to change and the new look
for each. The result is saved as `<photo>_<edits>_edited.jpg`, with the edit
mask and an input | mask | result comparison. Type `q` at any prompt to quit.

**Step-by-step instructions for everything (install, app, batch tools, retraining,
troubleshooting) are in `RUN_GUIDE.md`.** The sections below describe the method.

Segments a photo into six categories, gives each its own colour, then
"translates" the photo using that semantic map.

| Category | Colour | Cityscapes classes absorbed |
|---|---|---|
| Road & paths | purple | road, sidewalk |
| Sky | blue | sky |
| Trees & greenery | green | vegetation, terrain (grass) |
| People | red | person, rider |
| Buildings | orange | building, wall, fence |
| Foreign objects | yellow | everything else: cars, bikes, buses, poles, signs, traffic lights… |

Edit `categories.json` to change colours or move classes between groups.

## Setup
```
pip install -r requirements.txt
```
No GPU or PyTorch needed. The model (`models/ddrnet23_slim.onnx` + `.data`,
23 MB) is included.

## Run
```
python run.py --input test_images/          # whole folder
python run.py --input my_campus_photo.jpg   # one photo
```
Outputs go to `outputs/` (one set per photo):

| File | What it is |
|---|---|
| `*_mask.png` | flat colour semantic map |
| `*_overlay.jpg` | map blended on the photo, with outlines |
| `*_translated.jpg` | each region recoloured to its category colour, keeping the photo's light/shadow/texture |
| `*_classes19.png` | raw 19-class Cityscapes map (debugging) |
| `*_panel.jpg` | everything side by side with legend and % coverage |
| `summary.json` | % coverage of each category per photo |

### Useful options
| Option | Effect |
|---|---|
| `--ground-prior 60` | (on by default) in the lower half, "building" pixels that score close to a ground class become ground; fixes dry sandy soil and close-up asphalt read as sandstone buildings. `0` = off |
| `--detail 448` | also averages a softened copy; helps sharp close-up phone photos (pavement read as building) |
| `--zooms 1.0,0.6` | adds a zoomed-out pass; can help when objects fill the frame |
| `--unknown-margin 10` | low-confidence pixels become *Foreign objects* (experimental: catches unfamiliar objects but also some shadows) |
| `--texture 0.3` | weaker photo texture in the translated image (0 = flat colours) |
| `--no-tta` | skip mirrored-copy averaging (2× faster) |
| `--no-cleanup` | keep tiny speckle regions |

## How it works (pipeline stages)
1. **Model**: DDRNet23-Slim trained on Cityscapes (19 street classes), int8 ONNX
   export from Qualcomm AI Hub Models, MIT licence (`models/DDRNET_LICENSE`).
2. **Preprocess**: fit the photo into the 2048×1024 input keeping its aspect ratio, pad the rest.
3. **Inference**: 19 score maps at 1/8 resolution, averaged with the mirrored photo,
   then upsampled smoothly to full size; best class per pixel.
4. **Group**: 19 classes → 6 categories (`categories.json`).
5. **Clean up**: regions smaller than 0.02 % of the image are absorbed by their
   surroundings (People and Foreign objects are protected so distant people survive).
6. **Translate**: in LAB colour space, each region takes its category's colour
   (a, b) while lightness keeps the photo's normalised detail, so texture and shading show through.
7. **Report**: legend panel + coverage percentages.

## Known limitations
The model was trained only on dashcam footage from German cities. It does best on
eye-level street views. Expect errors on:
- objects that aren't among the 19 classes (benches, dustbins, animals), which get forced into the
  nearest class (a bench can become "People"); `--unknown-margin` partly helps;
- sharp close-up pavement or tiles, which can be read as building; `--detail 448` helps;
- elevated / top-down views.

For better accuracy on varied campus scenes, a stronger next step is a model trained on
ADE20K (150 classes, indoor + outdoor, e.g. SegFormer or Mask2Former) or fine-tuning on a
few dozen labelled campus photos.

---

# Poster version: region-selective translation + trained refinement

The project now follows the poster *Semantic-Controlled Image Translation:
Region-Aware, Structure-Preserving Image-to-Image Translation*.

## Region-selective translation (`run_edit.py`, `edit.py`)
```
python run_edit.py --input campus --edit sky=sunset
python run_edit.py --input photo.jpg --edit trees=autumn --edit road=wet
python run_edit.py --input photo.jpg --edit buildings=#3A6EA5
python run_edit.py --input photo.jpg --edit sky=reference --ref my_sunset.jpg
```
Regions: `road, sky, trees, people, buildings, foreign`.
Styles: sky `sunset|night|overcast|clear`; trees `autumn|lush|winter`;
road `wet|snow`; any region `#RRGGBB` (recolour, texture kept) or `reference`.

How the selected region is kept exact:
1. **Edit mask** from the category map (input + semantic map + edit/target label, as in the poster).
2. **Colour-aware boundary**: in a thin band around the mask edge each pixel is
   re-decided by comparing its colour with the *local* average colour of the
   selected region and of its neighbours (removes halos).
3. **Sky gap-fill**: sky seen through gaps in tree canopies is also selected.
4. **Guided filter** (He et al.) for a soft edge that follows real photo edges.
5. **Clamp**: alpha = 0 beyond the band, so far-away pixels are copied unchanged.
6. **Composite**: `out = alpha * edited + (1 - alpha) * input`.
7. **Sky connectivity rule**: "sky" blobs not connected to the top of the frame
   (e.g. a white cabinet) are given back to their surroundings.
8. **Building support rule**: "building" blobs in the lower part of the photo
   that touch no sky (e.g. dry sandy ground) are given to their surroundings.

## Training (`head.py`, `train/`)
A gradient-boosted **refinement head** (scikit-learn) is trained on the
**CamVid** dataset (367 train / 101 val / 233 test labelled street images),
using DDRNet's 19 scores plus colour, position and texture features per pixel.

Held-out **CamVid test** results (233 images, 6 categories):

| Method | mIoU | Pixel acc. |
|---|---|---|
| DDRNet only | 53.3 | 76.3 |
| + ground prior | 54.6 | 78.7 |
| + trained head | **74.8** | **91.9** |
| Hybrid (head only where DDRNet is unsure, τ=30, tuned on val) | 67.0 | 85.8 |

On the campus photos the full head carries CamVid's habits (e.g. white surfaces
→ sky), so the default labelling mode is **hybrid** (`--mode hybrid|head|ddrnet`).
Reproduce: `cd train && python cache_scores.py <CamVid> train <cache>` (and val,
test), then `python train_head.py`, `python eval_test.py 0 117`, `117 233`, then `python eval_test.py report`.

## Poster metrics (`metrics.py`), campus photos, averaged over 7 photos
See `out_edit/summary.md`. FID and LPIPS need pretrained Inception/AlexNet
weights that were not available offline, so they are not reported.
