# Semantic-Controlled Image Translation

**Region-aware, structure-preserving translation of campus photographs.**

Pick a part of a photo (sky, trees, road, buildings, people or foreign objects) and a new look for it
(e.g. *sunset sky*, *autumn trees*, *wet road*). Only that part changes; every other pixel stays exactly
as it was. Everything runs offline on a normal laptop CPU: no GPU, no internet.

| Team member | Roll number |
|---|---|
| Suggu Chandu | M26CSA017 |
| Ayush Kumar Verma | M26MER010 |
| Aditya Sahu | M26CPS001 |

Indian Institute of Technology Jodhpur

---

## Contents
1. [What it does](#1-what-it-does)
2. [Novelty](#2-novelty)
3. [How it works](#3-how-it-works)
4. [Results](#4-results)
5. [Installation](#5-installation)
6. [Quick start: the interactive app](#6-quick-start-the-interactive-app)
7. [Batch tools](#7-batch-tools)
8. [Training and evaluation (reproducing the numbers)](#8-training-and-evaluation-reproducing-the-numbers)
9. [Repository layout](#9-repository-layout)
10. [Report](#10-report)
11. [Limitations](#11-limitations)
12. [Credits and licences](#12-credits-and-licences)

---

## 1. What it does

**Input:** a photo, a region to change, and a target style.
**Output:** the same photo with only that region translated.

| Region | Styles |
|---|---|
| Sky | `sunset`, `night`, `overcast`, `clear` |
| Trees & greenery | `autumn`, `lush`, `winter` |
| Road & paths | `wet`, `snow` |
| Buildings, People, Foreign objects | any colour, `#RRGGBB` (texture kept) |
| Any region | `reference`: copy the colours of another photo |

The project also produces colour-coded semantic maps of photos in six categories:

| Category | Colour |
|---|---|
| Road & paths | purple |
| Sky | blue |
| Trees & greenery | green |
| People | red |
| Buildings | orange |
| Foreign objects (vehicles, bikes, poles, signs, …) | yellow |

## 2. Novelty

The segmentation network (DDRNet-23-Slim) is a published model; the novelty is the system built around it.

| Novelty | How it is implemented | Code |
|---|---|---|
| **Exact region-only translation** | `output = α · restyled + (1 − α) · input`, with α forced to 0 beyond a thin band around the region, so those pixels are copied unchanged | `edit.py` |
| **Halo-free edit masks** | At the region's edge each pixel is compared with the *local* average colour of the region and of its neighbours, then smoothed with a guided filter; sky seen through gaps in leaves is added | `edit.py` |
| **Geometric scene priors** (no retraining) | Ground prior: in the lower half, "building" becomes ground if a ground class scores within 60. Sky blobs not reaching the top 15 %, and building blobs not touching the sky, go to their surroundings | `segmenter.py`, `semantics.py` |
| **Confidence-gated refinement** | A gradient-boosted classifier trained on 830,000 CamVid pixels (network scores + colour, height, texture), used only where the network is unsure (top-two score gap < 30) | `head.py`, `train/` |
| **Light and interactive** | 8-bit model (23 MB) on CPU with ONNX Runtime; a guided app asks which region to change and how | `app.py` |

## 3. How it works

```
                         SCENE UNDERSTANDING
 photo ─► letterbox 2048×1024 ─► DDRNet-23-Slim ─► 19 class scores ─┬─► ground prior ──────────┐
          + flip averaging        (frozen,                           │                          ▼
                                   Cityscapes)                       └─► refinement head ─► confidence gate
                                                                         (trained, CamVid)        │
                                                                                                  ▼
                                               cleanup + sky-connectivity + building-support rules
                                                                                                  │
                                                                                category map (6 classes)
                         REGION-SELECTIVE TRANSLATION                                             │
 user: region + style ─► edit mask α (colour vote at edges, sky gap fill, guided filter, clamp) ◄─┘
                                   │
                                   ▼
                 output = α · style(photo) + (1 − α) · photo   →  only the chosen region changes
```

1. **Segment.** The photo is fitted into the network's 2048×1024 input without stretching; scores of the
   photo and its mirror image are averaged, upsampled to full size, and grouped into 6 categories
   (`categories.json`).
2. **Correct.** The ground prior and, where the network is unsure, the trained head fix domain errors
   (e.g. dry sandy ground read as a sandstone building). Cleanup and the sky/building rules remove
   implausible regions.
3. **Select.** The user's region gives a hard mask, refined into a soft mask α that follows real edges.
4. **Translate.** The style changes colour (LAB a/b channels) and tone while keeping the photo's lightness
   detail; the result is blended through α.

## 4. Results

**Segmentation, CamVid test set (233 images, 6 categories)**

| Method | mIoU | Pixel accuracy |
|---|---|---|
| DDRNet only | 53.3 | 76.3 |
| + ground prior | 54.6 | 78.7 |
| + trained head (all pixels) | **74.8** | **91.9** |
| Hybrid, τ = 30 | 67.0 | 85.8 |
| Final pipeline (hybrid + rules, used by the app) | 65.6 | 84.9 |

The trained head is the most accurate in-domain, but on the campus photos it carries CamVid's habits
(e.g. white surfaces read as sky), so the app uses the gated hybrid with the geometric rules.

**Region-selective translation, 7 IIT Jodhpur campus photos (means)**

| Edit | Method | Outside-mask L1 ↓ (0–255) | Boundary F ↑ |
|---|---|---|---|
| sky → sunset | global baseline | 63.59 | 0.650 |
| | **ours** | **1.83** | **0.920** |
| trees → autumn | global baseline | 60.00 | 0.962 |
| | **ours** | **0.71** | **0.970** |
| road → wet | global baseline | 44.89 | 0.676 |
| | **ours** | **0.87** | **0.782** |

Outside-mask L1 = average change of the pixels that were *not* selected. Ours is 97–99 % lower than
applying the same style to the whole image; beyond the soft edge band it is 0.44 (sky), 0.00 (trees)
and 0.00 (road).

**Runtime** (1200 px photo, 2-core CPU): labelling 1.99 s without the head, 7.70 s with it;
translation 0.27 s.

All numbers come from the files in `logs/` and `out_edit/*/metrics.json` (see [section 8](#8-training-and-evaluation-reproducing-the-numbers)).

## 5. Installation

Requires **Python 3.9+** (tested with 3.13), Windows, macOS or Linux. No GPU.

```bash
cd sct
python -m venv venv
# Windows:            venv\Scripts\activate
# macOS / Linux:      source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs onnxruntime, opencv-python, numpy, pillow, scikit-learn **1.9.1** and joblib.
scikit-learn must be exactly 1.9.1 to load the trained model (`models/head.joblib`). With another
version, use `--mode ddrnet` (no trained head).

Check the installation:

```bash
python run.py --input campus/test_7.jpg
```

This should print coverage percentages and write images to `outputs/`.
On macOS/Linux use `python3` if `python` is not found.

## 6. Quick start: the interactive app

```bash
python app.py
```

1. A **user guide** is printed first: what can be changed, the available styles, limitations and tips.
2. Press Enter for the output folder (`results/`), or type another name.
3. Type the path of a photo, e.g. `campus/test_7.jpg`.
4. The app lists the **regions found** with their share of the photo and saves a preview
   (`results/<photo>_regions.jpg`). Only regions present in the photo are offered.
5. Choose the region(s) to change: one number, or several separated by commas (`2,3`).
6. Choose the style for each region. For `colour`, type a hex code such as `#3A6EA5`;
   for `reference`, give the path of another photo.
7. The result is saved and the app reports how much changed inside and outside the region.
8. `1` = another change on the same photo, `2` = a new photo, `q` = quit (works at any prompt).

| Output file | Contents |
|---|---|
| `<photo>_<edits>_edited.jpg` | **the result**: only the chosen regions changed |
| `<photo>_<edits>_mask.png` | where the change was applied (white = changed) |
| `<photo>_<edits>_compare.jpg` | input, mask and result side by side |
| `<photo>_regions.jpg` | what was detected in the photo |

**In VS Code:** File → Open Folder → `sct`, then Terminal → New Terminal and run the commands above.
Use the terminal (or the ▶ Run button), not the debugger console, because the app needs typed input.

## 7. Batch tools

### `run.py`: colour-code whole photos (6 categories)

```bash
python run.py --input campus/                  # every photo in a folder
python run.py --input campus/test_3.jpg        # one photo
python run.py --input campus/ --ground-prior 0 --output out_campus_gp0   # ground prior off
```

Writes `<name>_mask.png`, `<name>_overlay.jpg`, `<name>_translated.jpg` (every region recoloured, texture
kept), `<name>_classes19.png` (raw 19-class map), `<name>_panel.jpg` (all with a legend) and `summary.json`
(coverage per category). Options: `--ground-prior` (default 60, `0` = off), `--detail 448`,
`--zooms 1.0,0.6`, `--unknown-margin 10` (experimental), `--texture 0.3`, `--no-tta`, `--no-cleanup`.
`run.py` uses DDRNet + ground prior (no trained head).

### `run_edit.py`: region-selective translation with metrics

```bash
python run_edit.py --input campus/ --edit sky=sunset
python run_edit.py --input campus/ --edit trees=autumn --edit road=wet
python run_edit.py --input campus/test_4.jpg --edit "buildings=#3A6EA5"
python run_edit.py --input campus/test_7.jpg --edit sky=reference --ref my_sunset.jpg
python run_edit.py --input campus/ --edit sky=night --mode ddrnet
```

`--mode`: `hybrid` (default), `head` or `ddrnet`. `--max-side` limits photo size (default 1600).

| Output | Contents |
|---|---|
| `<name>_edited.jpg` | the result |
| `<name>_global.jpg` | baseline: same style on the whole photo |
| `<name>_alpha.png` | soft edit mask |
| `<name>_compare.jpg` | input, mask, ours, baseline |
| `metrics.json` | outside-mask L1 (all unselected / beyond band), in-region change, boundary F-score, layout mIoU, for ours and the baseline |

### Changing categories and colours

Edit `categories.json`. Each category lists a colour `[R, G, B]` and the Cityscapes classes it absorbs
(road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky,
person, rider, car, truck, bus, train, motorcycle, bicycle). Unlisted classes go to *Foreign objects*.
Keep the six categories in their order: the editing code and the trained head refer to them by position.

## 8. Training and evaluation (reproducing the numbers)

The trained head is included (`models/head.joblib`); these steps are only needed to reproduce it and the
reported numbers. Timings are for a 2-core CPU.

```bash
# Campus results (coverage table, translation table, graphs)
python run.py --input campus --output out_campus
python run.py --input campus --output out_campus_gp0 --ground-prior 0
python run_edit.py --input campus --edit sky=sunset   --output out_edit/sky_sunset   --max-side 1200
python run_edit.py --input campus --edit trees=autumn --output out_edit/trees_autumn --max-side 1200
python run_edit.py --input campus --edit road=wet     --output out_edit/road_wet     --max-side 1200

# CamVid (SegNet release, 480x360; ~180 MB)
git clone --depth 1 --filter=blob:none --sparse https://github.com/alexgkendall/SegNet-Tutorial data/segnet
git -C data/segnet sparse-checkout set CamVid
export CAMVID_DIR=$PWD/data/segnet/CamVid CACHE_DIR=$PWD/data/cache   # Windows: set CAMVID_DIR=...  set CACHE_DIR=...

cd train
python cache_scores.py $CAMVID_DIR train $CACHE_DIR   # also val and test (~18 min total, 2.3 GB)
python eval_baseline.py                               # -> logs/eval_baseline.log
python train_head.py                                  # -> logs/train_head.log (overwrites models/head.joblib)
python sweep_hybrid.py                                # -> logs/sweep_hybrid_val.log (choice of tau)
python eval_test.py 0 117;  python eval_test.py 117 233;  python eval_test.py report    # -> logs/eval_test.log
python eval_final.py 0 117; python eval_final.py 117 233; python eval_final.py report   # -> logs/eval_final.log
cd ..

# Runtime, graphs, tables and figures for the report
python report/timing.py
python report/make_figs.py
python report/make_tables.py
python report/make_qual.py
```

**Training configuration:** scikit-learn `HistGradientBoostingClassifier`, learning rate 0.1,
`max_leaf_nodes` 63 (chosen on validation from {31, 63}), up to 300 trees, L2 1.0, early stopping on a
10 % split, `random_state=0`. Up to 400 pixels per category per training image (NumPy seed 0):
829,596 pixels. 30 features per pixel: 19 network scores, LAB colour and chroma, normalised height,
local brightness variation (7×7 and 21×21), edge strength, 15×15 mean colour. Gate threshold τ = 30.
The full list of settings is in the report's appendix (configuration table).

**Evaluation notes:** IoU from one confusion matrix over all labelled test pixels (CamVid void ignored);
test evaluation predicts the head on a stride-2 grid and upsamples. Model checksums are in
`logs/model_checksums.txt`.

## 9. Repository layout

```
sct/
├── app.py              interactive app (guide → photo → region → style → result)
├── run.py              batch colour-coded segmentation
├── run_edit.py         batch region-selective translation + metrics
├── segmenter.py        DDRNet loading, letterbox, flip averaging, ground prior
├── head.py             refinement head: features and prediction
├── semantics.py        category grouping, cleanup, sky/building rules, rendering, panels
├── edit.py             edit mask (colour vote, gap fill, guided filter) and styles
├── metrics.py          outside-mask L1, boundary F-score, layout mIoU
├── categories.json     categories, colours, class mapping
├── requirements.txt
├── README.md           this file
├── RUN_GUIDE.md        step-by-step guide for beginners (install, VS Code, troubleshooting)
├── models/             ddrnet23_slim.onnx + .data (DDRNet, MIT licence), head.joblib (trained head)
├── train/              CamVid caching, training, evaluation scripts
├── logs/               final logs behind every reported number, model checksums
├── report/             CVPR LaTeX source, figure/table/timing scripts, figures
├── campus/             the 7 IIT Jodhpur test photos
├── test_images/        4 sample street photos
├── out_campus/         segmentation results (ground prior on)
├── out_campus_gp0/     segmentation results (ground prior off)
├── out_edit/           translation results: sky_sunset/, trees_autumn/, road_wet/, summary.md
└── outputs/            segmentation of the sample photos
```

## 10. Report

The CVPR-format report (`report/main.tex`) is built with:

```bash
cd report
latexmk -pdf main.tex
```

It has 6 pages of main text, then references and an appendix with configurations, final logs, per-photo
results, graphs and outputs. Fill in `\repourl` and `\commithash` in `report/preamble.tex` before the
final build.

## 11. Limitations

- Styles change **colour and tone** only; they cannot add or remove objects or change shapes.
- Works best on **daylight, eye-level, outdoor** photos. Indoor, night, aerial or very close-up photos
  may be segmented wrongly.
- Objects outside the 19 Cityscapes classes (benches, mirrors, animals) are forced into a known class.
- Dry, sandy fields on the far horizon can be counted as buildings.
- Thin details (single leaves, wires) can keep a faint trace of their old colour at the edges.
- The head is trained on CamVid (Cambridge, UK), not on campus photos, so its gain does not fully
  transfer; campus accuracy is not measured because the campus photos are unlabelled.
- GAN baselines (pix2pix, CycleGAN, SPADE) and FID/LPIPS were not run: they need GPUs and pretrained
  networks that were not available offline.

**Future work:** label 15–20 campus photos to train and evaluate in-domain; add a bare-ground category;
use the edit mask to condition a generative model.

## 12. Credits and licences

- **DDRNet-23-Slim** (Hong et al., 2021), Cityscapes weights, 8-bit ONNX export from Qualcomm AI Hub,
  MIT licence (`models/DDRNET_LICENSE`), via the `urban-toolkit/curio` repository.
- **CamVid** (Brostow et al., 2009), SegNet release from `alexgkendall/SegNet-Tutorial`.
- **Guided filter** (He et al., 2013); **gradient boosting** via scikit-learn (Pedregosa et al., 2011).
- Campus photographs: taken by the team at IIT Jodhpur.
