# Semantic-Controlled Image Translation: Complete Run Guide

This guide explains how to set up and run every part of the project:
the interactive app, the two batch tools, and (optionally) retraining the model.

---

## 1. What is in the project

The project comes as four zip files. **Unzip all of them into the same place**: they
share one top-level folder, `sct/`, and fill in different sub-folders.

| Zip | Contents | Needed to run? |
|---|---|---|
| `part1_code.zip` | All code, both models, campus photos, sample photos, guides | **Yes** |
| `part2_segmentation_results.zip` | First-version results: 6-colour masks and panels | No (results) |
| `part3_region_edit_results.zip` | Region-selective translation results and metrics | No (results) |
| `part4_experiments.zip` | Ground-prior on/off and detail experiments | No (results) |

After unzipping:

```
sct/
├── app.py                 Interactive app (start here)
├── run.py                 Batch tool 1: colour-code whole photos (6 categories)
├── run_edit.py            Batch tool 2: region-selective translation + metrics
├── segmenter.py           DDRNet model loading, preprocessing, inference, ground prior
├── head.py                Trained refinement model (features + prediction)
├── semantics.py           Category grouping, cleanup, sky/building rules, rendering
├── edit.py                Edit mask (colour-aware, guided filter) and the styles
├── metrics.py             Outside-mask L1, boundary F-score, layout mIoU
├── categories.json        The 6 categories, their colours and model classes
├── requirements.txt       Python packages
├── README.md              Technical description of the method and results
├── RUN_GUIDE.md           This guide
├── models/
│   ├── ddrnet23_slim.onnx + ddrnet23_slim.data   Segmentation network (Cityscapes, MIT licence)
│   ├── DDRNET_LICENSE
│   └── head.joblib        Trained refinement model (CamVid)
├── train/                 Scripts to retrain and evaluate the refinement model
├── campus/                The 7 IIT Jodhpur test photos (test_1.jpg … test_7.jpg)
├── test_images/           4 sample street photos used during development
├── outputs/  out_campus/                    (part 2) segmentation results
├── out_edit/                                (part 3) region-edit results + summary.md
└── out_campus_gp0/  out_campus_d448/        (part 4) experiments
```

---

## 2. Install (once)

**You need:** Python 3.9 or newer, on Windows, macOS or Linux. No GPU is needed.

1. Open a terminal (Windows: *Command Prompt* or *PowerShell*; macOS/Linux: *Terminal*)
   and go into the project folder:
   ```
   cd path/to/sct
   ```
2. (Recommended) create a virtual environment so the packages don't clash with others:
   ```
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
   This installs onnxruntime, opencv-python, numpy, pillow, scikit-learn 1.9.1 and joblib.
   **scikit-learn must be version 1.9.1**, because the trained model file was saved with it.

4. Check it works:
   ```
   python run.py --input campus/test_7.jpg
   ```
   You should see a line with coverage percentages, and new files in `outputs/`.

On macOS/Linux, if `python` is not found, use `python3` everywhere in this guide.

---

## 3. The interactive app (main way to use the project)

```
python app.py
```

What happens, step by step:

1. **The user guide is printed first**: what the app does, which parts can be changed,
   the available looks, limitations and tips. Read it before choosing.
2. **Output folder:** press Enter to use `results/`, or type another name.
3. **Photo:** type the path to a .jpg or .png, e.g. `campus/test_7.jpg`.
4. **Regions found:** the app lists the regions it detected with their share of the
   photo, and saves a preview, `results/<photo>_regions.jpg`. Only regions found
   in the photo are offered.
5. **Which part to change:** type a number, or several separated by commas (`2,3`).
6. **New look for each part:** type the number of a style. For `colour` you then type
   a hex code like `#3A6EA5`; for `reference` you give the path of another photo
   whose colours should be copied.
7. **Result:** the app saves the files and prints how much changed inside and outside
   the selected area.
8. **What next:** `1` = another change on the same photo, `2` = new photo, `q` = quit.
   You can type `q` at any prompt.

Available looks:

| Region | Looks |
|---|---|
| Sky | sunset, night, overcast, clear |
| Trees & greenery | autumn, lush, winter |
| Road & paths | wet, snow |
| Buildings, People, Foreign objects | any colour (texture kept) |
| Any region | colour (hex code), reference photo |

Example session:

```
Output folder [press Enter for 'results']:
Path to your photo (or q to quit): campus/test_7.jpg
Regions found (preview saved to results/test_7_regions.jpg):
  - Road & paths        31.5%
  - Sky                 32.2%
  - Trees & greenery    25.4%
  ...
Which part do you want to change?   -> 2,3
New look for Sky:                   -> 4   (sunset)
New look for Trees & greenery:      -> 1   (autumn)
Done.
  Result:   results/test_7_sky-sunset_trees-autumn_edited.jpg
```

Files written for each change:

| File | What it is |
|---|---|
| `<photo>_<edits>_edited.jpg` | **The result**: only the chosen regions changed |
| `<photo>_<edits>_mask.png` | Where the change was applied (white = changed) |
| `<photo>_<edits>_compare.jpg` | Input, mask and result side by side |
| `<photo>_regions.jpg` | What the app detected in the photo |

---

## 4. Batch tool 1: colour-code whole photos (`run.py`)

Labels every pixel with one of the 6 categories and produces colour-coded images.
Useful for the segmentation part of the report.

```
python run.py --input campus/                   # all photos in a folder
python run.py --input campus/test_3.jpg         # one photo
python run.py --input campus/ --output my_out   # choose the output folder
```

Outputs per photo (default folder `outputs/`):

| File | What it is |
|---|---|
| `<name>_mask.png` | Flat colour map of the 6 categories |
| `<name>_overlay.jpg` | Map blended on the photo, with outlines |
| `<name>_translated.jpg` | Every region recoloured to its category colour, texture kept |
| `<name>_classes19.png` | Raw 19-class Cityscapes map (debugging) |
| `<name>_panel.jpg` | All of the above with a legend and % coverage |
| `summary.json` | % coverage per category for every photo |

Options:

| Option | Effect |
|---|---|
| `--ground-prior 60` | (default on) fixes dry soil and close-up asphalt read as building; `0` = off |
| `--detail 448` | also averages a softened copy; can help very sharp close-up photos |
| `--zooms 1.0,0.6` | adds a zoomed-out pass |
| `--unknown-margin 10` | low-confidence pixels become Foreign objects (experimental) |
| `--texture 0.3` | weaker texture in `_translated.jpg` (0 = flat colours) |
| `--no-tta` | 2× faster, slightly less accurate |
| `--no-cleanup` | keep tiny speckle regions |

Note: `run.py` uses the original segmentation pipeline (DDRNet + ground prior). The
app and `run_edit.py` use the improved hybrid labelling with the trained model.

---

## 5. Batch tool 2: region-selective translation with metrics (`run_edit.py`)

The same translation as the app, but for many photos at once and with the poster's
evaluation metrics against a whole-image baseline.

```
python run_edit.py --input campus/ --edit sky=sunset
python run_edit.py --input campus/ --edit trees=autumn --edit road=wet
python run_edit.py --input campus/test_4.jpg --edit "buildings=#3A6EA5"
python run_edit.py --input campus/test_7.jpg --edit sky=reference --ref my_sunset.jpg
python run_edit.py --input campus/ --edit sky=night --mode ddrnet --output out_ddrnet
```

`--edit` is `region=style`, repeatable. Regions: `road, sky, trees, people, buildings, foreign`.

`--mode` chooses how the scene is labelled:
- `hybrid` (default): trained model used only where DDRNet is unsure
- `head`: trained model everywhere
- `ddrnet`: no trained model

Outputs per photo (default folder `out_edit/`):

| File | What it is |
|---|---|
| `<name>_edited.jpg` | **The result** (region-selective) |
| `<name>_global.jpg` | Baseline: the same style applied to the whole photo |
| `<name>_alpha.png` | The soft edit mask |
| `<name>_compare.jpg` | Input, mask, our result, baseline (with outside-mask L1 in the labels) |
| `metrics.json` | All metrics for ours and the baseline |

Metrics in `metrics.json`:

| Metric | Meaning | Better |
|---|---|---|
| `outside_L1_all_unselected` | Average pixel change (0–255) outside the selected region | Lower |
| `outside_L1_beyond_band` | Same, ignoring the thin soft-edge band | Lower (ideally 0) |
| `in_region_change` | Average change inside the region (proves the edit happened) | n/a |
| `boundary_F` | Are the photo's edges still in the same places? (0–1) | Higher |
| `layout_mIoU_unselected` | Re-segment the output: is the rest of the scene still recognised the same? | Higher |

To rebuild the summary table in `out_edit/summary.md`, run the three edits used in
the report:
```
python run_edit.py --input campus --edit sky=sunset   --output out_edit/sky_sunset   --max-side 1200
python run_edit.py --input campus --edit trees=autumn --output out_edit/trees_autumn --max-side 1200
python run_edit.py --input campus --edit road=wet     --output out_edit/road_wet     --max-side 1200
```

---

## 6. Changing categories and colours (`categories.json`)

Each entry has a name, a colour `[R, G, B]` and the Cityscapes classes it absorbs:

```json
{"name": "Buildings", "color": [255, 150, 30], "classes": ["building", "wall", "fence"]}
```

The 19 Cityscapes classes: road, sidewalk, building, wall, fence, pole, traffic light,
traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train,
motorcycle, bicycle. Any class not listed goes to the `fallback` category
(Foreign objects).

Change colours freely. If you move classes between categories, keep the six
categories in the same order: the region-editing code and the trained model refer to
them by position.

---

## 7. Retraining the refinement model (optional)

Only needed to reproduce the training results; the trained model is already included
as `models/head.joblib`. Timings are for a 2-core laptop CPU.

**Step 1: get CamVid** (about 180 MB, from GitHub; needs `git`):
```
git clone --depth 1 --filter=blob:none --sparse https://github.com/alexgkendall/SegNet-Tutorial data/segnet
cd data/segnet
git sparse-checkout set CamVid
cd ../..
```
Then move or link `data/segnet/CamVid` to `data/CamVid`, or set an environment
variable instead:
- Windows: `set CAMVID_DIR=C:\path\to\sct\data\segnet\CamVid`
- macOS / Linux: `export CAMVID_DIR=$PWD/data/segnet/CamVid`

**Step 2: cache DDRNet's scores** (about 18 min, 2.3 GB on disk):
```
cd train
python cache_scores.py ../data/CamVid train ../data/cache
python cache_scores.py ../data/CamVid val   ../data/cache
python cache_scores.py ../data/CamVid test  ../data/cache
```
If you used a different folder, pass it here and set `CACHE_DIR` to the same folder.
Interrupted runs can be restarted; finished images are skipped.

**Step 3: measure the untrained pipeline** (about 1 min):
```
python eval_baseline.py
```

**Step 4: train** (about 2 min of training plus about 10 min of validation per setting):
```
python train_head.py
```
This tries two settings, keeps the better one on the validation split, and
overwrites `models/head.joblib`. Back up the included file first if you want to keep it.

**Step 5: held-out test results** (about 6 min per half):
```
python eval_test.py 0 117
python eval_test.py 117 233
python eval_test.py report
```

**Step 6 (optional): tune the hybrid threshold on validation:**
```
python sweep_hybrid.py
```

Expected test results (233 CamVid images, 6 categories):

| Method | mIoU | Pixel accuracy |
|---|---|---|
| DDRNet only | 53.3 | 76.3 |
| + ground prior | 54.6 | 78.7 |
| + trained head | 74.8 | 91.9 |
| Hybrid (τ = 30) | 67.0 | 85.8 |

---

## 8. Results included

| Folder | What it shows |
|---|---|
| `out_campus/` | 6-colour segmentation of the 7 campus photos (panels, masks, overlays), plus `campus_overview.jpg` |
| `outputs/` | Segmentation of the 4 sample street photos |
| `out_edit/sky_sunset`, `trees_autumn`, `road_wet` | Region-selective translation of all 7 campus photos, baseline, masks, metrics |
| `out_edit/summary.md` | Metrics table averaged over the 7 photos |
| `out_campus_gp0/` | Segmentation with the ground prior **off** (compare with `out_campus/`) |
| `out_campus_d448/` | Segmentation with `--detail 448` |

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: onnxruntime` (or cv2, sklearn) | Run `pip install -r requirements.txt` inside the activated virtual environment |
| Warning or error loading `head.joblib` | Install exactly `scikit-learn==1.9.1`, or run with `--mode ddrnet` (no trained model) |
| `cannot read …` / photo not found | Check the path; put it in quotes if it contains spaces |
| Very slow | Normal speed is 2–5 s per photo. Use `--no-tta` with `run.py`, or `--max-side 1000` with `run_edit.py` |
| Out of memory on huge photos | Photos are reduced to 1600 px (app) or 2048 px (`run.py`); lower `--max-side` |
| Wrong region changed | Open `_regions.jpg` to see what was detected. The model can misread unusual objects; see the limitations in the app's guide |
| A colour edit doesn't take effect | Write the colour with 6 hex digits, e.g. `#FF8800` |
