"""Stage 1-3: load the pretrained DDRNet23-Slim (Cityscapes) model and turn a
photo into per-pixel class scores / labels.

Pipeline
  photo (any size, BGR from OpenCV)
    -> resize to a working size (long side <= max_side)
    -> letterbox into the model's 2048x1024 canvas (keep aspect ratio, pad)
    -> DDRNet  ->  19 score maps at 1/8 resolution (256x128)
    -> [optional] same for the mirrored photo, flipped back and averaged (TTA)
    -> [optional] extra zoom levels / a softened copy, averaged (--zooms, --detail)
    -> crop away the padding, upsample scores bilinearly to the working size
    -> argmax  ->  label map with values 0..18 (Cityscapes train ids)
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


class Segmenter:
    def __init__(self, model_path=None, threads=0):
        model_path = model_path or os.path.join(ROOT, "models", "ddrnet23_slim.onnx")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.H, self.W = inp.shape          # 1024, 2048
        self.classes = CITYSCAPES_CLASSES

    # ---------- Stage 2: preprocessing ----------
    def _letterbox(self, rgb, zoom=1.0):
        """Fit the image inside HxW keeping aspect ratio (times `zoom` <= 1,
        i.e. zoomed out); pad the remainder."""
        h, w = rgb.shape[:2]
        s = min(self.W / w, self.H / h) * zoom
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        top, left = (self.H - nh) // 2, (self.W - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, (top, left, nh, nw)

    # ---------- Stage 3: inference ----------
    def _run(self, canvas_rgb):
        x = np.ascontiguousarray(canvas_rgb.transpose(2, 0, 1)[None])   # 1x3xHxW uint8
        out = self.sess.run(None, {self.input_name: x})[0][0]           # 19 x H/8 x W/8 (uint8)
        return out.astype(np.float32)

    def _scores_one(self, rgb, zoom, tta):
        canvas, (top, left, nh, nw) = self._letterbox(rgb, zoom)
        sc = self._run(canvas)
        if tta:  # horizontal-flip test-time augmentation
            sc = 0.5 * (sc + self._run(canvas[:, ::-1].copy())[:, :, ::-1])
        sc = sc.transpose(1, 2, 0)                                       # H/8 x W/8 x 19
        # upsample scores to the canvas, crop the padding, resize to the photo
        sc = cv2.resize(sc, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        sc = sc[top:top + nh, left:left + nw]
        h, w = rgb.shape[:2]
        if (nh, nw) != (h, w):
            sc = cv2.resize(sc, (w, h), interpolation=cv2.INTER_LINEAR)
        return sc

    def scores(self, bgr, tta=True, zooms=(1.0,)):
        """Return float32 scores (h, w, 19) aligned with `bgr`, averaged over
        flip (tta) and zoom levels. A zoomed-out pass (0.6) helps close-up,
        on-foot photos, whose objects look larger than in Cityscapes."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return sum(self._scores_one(rgb, z, tta) for z in zooms) / len(zooms)

    def segment(self, bgr, tta=True, zooms=(1.0,), detail=0, ground_prior=60.0):
        """`detail` > 0: also run on a copy whose short side is `detail` px
        and average. Softening very sharp phone photos can help, because
        Cityscapes frames are fairly soft.

        `ground_prior` > 0: in the lower half of the photo, a pixel labelled
        'building' is relabelled to its best ground class (road / sidewalk /
        terrain) if that class scores within `ground_prior` of building.
        Fixes dry sandy soil and close-up asphalt being read as sandstone
        buildings (common on desert campuses, absent from Cityscapes)."""
        sc = self.scores(bgr, tta=tta, zooms=zooms)
        h, w = bgr.shape[:2]
        if detail and detail < min(h, w):
            f = detail / min(h, w)
            small = cv2.resize(bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
            sc = 0.5 * (sc + cv2.resize(self.scores(small, tta=tta, zooms=zooms), (w, h)))
        top2 = np.partition(sc, -2, axis=2)[:, :, -2:]
        margin = top2[:, :, 1] - top2[:, :, 0]
        labels = sc.argmax(axis=2).astype(np.uint8)
        if ground_prior > 0:
            ground = np.array([0, 1, 9])                      # road, sidewalk, terrain
            gsc = sc[:, :, ground]
            lower = (np.arange(h) >= h // 2)[:, None]
            fix = (labels == 2) & lower & (sc[:, :, 2] - gsc.max(axis=2) < ground_prior)
            labels[fix] = ground[gsc.argmax(axis=2)][fix]
        return labels, margin
