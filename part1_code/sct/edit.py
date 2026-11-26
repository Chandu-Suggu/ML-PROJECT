"""Region-selective, structure-preserving translation (the poster's core idea).

Inputs:  photo + semantic category map + EDIT SPEC (which categories -> which target style)
Output:  photo where only the selected regions are translated; every pixel
         outside the selection is copied from the original.

Pipeline for each edit
  1. hard mask  = pixels whose category is selected
  2. soft alpha = guided filter of the hard mask, guided by the photo, so the
                  boundary follows real edges (branches against sky, roof lines)
                  then clamped: 0 beyond a small band outside the mask,
                  1 deep inside it  -> nothing far from the selection changes
  3. edited     = the target style applied to the whole photo (cheap, per pixel)
  4. output     = alpha * edited + (1 - alpha) * original

Styles keep structure: they change colour (LAB a/b channels) and tone, and
keep the photo's lightness detail, so edges and texture survive.
"""
import cv2
import numpy as np

CATS = {"road": 0, "sky": 1, "trees": 2, "people": 3, "buildings": 4, "foreign": 5}


# ---------------- edit mask ----------------
def _box(x, r):
    return cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r=8, eps=1e-3):
    """He et al. guided filter (grey guide). guide, src: float32 in [0, 1]."""
    mI, mp = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mI * mp
    var = _box(guide * guide, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return _box(a, r) * guide + _box(b, r)


def edit_alpha(rgb, cat, classes, band_frac=0.02, r_frac=0.01):
    """Soft, edge- and colour-aware selection mask in [0, 1].

    1. hard mask from the segmentation
    2. in a band around its boundary, re-decide each pixel by colour: is it
       closer to the selected region's colours or to its neighbours' colours?
       (fixes halos, e.g. sky seen between leaves labelled as tree)
    3. guided filter for a soft edge that follows the photo's real edges
    4. clamp: 0 outside the band, 1 deep inside -> far pixels never change
    """
    hard = np.isin(cat, list(classes))
    if not hard.any():
        return hard.astype(np.float32)
    h, w = hard.shape
    band = max(3, int(band_frac * min(h, w)))
    r = max(4, int(r_frac * min(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    h8 = hard.astype(np.uint8)
    near = cv2.dilate(h8, k) > 0               # may change: mask + band
    core = cv2.erode(h8, k) > 0                # certainly selected
    zone = near & ~core                        # undecided boundary band
    lab = _lab(rgb)
    soft = hard.astype(np.float32)
    outside = (cv2.dilate(h8, k, iterations=3) > 0) & ~near   # clearly unselected, nearby
    if core.any() and outside.any():
        # local colour models: average colour of nearby certain-selected and
        # nearby certain-unselected pixels (handles gradient skies, varied backgrounds)
        win = 4 * band + 1
        def local_mean(sel):
            m = sel.astype(np.float32)
            num = cv2.boxFilter(lab * m[..., None], -1, (win, win), normalize=False)
            den = cv2.boxFilter(m, -1, (win, win), normalize=False)
            return num / np.maximum(den, 1e-3)[..., None], den
        mu_s, ns = local_mean(core)
        mu_b, nb = local_mean(outside)
        x = lab[zone]
        ds = ((x - mu_s[zone]) ** 2).sum(-1)
        db = ((x - mu_b[zone]) ** 2).sum(-1)
        p = db / (ds + db + 1e-6)                       # 1 = looks like the selection
        ok = (ns[zone] > 0.5) & (nb[zone] > 0.5)        # both models available here
        prior = hard[zone].astype(np.float32)
        soft[zone] = np.where(ok, 0.8 * p + 0.2 * prior, prior)
    gap = np.zeros_like(hard)
    if 1 in classes and core.sum() > 100:
        # sky seen through gaps in tree canopies (and around poles) is labelled
        # tree / foreign by the segmenter: take those pixels if their colour
        # matches the sky's
        x = lab[core]
        mu, sd = x.mean(0), x.std(0) + 2.0
        z = np.abs((lab - mu) / sd).max(-1)
        gap = np.isin(cat, [2, 5]) & ~hard & (z < 2.0)
        soft[gap] = 1.0
        near = near | gap
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    alpha = np.clip(guided_filter(grey, soft, r=r, eps=1e-3), 0, 1)
    alpha = np.where(near, alpha, 0.0)
    alpha = np.where(core | gap, 1.0, alpha)
    return alpha.astype(np.float32)


# ---------------- styles ----------------
def _lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _rgb(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _hex_lab(hexcol):
    c = np.array([[[int(hexcol[i:i + 2], 16) for i in (0, 2, 4)]]], np.uint8)
    return cv2.cvtColor(c, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]


def _detail(L, mask):
    """Photo lightness detail, normalised inside the mask (mean 0, std 1)."""
    v = L[mask > 0.5] if (mask > 0.5).any() else L.ravel()
    return (L - v.mean()) / (v.std() + 1e-6)


def _gradient(h, w, top_hex, bottom_hex, mask):
    """Vertical colour gradient (LAB) spanning the mask's own rows."""
    rows = np.where(mask.max(axis=1) > 0.5)[0]
    y0, y1 = (rows.min(), rows.max()) if rows.size else (0, h - 1)
    t = np.clip((np.arange(h) - y0) / max(y1 - y0, 1), 0, 1)[:, None, None]
    top, bot = _hex_lab(top_hex), _hex_lab(bottom_hex)
    return np.broadcast_to(top * (1 - t) + bot * t, (h, w, 3)).copy()


def style_sky(rgb, mask, kind):
    lab = _lab(rgb)
    h, w = mask.shape
    d = _detail(lab[..., 0], mask)            # clouds survive as lightness detail
    pal = {"sunset": ("3B2A6B", "FF9A4D", 14), "night": ("0B1026", "27365E", 8),
           "overcast": ("B9BEC6", "D8DBDF", 10), "clear": ("2F7FE0", "A8D4FF", 10)}
    top, bot, k = pal[kind]
    g = _gradient(h, w, top, bot, mask)
    g[..., 0] = g[..., 0] + k * d
    if kind == "night":                       # a few faint stars
        rng = np.random.default_rng(0)
        s = rng.random((h, w)) > 0.9993
        g[..., 0] = np.where(s, 230, g[..., 0])
    return _rgb(g)


def style_trees(rgb, mask, kind):
    lab = _lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    if kind == "autumn":                      # green -> orange/red, keep shading
        a2, b2 = 128 + 0.35 * (a - 128) + 32, 128 + 0.5 * (b - 128) + 45
        return _rgb(np.dstack([L * 0.95, a2, b2]))
    if kind == "lush":                        # deeper, more saturated green
        return _rgb(np.dstack([L * 0.92, 128 + 1.6 * (a - 128) - 10, 128 + 1.4 * (b - 128) + 6]))
    if kind == "winter":                      # frosted, pale
        return _rgb(np.dstack([np.clip(L * 0.6 + 105, 0, 255), 128 + 0.15 * (a - 128), 128 + 0.15 * (b - 128) - 4]))
    raise ValueError(kind)


def style_road(rgb, mask, kind):
    lab = _lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    if kind == "wet":                         # darker, more contrast, cool sheen
        d = _detail(L, mask)
        L2 = np.clip(L * 0.55 + 14 * d, 0, 255)
        return _rgb(np.dstack([L2, 128 + 0.5 * (a - 128), 128 + 0.5 * (b - 128) - 8]))
    if kind == "snow":
        d = _detail(L, mask)
        return _rgb(np.dstack([np.clip(225 + 12 * d, 0, 255), np.full_like(a, 127), np.full_like(b, 124)]))
    raise ValueError(kind)


def style_colour(rgb, mask, hexcol, texture=0.65):
    """Recolour to any colour, keeping texture (same idea as the translated panel)."""
    lab = _lab(rgb)
    t = _hex_lab(hexcol)
    d = _detail(lab[..., 0], mask)
    return _rgb(np.dstack([t[0] + texture * 28 * d, np.full_like(d, t[1]), np.full_like(d, t[2])]))


def style_reference(rgb, mask, ref_rgb, ref_mask=None):
    """Example-based: match the region's LAB mean/std to a reference photo (region)."""
    lab, ref = _lab(rgb), _lab(ref_rgb)
    sel = mask > 0.5
    rsel = ref_mask > 0.5 if ref_mask is not None else np.ones(ref.shape[:2], bool)
    out = lab.copy()
    for c in range(3):
        m, s = lab[..., c][sel].mean(), lab[..., c][sel].std() + 1e-6
        rm, rs = ref[..., c][rsel].mean(), ref[..., c][rsel].std() + 1e-6
        out[..., c] = (lab[..., c] - m) / s * rs + rm
    return _rgb(out)


STYLES = {
    "sky": {"sunset", "night", "overcast", "clear"},
    "trees": {"autumn", "lush", "winter"},
    "road": {"wet", "snow"},
}


def apply_style(rgb, mask, target, style, ref=None):
    if style.startswith("#"):
        return style_colour(rgb, mask, style[1:])
    if style == "reference":
        return style_reference(rgb, mask, ref)
    if target == "sky":
        return style_sky(rgb, mask, style)
    if target == "trees":
        return style_trees(rgb, mask, style)
    if target == "road":
        return style_road(rgb, mask, style)
    raise ValueError(f"style '{style}' is not available for '{target}'. "
                     f"Options: {sorted(STYLES.get(target, []))}, '#RRGGBB' or 'reference'")


# ---------------- full edit ----------------
def parse_edits(specs):
    """['sky=sunset', 'trees=autumn', 'buildings=#3366FF'] -> [(target, cat_ids, style)]"""
    out = []
    for s in specs:
        target, style = s.split("=", 1)
        target = target.strip().lower()
        if target not in CATS:
            raise ValueError(f"unknown region '{target}', choose from {list(CATS)}")
        out.append((target, {CATS[target]}, style.strip()))
    return out


def translate_regions(rgb, cat, edits, ref=None, global_baseline=False):
    """Apply each edit only inside its region. Returns (output, union_alpha, union_hard_mask).

    global_baseline=True applies the same styles to the WHOLE image (a
    non-region-aware translation, used as the comparison baseline)."""
    out = rgb.astype(np.float32)
    union_a = np.zeros(cat.shape, np.float32)
    union_m = np.zeros(cat.shape, bool)
    for target, ids, style in edits:
        hard = np.isin(cat, list(ids))
        alpha = np.ones(cat.shape, np.float32) if global_baseline else edit_alpha(rgb, cat, ids)
        style_mask = hard.astype(np.float32) if hard.any() else np.ones(cat.shape, np.float32)
        edited = apply_style(out.clip(0, 255).astype(np.uint8), style_mask, target, style, ref).astype(np.float32)
        out = alpha[..., None] * edited + (1 - alpha[..., None]) * out
        union_a = np.maximum(union_a, alpha)
        union_m |= hard
    return out.clip(0, 255).astype(np.uint8), union_a, union_m
