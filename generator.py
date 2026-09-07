"""
NetraXAI prototype — synthetic fundus image generator.

Generates realistic-looking synthetic retinal images (no internet / dataset needed)
with configurable DR grade (0-4) and image quality (good / borderline / poor).
Because lesions, vessels and artifacts are explicitly painted, the downstream
detector has a known, controllable ground truth — ideal for pipeline demos.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 512           # square output
RADIUS = SIZE * 0.47  # fundus disc radius (px)
CX = SIZE / 2.0
CY = SIZE / 2.0
OD_X = CX + SIZE * 0.28   # optic disc centre (right side)
OD_Y = CY - SIZE * 0.04
OD_R = SIZE * 0.105        # optic disc radius


def _rnd_point(rnd: random.Random) -> tuple[float, float]:
    """Random point inside the fundus disc (kept away from the very centre)."""
    while True:
        ang = rnd.uniform(0.0, math.tau)
        r = RADIUS * rnd.uniform(0.08, 0.92)
        x = CX + math.cos(ang) * r
        y = CY + math.sin(ang) * r
        if math.hypot(x - OD_X, y - OD_Y) > OD_R * 1.9:
            return x, y


def _vessels(draw: ImageDraw.ImageDraw, rnd: random.Random, color=(134, 26, 26)) -> None:
    """Branching arterial tree radiating from the optic disc."""
    def branch(x: float, y: float, ang: float, length: float, width: float) -> None:
        if length < 7 or width < 0.6:
            return
        x2 = x + math.cos(ang) * length
        y2 = y + math.sin(ang) * length
        if math.hypot(x2 - CX, y2 - CY) < RADIUS * 0.97:
            draw.line([x, y, x2, y2], fill=color, width=max(1, int(round(width))))
            for _ in range(rnd.randint(2, 3)):
                branch(x2, y2, ang + rnd.uniform(-0.55, 0.55),
                       length * rnd.uniform(0.55, 0.85), width * rnd.uniform(0.55, 0.8))

    trunks = 20 if False else rnd.randint(9, 12)  # number of major branches
    for _ in range(trunks):
        ang = rnd.uniform(0.0, math.tau)
        branch(OD_X, OD_Y, ang, rnd.uniform(RADIUS * 0.55, RADIUS * 0.95), rnd.uniform(4.5, 6.5))


def _lesions(draw: ImageDraw.ImageDraw, img, rnd: random.Random, grade: int) -> dict:
    """Add the lesion signs that define each ICDR grade; return their seed markers."""
    w, h = img.size
    seeds = {"ma": [], "hem": [], "ex": [], "soft": [], "nv": [], "vitreous": []}

    def near_od(x, y):  # keep lesions clear of the optic disc
        return math.hypot(x - OD_X, y - OD_Y) < OD_R * 1.7

    def add_ma(n):
        for _ in range(n):
            x, y = _rnd_point(rnd)
            if near_od(x, y):
                continue
            r = rnd.uniform(1.4, 2.6)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(118, 16, 16))
            seeds["ma"].append((int(x), int(y), r))

    def add_hem(n, big=0):
        for _ in range(n):
            x, y = _rnd_point(rnd)
            if near_od(x, y):
                continue
            r = rnd.uniform(4.5, 7.5) if rnd.random() > 0.25 else rnd.uniform(8, 13)
            r = r * (1.5 if big else 1.0)
            draw.ellipse([x - r, y - r, x + r * 1.3, y + r], fill=(96, 8, 8))
            seeds["hem"].append((int(x), int(y), r))

    def add_ex(n, soft=0):
        for _ in range(n):
            x, y = _rnd_point(rnd)
            if near_od(x, y):
                continue
            r = rnd.uniform(3.5, 7.0)
            if rnd.random() < soft:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(226, 216, 172))
                seeds["soft"].append((int(x), int(y), r))
            else:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(238, 222, 138))
                seeds["ex"].append((int(x), int(y), r))

    def add_nv():
        # thin, tortuous bright-red new vessel tufts in the periphery
        for _ in range(rnd.randint(3, 6)):
            x, y = _rnd_point(rnd)
            seeds["nv"].append((int(x), int(y), 0.0))
            ang = rnd.uniform(0.0, math.tau)
            px, py = x, y
            for step in range(rnd.randint(5, 9)):
                ang += rnd.uniform(-0.9, 0.9)
                nx = px + math.cos(ang) * rnd.uniform(5, 9)
                ny = py + math.sin(ang) * rnd.uniform(5, 9)
                draw.line([px, py, nx, ny], fill=(208, 74, 74), width=1)
                px, py = nx, ny

    def add_vitreous():
        for _ in range(rnd.randint(1, 2)):
            x, y = _rnd_point(rnd)
            r = rnd.uniform(16, 34)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(70, 10, 10))
            seeds["vitreous"].append((int(x), int(y), r))

    if grade >= 1:
        add_ma(rnd.randint(4, 8))
    if grade >= 2:
        add_ma(rnd.randint(4, 7))
        add_hem(rnd.randint(3, 6))
        add_ex(rnd.randint(2, 4))
    if grade >= 3:
        add_ma(rnd.randint(8, 12))
        add_hem(rnd.randint(4, 6), big=1)
        add_ex(rnd.randint(2, 4), soft=1)
    if grade >= 4:
        add_nv()
        add_vitreous()
        add_hem(rnd.randint(2, 4))
    return seeds


def make_fundus(grade: int, quality: int = 0, seed: int | None = None, size=SIZE) -> Image.Image:
    """
    grade:  0-4 ICDR severity
    quality: 0 = good, 1 = borderline (enhance), 2 = poor (reject)
    """
    rnd = random.Random(seed if seed is not None else grade * 1000 + quality * 100 + grade)
    s = size

    # ---- retinal background (red-orange radial gradient) ---------------------
    yy, xx = np.mgrid[0:s, 0:s]
    d = np.sqrt((xx - CX * size / 512) ** 2 + (yy - CY * size / 512) ** 2) / (RADIUS * size / 512)
    d = np.clip(d, 0, 1.4)

    r_chan = np.clip(214 - 78 * d + rnd.uniform(-6, 6), 0, 255)
    g_chan = np.clip(126 - 46 * d - 8 * (d ** 2) + rnd.uniform(-5, 5), 0, 255)
    b_chan = np.clip(104 - 40 * d - 4 * (d ** 2) + rnd.uniform(-5, 5), 0, 255)

    # vignette (darker edge for depth) and black background outside the disc
    vig = 1.0 - 0.30 * np.clip(d, 0, 1) ** 2
    inside = d < 1.0
    r_chan = r_chan * vig * inside
    g_chan = g_chan * vig * inside
    b_chan = b_chan * vig * inside

    # subtle sensor noise for realism + crispness
    nrng = np.random.default_rng(seed)
    for ch in (r_chan, g_chan, b_chan):
        ch += nrng.normal(0, 2.5, ch.shape)

    rgb = np.stack([r_chan, g_chan, b_chan], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)

    # ---- optic disc (pale yellow) -------------------------------------------
    od = Image.new("L", (s, s), 0)
    odd = ImageDraw.Draw(od)
    odd.ellipse([OD_X - OD_R, OD_Y - OD_R, OD_X + OD_R, OD_Y + OD_R], fill=255)
    od_blur = od.filter(ImageFilter.GaussianBlur(3))
    odmask = np.asarray(od_blur, dtype=np.float32) / 255.0
    arr = np.asarray(img, dtype=np.float32)
    od_tint = np.array([224, 196, 128], dtype=np.float32)
    arr = arr + (od_tint[None, None, :] - arr) * odmask[..., None] * 0.75
    center = np.exp(-((xx - OD_X) ** 2 + (yy - OD_Y) ** 2) / (2 * 12 ** 2))[..., None]
    arr = arr + (np.array([238, 216, 168], dtype=np.float32)[None, None, :] - arr) * center * 0.5
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)

    # ---- fovea / macula (dark centre) ----------------------------------------
    fo = Image.new("L", (s, s), 0)
    fod = ImageDraw.Draw(fo)
    fod.ellipse([CX - size * 0.085, CY - size * 0.085, CX + size * 0.085, CY + size * 0.085], fill=255)
    fo_blur = fo.filter(ImageFilter.GaussianBlur(6))
    fomask = np.asarray(fo_blur, dtype=np.float32) / 255.0
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * (1.0 - fomask[..., None] * 0.22)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)

    # ---- vessels + lesions ---------------------------------------------------
    _vessels(draw, rnd)
    seeds = _lesions(draw, img, rnd, grade)
    img = img.filter(ImageFilter.GaussianBlur(0.8))

    # ---- quality variants ----------------------------------------------------
    if quality == 1:            # borderline: soft focus, darker, vignetting
        img = img.filter(ImageFilter.GaussianBlur(2.2))
        img = img.point(lambda v: int(v * 0.82))
        arr = np.asarray(img, dtype=np.float32)
        arr = arr * (1.0 - 0.32 * np.clip(d, 0, 1) ** 2.5)[..., None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    elif quality == 2:          # poor: heavy blur, very dark, glare + eyelash
        img = img.filter(ImageFilter.GaussianBlur(4.2))
        img = img.point(lambda v: int(v * 0.55))
        arr = np.asarray(img, dtype=np.float32)
        arr = arr * (1.0 - 0.55 * np.clip(d, 0, 1) ** 2.5)[..., None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(img)
        # diagonal glare streak
        gld = Image.new("L", (s, s), 0)
        g2 = ImageDraw.Draw(gld)
        for i in range(0, s, 5):
            g2.line([(i, 0), (0, i)], fill=int(255 * (1 - i / s)), width=3)
        glare = gld.filter(ImageFilter.GaussianBlur(9))
        garr = np.asarray(img, dtype=np.float32)
        gm = np.asarray(glare, dtype=np.float32) / 255.0
        garr = garr + 240 * gm[..., None]
        img = Image.fromarray(np.clip(garr, 0, 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(img)
        # eyelash arcs near edge
        for i in range(rnd.randint(5, 9)):
            x, y = _rnd_point(rnd)
            if 0.45 < math.hypot(x - CX, y - CY) / RADIUS < 1.0:
                draw.arc([x - 40, y - 12, x + 40, y + 12], 0, 180, fill=(18, 16, 16), width=3)

    img = img.resize((size, size))
    img.info["lesion_seeds"] = seeds
    return img


QUALITY_NAMES = {0: "good", 1: "borderline", 2: "poor"}
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


def preview(out_dir: str) -> list[str]:
    """Save a small contact sheet of all (grade x quality) combos for sanity checks."""
    os.makedirs(out_dir, exist_ok=True)
    from PIL import ImageDraw as _D
    paths = []
    for g in range(5):
        for q in range(3):
            img = make_fundus(g, q, seed=42)
            d = _D.Draw(img)
            d.text((10, 10), f"G{g} Q{QUALITY_NAMES[q]}", fill=(255, 255, 255))
            p = os.path.join(out_dir, f"g{g}_q{q}.png")
            img.save(p)
            paths.append(p)
    return paths


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "sample_preview")
    saved = preview(out)
    print("Saved", len(saved), "previews to", out)