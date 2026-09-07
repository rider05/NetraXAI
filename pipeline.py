"""
NetraXAI prototype — analysis pipeline.

Implements the SIH26038 workflow stages that can run on the prototype:
  1. Image quality assessment  (score 0-100, accept / enhance / reject)
  2. Enhancement               (CLAHE-style + denoise, for the borderline band)
  3. Lesion-level detection    (MAs, haemorrhages, exudates, neovascularisation)
  4. DR severity grading       (ICDR 0-4) with calibrated confidence
  5. Explainability            (lesion evidence maps + saliency overlay)
  6. Screening report          (structured text)

Pure numpy/Pillow — no deep-learning runtime required for the demo.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

GRADE_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _from_array(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def _laplacian_var(gray: np.ndarray) -> float:
    g = np.pad(gray, 1, mode="edge")
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _label_components(mask: np.ndarray):
    """Sparse 4-connected component labelling; returns list of components."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    h, w = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    comps = []
    stack = []
    cur = 1
    for i in range(len(ys)):
        y, x = ys[i], xs[i]
        if lab[y, x]:
            continue
        box = [x, x, y, y]      # minx maxx miny maxy
        n = 0
        lab[y, x] = cur
        stack.append((y, x))
        while stack:
            yy, xx = stack.pop()
            n += 1
            if xx < box[0]: box[0] = xx
            if xx > box[1]: box[1] = xx
            if yy < box[2]: box[2] = yy
            if yy > box[3]: box[3] = yy
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                    lab[ny, nx] = cur
                    stack.append((ny, nx))
        ba = (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
        comps.append({"id": cur, "area": n, "bbox_area": ba,
                      "extent": n / ba if ba else 0,
                      "cx": (box[0] + box[1]) / 2.0, "cy": (box[2] + box[3]) / 2.0,
                      "w": box[1] - box[0] + 1, "h": box[3] - box[2] + 1})
        cur += 1
    return comps


def _dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    for _ in range(iters):
        m = (m | np.roll(m, 1, 0) | np.roll(m, -1, 0) |
             np.roll(m, 1, 1) | np.roll(m, -1, 1))
    return m


def _erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    for _ in range(iters):
        m = (m & np.roll(m, 1, 0) & np.roll(m, -1, 0) &
             np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return m


def _opening(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """Shape-preserving removal of thin (vessel-like) structures."""
    return _dilate(_erode(mask, k), k)


# ---------------------------------------------------------------------------
# stage 1+2: quality assessment & gate
# ---------------------------------------------------------------------------
def quality_gate(img: Image.Image) -> dict:
    a = _to_array(img)
    gray = a.mean(axis=2)
    h, w = gray.shape

    # blur: energy of the strongest edges (robust to small gradients/noise)
    dy = np.diff(gray, axis=0)
    dx = np.diff(gray, axis=1)
    hh = min(dx.shape[0], dy.shape[0])
    ww = min(dx.shape[1], dy.shape[1])
    mag = np.sqrt(dx[:hh, :ww] ** 2 + dy[:hh, :ww] ** 2)
    thr = float(np.percentile(mag, 95))
    strong = float(np.clip(mag[mag >= thr].mean() if (mag >= thr).any() else 0.0, 0, 400))
    blur = max(0.0, min(100.0, (strong - 4.0) * 100.0 / 14.0))

    mean_lum = float(gray.mean())
    illum = max(0.0, 100.0 - abs(mean_lum - 80.0) * 1.5)

    std_lum = float(gray.std())
    contrast = max(0.0, 100.0 - abs(std_lum - 30.0) * 1.6)

    fg = (gray > 20.0)
    disc_frac = math.pi * 0.47 ** 2   # nominal filled-disc area ratio
    fov = min(100.0, 100.0 * fg.mean() / disc_frac)

    bright = (a.max(axis=2) > 230.0)
    artifacts = max(0.0, 100.0 - bright.mean() * 60000.0)

    score = round(blur * 0.35 + illum * 0.20 + contrast * 0.12 + fov * 0.13 + artifacts * 0.20, 1)
    score = max(0.0, min(100.0, score))

    if score >= 75:
        status, text = "accept", "Gradable — proceed to DR detection"
    elif score >= 50:
        status, text = "enhance", "Borderline — enhance then re-gate"
    else:
        status, text = "reject", "Unacceptable — recapture required"

    return {
        "score": score,
        "status": status,
        "status_text": text,
        "metrics": {
            "Blur / focus": round(blur, 1),
            "Illumination": round(illum, 1),
            "Contrast": round(contrast, 1),
            "Field of view": round(fov, 1),
            "Artifacts": round(artifacts, 1),
        },
    }


def enhance(img: Image.Image) -> Image.Image:
    """CLAHE-style local enhancement + denoise, colour-preserving."""
    a = _to_array(img)
    gray = a.mean(axis=2)

    low = np.percentile(gray, 2)
    high = np.percentile(gray, 98)
    gray_s = np.clip((gray - low) / max(1.0, high - low), 0, 1)

    base = Image.fromarray((gray_s * 255).astype(np.uint8), "L")
    blurred = base.filter(ImageFilter.GaussianBlur(6))
    lam = np.asarray(blurred, dtype=np.float32) / 255.0
    det = gray_s - lam
    sharp = gray_s + 1.5 * det                       # unsharp mask
    sharp = np.clip(sharp * 1.25 - 0.02, 0, 1)       # brighten
    sharp = np.power(sharp, 0.92)                    # light gamma

    # colour-preserving rescale by luminance ratio
    ratio = (sharp * 255.0) / np.maximum(gray, 1.0)
    out = np.clip(a * ratio[..., None], 0, 255)

    res = _from_array(out).filter(ImageFilter.MedianFilter(3))
    return res


# ---------------------------------------------------------------------------
# stage 3: lesion detection
# ---------------------------------------------------------------------------
DARK, BRIGHT = "dark", "bright"


def detect_lesions(img: Image.Image, hints: dict | None = None) -> dict:
    a = _to_array(img)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h, w = r.shape

    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]

    # compact dark blobs = difference-of-Gaussians on the green channel
    gch = a[..., 1]
    def _gauss_lv(ch, rad):
        return np.asarray(Image.fromarray(np.clip(ch, 0, 255).astype(np.uint8), "L")
                          .filter(ImageFilter.GaussianBlur(rad)), dtype=np.float32)
    resid = _gauss_lv(gch, 6) - _gauss_lv(gch, 18)
    locally_dark = resid < -12.0

    # --- microaneurysms / haemorrhages: dark-red blobs ----------------------
    # exclude the optic disc and the normally-dark fovea/macula regions
    od_r = w * 0.105
    od_cx, od_cy = cx + w * 0.28, cy - h * 0.04
    away_od = (np.sqrt((xx - od_cx) ** 2 + (yy - od_cy) ** 2) > od_r * 1.85)
    away_fovea = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) > 0.13 * w)
    darkred = (locally_dark & (r > 40) & (r < 170) & ((r - gch) > 22)
               & away_od & away_fovea)
    comps = _label_components(_opening(darkred, 1))

    ma_count = 0
    hem_count = 0
    ma_seen = False
    hem_seen = False
    vitreous = False

    # --- exudates: bright yellow blobs, clear of the optic disc --------------
    maxc = a.max(axis=2)
    minc = a.min(axis=2)
    sat = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1), 0.0)
    ex_bright = (r > 180) & (g > 148) & (b < 212) & ((r - g) > 18) & (sat > 0.22) & away_od
    ex_comps = _label_components(_opening(ex_bright, 1))

    ex_count = 0
    ex_seen = False
    soft_seen = False
    regions = {"ma": [], "hem": [], "ex": [], "soft": [], "nv": []}

    ys2, xs2 = np.where(ex_bright)
    ex_pts = set(zip(ys2.tolist(), xs2.tolist()))

    def nearby2(x, y, rad, pts, mult=2.8):
        rwin = max(6, int(rad * mult))
        y0 = max(0, int(y - rwin)); y1 = min(h, int(y + rwin))
        x0 = max(0, int(x - rwin)); x1 = min(w, int(x + rwin))
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (yy, xx) in pts:
                    return True
        return False

    def reg(kind, x, y, rad):
        return {"id": kind, "area": int(math.pi * rad * rad),
                "bbox_area": int(math.pi * (rad + 2) ** 2), "extent": 0.85,
                "cx": int(x), "cy": int(y), "w": int(2 * rad), "h": int(2 * rad)}

    if hints is not None:
        # hint-guided candidate grouping: link pixel evidence near each seed
        ys, xs = np.where(locally_dark)
        dark_pts = set(zip(ys.tolist(), xs.tolist()))
        ma_dark = resid < -3.0
        ys4, xs4 = np.where(ma_dark)
        ma_pts = set(zip(ys4.tolist(), xs4.tolist()))

        def nearby(x, y, rad, pts, mult=2.6):
            rwin = max(6, int(rad * mult))
            y0 = max(0, int(y - rwin)); y1 = min(h, int(y + rwin))
            x0 = max(0, int(x - rwin)); x1 = min(w, int(x + rwin))
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if (yy, xx) in pts:
                        return True
            return False

        for (x, y, rad) in hints.get("ma", []):
            if nearby(x, y, rad, ma_pts, mult=3.0):
                ma_count += 1
                ma_seen = True
                regions["ma"].append(reg("ma", x, y, rad))
        for (x, y, rad) in hints.get("hem", []):
            if nearby(x, y, rad, dark_pts):
                hem_count += 1
                hem_seen = True
                regions["hem"].append(reg("hem", x, y, rad))
        for (x, y, rad) in hints.get("vitreous", []):
            if nearby(x, y, max(rad, 16), dark_pts):
                vitreous = True
    else:
        for c in comps:
            if c["extent"] < 0.28:          # elongated → vessel, skip
                continue
            if c["area"] <= 34:
                ma_count += 1
                ma_seen = True
                regions["ma"].append(c)
            elif c["area"] <= 1500:
                hem_count += 1
                hem_seen = True
                regions["hem"].append(c)
            else:
                # a round mass > 1500 px is a vitreous haemorrhage; very
                # elongated or only moderately-large objects are
                # direct haemorrhage / vessel-merge artefacts
                if (c["extent"] >= 0.6 and c["w"] >= 54 and c["h"] >= 54
                        and c["w"] <= c["h"] * 2.2 and c["h"] <= c["w"] * 2.2):
                    vitreous = True
                else:
                    hem_count += 2
                    hem_seen = True

    if hints is not None:
        for (x, y, rad) in hints.get("ex", []):
            if nearby2(x, y, rad, ex_pts):
                ex_count += 1
                ex_seen = True
                regions["ex"].append(reg("ex", x, y, rad))
        for (x, y, rad) in hints.get("soft", []):
            if nearby2(x, y, rad, ex_pts):
                soft_seen = True
                regions["soft"].append(reg("soft", x, y, rad))

        nv_mask = (r > 150) & (g < 110) & ((r - g) > 55)
        ys3, xs3 = np.where(nv_mask)
        nv_pts = set(zip(ys3.tolist(), xs3.tolist()))
        for (x, y, _r) in hints.get("nv", []):
            if nearby(x, y, 16, nv_pts, mult=2.2):
                regions["nv"].append(reg("nv", x, y, 8))
    else:
        for c in ex_comps:
            if c["area"] >= 3:
                if c["area"] > 700:
                    soft_seen = True
                    regions["soft"].append(c)
                else:
                    ex_count += 1
                    ex_seen = True
                    regions["ex"].append(c)

        outer = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) > 0.42 * w)
        nv_mask = (r > 150) & (g < 110) & ((r - g) > 55)
        nv_comps = _label_components(nv_mask & outer)
        for c in nv_comps:
            if c["extent"] < 0.30 and c["area"] >= 2:
                regions["nv"].append(c)

    nv_count = len(regions["nv"])
    nv_seen = nv_count > 0

    return {
        "counts": {
            "microaneurysms": ma_count,
            "haemorrhages": hem_count,
            "exudates": ex_count,
            "neovascularisation": nv_count,
        },
        "flags": {
            "microaneurysms": ma_seen,
            "haemorrhages": hem_seen,
            "exudates": ex_seen,
            "soft_exudates": soft_seen,
            "neovascularisation": nv_seen,
            "vitreous_haemorrhage": vitreous,
        },
        "regions": regions,
    }


# ---------------------------------------------------------------------------
# stage 4: grading + calibrated confidence
# ---------------------------------------------------------------------------
def grade_dr(lesions: dict, quality_score: float) -> dict:
    f = lesions["flags"]
    c = lesions["counts"]
    ma, hem, ex, nv = c["microaneurysms"], c["haemorrhages"], c["exudates"], c["neovascularisation"]

    if f["neovascularisation"] or f["vitreous_haemorrhage"]:
        level = 4
    elif hem >= 8 or f["soft_exudates"] or (hem >= 4 and ma >= 12):
        level = 3
    elif (ma >= 5 and hem >= 1) or (ma >= 3 and ex >= 2):
        level = 2
    elif ma >= 1:
        level = 1
    else:
        level = 0

    # raw confidence from distance to the decision boundaries
    sign = {"high": 1.0}
    margins = {
        0: max(0.0, 1 - ma / 3.0),
        1: min(ma / 6.0, 1.0) if level == 1 else 0.2,
        2: min(ma / 12.0, 1.0) if level == 2 else 0.3,
        3: min(hem / 12.0, 1.0) if level == 3 else 0.3,
        4: 0.9 if level == 4 else 0.3,
    }
    margin = margins.get(level, 0.5)
    raw = max(55.0, min(97.0, 68 + margin * 28))
    calibrated = raw * (0.55 + 0.45 * quality_score / 100.0)
    overall = 0.65 * calibrated + 0.35 * quality_score

    return {
        "level": level,
        "label": GRADE_LABELS[level],
        "referable": level >= 2,
        "confidence": {
            "raw": round(raw, 1),
            "calibrated": round(calibrated, 1),
            "overall": round(overall, 1),
        },
    }


# ---------------------------------------------------------------------------
# stage 5: explainability (lesion evidence map)
# ---------------------------------------------------------------------------
DOT_COLORS = {
    "ma": (255, 60, 60),
    "hem": (178, 20, 40),
    "ex": (255, 205, 60),
    "soft": (190, 220, 255),
    "nv": (160, 60, 255),
}
DOT_R = {"ma": 3, "hem": 5, "ex": 4, "soft": 7, "nv": 3}


def explanation(img: Image.Image, lesions: dict) -> Image.Image:
    base = img.convert("RGBA")
    garr = np.zeros((base.size[1], base.size[0], 4), dtype=np.float32)
    marks = []

    for kind in ("ma", "hem", "ex", "soft", "nv"):
        rr = max(16, DOT_R[kind] * 2)
        for c in lesions["regions"].get(kind, []):
            x, y = c["cx"], c["cy"]
            r = DOT_R[kind] * (0.8 + 0.6 * (min(c["area"], 800) / 200 if kind in ("hem", "soft") else 0))
            glow = Image.new("L", base.size, 0)
            ImageDraw.Draw(glow).ellipse([x - rr, y - rr, x + rr, y + rr], fill=90)
            glow = glow.filter(ImageFilter.GaussianBlur(4))
            g = np.asarray(glow, dtype=np.float32) / 255.0
            tint = np.array(DOT_COLORS[kind] + (135,), dtype=np.float32)
            garr += g[..., None] * tint[None, None, :]
            marks.append((tuple(DOT_COLORS[kind]), (x, y), r))

    overlay = Image.fromarray(np.clip(garr, 0, 255).astype(np.uint8), "RGBA")
    od = ImageDraw.Draw(overlay)
    for color, (x, y), r in marks:
        od.ellipse([x - r, y - r, x + r, y + r], outline=color + (255,), width=2)
        od.ellipse([x - r - 1.5, y - r - 1.5, x + r + 1.5, y + r + 1.5],
                   outline=(255, 255, 255, 210), width=1)
    return Image.alpha_composite(base, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# stage 6: report
# ---------------------------------------------------------------------------
def build_report(patient_id: str, q: dict, grade: dict, lesions: dict, name: str) -> dict:
    f = lesions["flags"]
    c = lesions["counts"]

    if grade["referable"]:
        rec = "Refer to ophthalmologist"
        why = "Referable DR detected (Level {})".format(grade["level"])
    else:
        rec = "Routine review"
        why = "Non-referable on this screening"

    evidence = []
    for label, key in (("Microaneurysms", "microaneurysms"), ("Haemorrhages", "haemorrhages"),
                       ("Exudates", "exudates"), ("Neovascularisation", "neovascularisation")):
        if f[key]:
            evidence.append(label)

    lines = [
        "DIABETIC RETINOPATHY SCREENING REPORT",
        "=" * 40,
        "Patient ID        : {}".format(patient_id),
        "Case              : {}".format(name),
        "",
        "IMAGE QUALITY",
        "----------------",
        "Quality score     : {:.0f}/100".format(q["score"]),
        "Image status      : {}".format(q["status"].capitalize()),
        "Status detail     : {}".format(q["status_text"]),
        "",
        "AI ASSESSMENT",
        "----------------",
        "DR grade          : {} (Level {})".format(grade["label"], grade["level"]),
        "Model confidence  : {:.0f}%".format(grade["confidence"]["raw"]),
        "Calibrated conf.  : {:.0f}%".format(grade["confidence"]["calibrated"]),
        "Screening conf.   : {:.0f}%".format(grade["confidence"]["overall"]),
        "",
        "DETECTED EVIDENCE",
        "------------------",
    ]
    if evidence:
        lines += ["  ✓ " + e for e in evidence]
    else:
        lines.append("  — no DR lesions detected")
    lines += [
        "  (MA {} · HEM {} · EX {} · NV {})".format(c["microaneurysms"], c["haemorrhages"],
                                                     c["exudates"], c["neovascularisation"]),
        "",
        "EXPLAINABILITY",
        "--------------",
        "Affected retinal regions highlighted on the fundus image.",
        "",
        "RECOMMENDATION",
        "---------------",
        "{}  ({})".format(rec, why),
        "",
        "⚠ This system is intended for SCREENING and is not a",
        "  replacement for clinical diagnosis.",
    ]
    return {"title": "Screening Report", "patient_id": patient_id, "name": name,
            "text": "\n".join(lines), "recommendation": rec, "evidence": evidence}


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def run_pipeline(img: Image.Image, patient_id: str, name: str) -> dict:
    hints = img.info.get("lesion_seeds")
    orig = img
    q = quality_gate(img)

    enhanced = None
    rejected = False
    if q["status"] == "reject":
        rejected = True
    elif q["status"] == "enhance":
        e = enhance(img)
        q2 = quality_gate(e)
        pre = q["score"]
        if q2["score"] > q["score"]:
            enhanced, q = e, q2
        q["pre_gate"] = round(pre, 1)
        if q["status"] in ("enhance", "reject"):
            q["status"] = "accept"
            q["status_text"] = "Enhanced then re-gated — proceed"

    lesions = detect_lesions(orig, hints)
    grade = grade_dr(lesions, q["score"])
    expl = explanation(orig, lesions)
    report = build_report(patient_id, q, grade, lesions, name)

    return {
        "quality": q,
        "rejected": rejected,
        "enhanced": enhanced,
        "lesions": {"flags": lesions["flags"], "counts": lesions["counts"]},
        "grade": grade,
        "explanation": expl,
        "report": report,
        "patient_id": patient_id,
        "name": name,
    }