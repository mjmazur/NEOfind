"""Detection overlay: the static-sky template on the common grid with detected
tracks (and, for simulations, true tracks) drawn on it."""

import numpy as np
from PIL import Image, ImageDraw

from neosim.preview import display_limits, to_uint8


def plot_detections(result, path, truth=None, detected_only: bool = False) -> None:
    pre, reg = result.pre, result.reg
    img = pre.template.copy()
    img[pre.coverage == 0] = np.nan
    rgb = Image.fromarray(to_uint8(img, display_limits(img)), mode="L").convert("RGB")
    draw = ImageDraw.Draw(rgb)
    h = img.shape[0]

    def flip(x, y):
        return float(x), float(h - 1 - y)

    if truth:
        for a in truth["asteroids"]:
            x, y = reg.wcs.all_world2pix([p["ra_deg"] for p in a["frames"]],
                                         [p["dec_deg"] for p in a["frames"]], 0)
            pts = [flip(px, py) for px, py in zip(x, y)]
            draw.line(pts, fill=(80, 160, 255), width=1)
            for px, py in pts:
                draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(80, 160, 255))

    for i, d in enumerate(result.detections):
        if detected_only and not d["detected"]:
            continue
        color = (255, 60, 60) if d["detected"] else (255, 180, 0)
        xs, ys = reg.wcs.all_world2pix([f["ra_deg"] for f in d["frames"]],
                                       [f["dec_deg"] for f in d["frames"]], 0)
        pts = [flip(x, y) for x, y, f in zip(xs, ys, d["frames"]) if f["covered"]]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=1)
        for (px, py), f in zip([flip(x, y) for x, y in zip(xs, ys)], d["frames"]):
            if f["used"]:
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], outline=color)
        if pts:
            draw.text((pts[0][0] + 6, pts[0][1] - 12), f"{i}:{d['score']:.0f}", fill=color)
    rgb.save(path)
