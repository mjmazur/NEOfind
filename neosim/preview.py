"""Quick-look images: per-frame PNGs, a blink GIF of all frames resampled onto
the reference frame's pixel grid, and a median stack with the true asteroid
tracks drawn on it."""

import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates


def reproject(data: np.ndarray, wcs, ref_wcs, ref_shape) -> np.ndarray:
    """Bilinear resampling of ``data`` onto ``ref_wcs``'s pixel grid (NaN off-frame)."""
    yy, xx = np.mgrid[0:ref_shape[0], 0:ref_shape[1]]
    ra, dec = ref_wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
    x, y = wcs.all_world2pix(ra, dec, 0)
    out = map_coordinates(data.astype(np.float64), [y, x], order=1, cval=np.nan,
                          prefilter=False)
    return out.reshape(ref_shape)


def display_limits(img: np.ndarray):
    """Robust linear stretch: sky median - 2 sigma to + 8 sigma."""
    v = img[np.isfinite(img)]
    med = np.median(v)
    sigma = 1.4826 * np.median(np.abs(v - med))
    return med - 2.0 * sigma, med + 8.0 * sigma


def to_uint8(img: np.ndarray, limits=None) -> np.ndarray:
    lo, hi = limits if limits is not None else display_limits(img)
    scaled = np.clip((img - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    out = np.round(255 * np.nan_to_num(scaled, nan=0.0)).astype(np.uint8)
    return out[::-1]  # FITS row 0 is the bottom of the image


def save_png(img: np.ndarray, path: Path, limits=None) -> None:
    Image.fromarray(to_uint8(img, limits), mode="L").save(path)


def write_previews(out_dir: Path, images, wcss, ref_index: int, tracks) -> None:
    """``tracks``: {asteroid id: [(ra, dec) per frame]} for the overlay."""
    ref_wcs, shape = wcss[ref_index], images[ref_index].shape
    limits = display_limits(images[ref_index])
    for k, img in enumerate(images):
        save_png(img, out_dir / f"frame_{k:04d}.png", limits)

    resampled = [reproject(img, w, ref_wcs, shape) for img, w in zip(images, wcss)]
    gif = [Image.fromarray(to_uint8(r, limits), mode="L") for r in resampled]
    gif[0].save(out_dir / "blink.gif", save_all=True, append_images=gif[1:],
                duration=400, loop=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels outside coverage
        median = np.nanmedian(np.stack(resampled), axis=0)
    rgb = Image.fromarray(to_uint8(median, limits), mode="L").convert("RGB")
    draw = ImageDraw.Draw(rgb)
    height = shape[0]
    for aid, radec in tracks.items():
        ra, dec = np.array(radec).T
        x, y = ref_wcs.all_world2pix(ra, dec, 0)
        pts = [(float(px), float(height - 1 - py)) for px, py in zip(x, y)]
        draw.line(pts, fill=(255, 60, 60), width=1)
        for px, py in pts:
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(255, 60, 60))
        draw.text((pts[0][0] + 6, pts[0][1] + 6), str(aid), fill=(255, 220, 0))
    rgb.save(out_dir / "overlay.png")
