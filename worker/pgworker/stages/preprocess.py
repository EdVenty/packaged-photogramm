"""Image preprocessing: optional downscale and underwater colour correction.

Outputs (same file names as the dataset, EXIF preserved so COLMAP keeps focal priors):
  work/images/           downscaled originals (only if preprocess.max_image_size > 0)
  work/images_enhanced/  colour-corrected images (only if underwater.enhance != none)
"""
from __future__ import annotations

import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..context import Cancelled, StageContext, StageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- algorithms
def gray_world(img):
    import numpy as np
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    return np.clip(img * (means.mean() / means), 0.0, 1.0)


def clahe(img, clip: float = 2.0):
    import cv2
    import numpy as np
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[..., 0] = eq.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def red_compensation(img, alpha: float = 1.0):
    """Ancuti et al. 2018: restore the strongly attenuated red channel from green."""
    import numpy as np
    r, g = img[..., 0], img[..., 1]
    out = img.copy()
    out[..., 0] = np.clip(r + alpha * (g.mean() - r.mean()) * (1.0 - r) * g, 0.0, 1.0)
    return out


def udcp(img, patch: int = 15, omega: float = 0.9, t_min: float = 0.15):
    """Underwater Dark Channel Prior dehazing (dark channel over G and B only)."""
    import cv2
    import numpy as np
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    gb_min = img[..., 1:].min(axis=2)
    dark = cv2.erode(gb_min, kernel)
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = img.reshape(-1, 3)[idx].mean(axis=0) + 1e-6
    norm_dark = cv2.erode((img[..., 1:] / A[1:]).min(axis=2), kernel)
    t = 1.0 - omega * norm_dark
    t = cv2.GaussianBlur(t.astype(np.float32), (0, 0), sigmaX=patch)
    t = np.maximum(t, t_min)[..., None]
    J = (img - A) / t + A
    return np.clip(J, 0.0, 1.0)


def enhance(img, method: str):
    if method == "grayworld":
        return gray_world(img)
    if method == "clahe":
        return clahe(img)
    if method == "redcomp":
        return clahe(gray_world(red_compensation(img)), clip=1.5)
    if method == "udcp":
        return clahe(gray_world(udcp(img)), clip=1.5)
    return img


# --------------------------------------------------------------------------- per-image job
def process_one(src: str, out_resized: str | None, out_enh: str | None, max_size: int,
                method: str) -> str:
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(src) as im:
        exif = im.info.get("exif")
        arr = np.asarray(im.convert("RGB"))
    h, w = arr.shape[:2]
    if max_size and max(h, w) > max_size:
        s = max_size / float(max(h, w))
        arr = cv2.resize(arr, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)

    def save(a, dst: str) -> None:
        img = Image.fromarray(a)
        ext = Path(dst).suffix.lower()
        kw = {}
        if ext in (".jpg", ".jpeg"):
            kw["quality"] = 95
        if exif and ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            kw["exif"] = exif
        img.save(dst, **kw)

    if out_resized:
        save(arr, out_resized)
    if out_enh:
        f = arr.astype(np.float32) / 255.0
        e = enhance(f, method)
        save((np.clip(e, 0, 1) * 255 + 0.5).astype(np.uint8), out_enh)
    return src


# --------------------------------------------------------------------------- stage
def run(ctx: StageContext) -> None:
    max_size = int(ctx.params.get("preprocess", {}).get("max_image_size", 0) or 0)
    uw = ctx.params.get("underwater", {})
    method = uw.get("enhance", "none") if uw.get("enabled") else "none"
    srcs = sorted(p for p in ctx.dataset_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not srcs:
        raise StageError("dataset has no images")
    out_r = ctx.work / "images"
    out_e = ctx.work / "images_enhanced"
    for d in (out_r, out_e):
        shutil.rmtree(d, ignore_errors=True)
    if max_size:
        out_r.mkdir(parents=True)
    if method != "none":
        out_e.mkdir(parents=True)
    ctx.step(0.0, 1.0, f"preprocess {len(srcs)} images (resize={max_size or 'off'}, "
                       f"enhance={method})")
    workers = int(os.environ.get("PG_PREPROCESS_WORKERS", "0")) or max(1, (os.cpu_count() or 2))
    done = 0
    # spawn: the agent is multi-threaded (heartbeats), fork() is unsafe there.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = [ex.submit(process_one, str(s),
                          str(out_r / s.name) if max_size else None,
                          str(out_e / s.name) if method != "none" else None,
                          max_size, method) for s in srcs]
        try:
            for fut in as_completed(futs):
                ctx.check_cancel()
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    raise StageError(f"preprocess failed: {e}") from e
                done += 1
                ctx.sub_progress(done / len(srcs))
                if done % 25 == 0 or done == len(srcs):
                    ctx.log(f"processed {done}/{len(srcs)}")
        except (Cancelled, StageError):
            for f in futs:
                f.cancel()
            raise
    ctx.summary = f"preprocessed {done} images"
