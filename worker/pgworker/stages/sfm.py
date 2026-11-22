"""Structure-from-Motion stage (COLMAP or Refractive COLMAP).

Produces ``work/dense`` — an undistorted COLMAP workspace (``images/`` + TXT model in
``sparse/``) used by both the MVS and the splat stages.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from ..colmap_utils import (OptionResolver, largest_model, model_stats,
                            strip_refraction_from_cameras_txt)
from ..context import StageContext, StageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_RE_FRAC = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
_RE_BLOCK = re.compile(r"block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]")
_RE_REG = re.compile(r"Registering image #\d+ \((\d+)\)")


def _frac_parser(line: str):
    m = _RE_BLOCK.search(line)
    if m:
        i, n1, j, n2 = map(int, m.groups())
        return ((i - 1) * n2 + j) / max(1, n1 * n2)
    m = _RE_FRAC.search(line)
    if m:
        a, b = map(int, m.groups())
        return a / max(1, b)
    return None


def _count_images(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else 0


def _nonempty(d: Path) -> bool:
    return _count_images(d) > 0


def select_image_dirs(ctx: StageContext) -> tuple[Path, Path]:
    """Return (images for feature matching, images for undistortion/texturing/splats)."""
    uw = ctx.params.get("underwater", {})
    pre = ctx.work / "images"
    enh = ctx.work / "images_enhanced"
    base = pre if _nonempty(pre) else ctx.dataset_images
    sfm_imgs = enh if (uw.get("enabled") and _nonempty(enh)) else base
    tex_imgs = enh if (uw.get("enabled") and uw.get("enhance_for") == "all"
                       and _nonempty(enh)) else base
    return sfm_imgs, tex_imgs


def run(ctx: StageContext) -> None:
    sfm = ctx.params["sfm"]
    uw = ctx.params.get("underwater", {})
    refractive = bool(uw.get("enabled")) and uw.get("camera") == "refractive"
    colmap = os.environ.get("PG_COLMAP_BIN", "colmap")
    R = OptionResolver(ctx.capture, colmap, ctx.log)

    sfm_imgs, tex_imgs = select_image_dirs(ctx)
    n_images = _count_images(sfm_imgs)
    if n_images < 2:
        raise StageError(f"not enough images in {sfm_imgs}")
    ctx.log(f"images for SfM: {sfm_imgs} ({n_images}); for dense/texture: {tex_imgs}")
    ctx.log(f"mode: {'REFRACTIVE (' + uw.get('port', 'flat') + ' port)' if refractive else 'standard'}")

    db = ctx.work / "database.db"
    sparse = ctx.work / "sparse"
    dense = ctx.work / "dense"
    for p in (db, ctx.work / "database.db-shm", ctx.work / "database.db-wal"):
        if p.exists():
            p.unlink()
    for d in (sparse, dense):
        shutil.rmtree(d, ignore_errors=True)
    sparse.mkdir(parents=True)

    use_gpu = bool(sfm.get("use_gpu", True))
    robust = bool(sfm.get("robust_features"))

    # ---------------------------------------------------------------- features
    ctx.step(0.0, 0.2, "feature extraction")
    fe = "feature_extractor"
    opts = [
        (["ImageReader.camera_model"], sfm["camera_model"]),
        (["ImageReader.single_camera"], bool(sfm["single_camera"])),
        (["ImageReader.camera_params"], sfm.get("camera_params") or None),
        # DSP-SIFT / affine shape are CPU-only in COLMAP.
        (["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"], use_gpu and not robust),
        (["FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"],
         sfm.get("max_image_size") or None),
        (["SiftExtraction.max_num_features"], sfm["max_num_features"]),
    ]
    if robust:
        opts += [
            (["SiftExtraction.domain_size_pooling"], True),
            (["SiftExtraction.estimate_affine_shape"], True),
            (["SiftExtraction.peak_threshold"], 0.004),
        ]
    cmd = [colmap, fe, "--database_path", db, "--image_path", sfm_imgs] + R.args(fe, opts)
    if refractive:
        model = "DOMEPORT" if uw.get("port") == "dome" else "FLATPORT"
        try:
            cmd += R.args(fe, [(["ImageReader.camera_refrac_model"], model),
                               (["ImageReader.camera_refrac_params"],
                                uw["refrac_params"].replace(" ", ""))], required=True)
        except ValueError as e:
            raise StageError(f"{e}. Refractive mode requires the Refractive COLMAP build "
                             f"(worker-sfm-refrac).") from e
    ctx.run(cmd, progress=_frac_parser)

    # ---------------------------------------------------------------- matching
    ctx.step(0.2, 0.45, f"{sfm['matcher']} matching")
    mcmd = f"{sfm['matcher']}_matcher"
    mopts = [
        (["FeatureMatching.use_gpu", "SiftMatching.use_gpu"], use_gpu),
        (["FeatureMatching.guided_matching", "SiftMatching.guided_matching"],
         True if sfm.get("guided_matching") else None),
    ]
    if sfm["matcher"] == "sequential":
        mopts += [(["SequentialMatching.overlap"], sfm.get("sequential_overlap", 10)),
                  (["SequentialMatching.loop_detection"], False)]
    cmd = [colmap, mcmd, "--database_path", db] + R.args(mcmd, mopts)
    if refractive:
        cmd += R.args(mcmd, [(["TwoViewGeometry.enable_refraction"], True)], required=False)
    ctx.run(cmd, progress=_frac_parser)

    # ---------------------------------------------------------------- mapping
    ctx.step(0.45, 0.8, "sparse reconstruction (mapper)")
    mapper = "hierarchical_mapper" if sfm.get("mapper") == "hierarchical" else "mapper"
    cmd = [colmap, mapper, "--database_path", db, "--image_path", sfm_imgs,
           "--output_path", sparse]
    if refractive:
        cmd += R.args(mapper, [(["Mapper.enable_refraction"], True)], required=True)
        cmd += R.args(mapper, [(["Mapper.ba_refine_refrac_params"],
                                bool(uw.get("refine_refrac_params")))])

    def reg_progress(line: str):
        m = _RE_REG.search(line)
        return int(m.group(1)) / n_images if m else None

    ctx.run(cmd, progress=reg_progress)

    model, st = largest_model(sparse)
    if model is None or st["images"] < 2:
        raise StageError("COLMAP mapper produced no usable model "
                         "(too few matches — try 'robust_features', more overlap or another matcher)")
    ctx.log(f"selected model {model.name}: {st['images']}/{n_images} images, "
            f"{st['points']} points")
    ctx.summary = f"registered {st['images']}/{n_images} images, {st['points']} points"
    if st["images"] < 0.5 * n_images:
        ctx.log(f"WARNING: only {st['images']} of {n_images} images registered")

    # ---------------------------------------------------------------- exports
    sp_out = ctx.outputs / "sparse"
    shutil.rmtree(sp_out, ignore_errors=True)
    sp_out.mkdir(parents=True)
    ctx.run([colmap, "model_converter", "--input_path", model, "--output_path", sp_out,
             "--output_type", "TXT"])
    if (sp_out / "cameras.txt").exists():
        strip_refraction_from_cameras_txt(sp_out / "cameras.txt")
    ctx.run([colmap, "model_converter", "--input_path", model, "--output_path",
             ctx.outputs / "sparse_points.ply", "--output_type", "PLY"], check=False)

    # ---------------------------------------------------------------- undistortion
    ctx.step(0.8, 1.0, "image undistortion")
    cmd = [colmap, "image_undistorter", "--image_path", tex_imgs, "--input_path", model,
           "--output_path", dense, "--output_type", "COLMAP"]
    if sfm.get("undistort_max_image_size"):
        cmd += ["--max_image_size", sfm["undistort_max_image_size"]]
    ctx.run(cmd, progress=_frac_parser)

    dsp = dense / "sparse"
    if not dsp.is_dir():
        raise StageError("image_undistorter produced no sparse model")
    # Normalise to a TXT model: readable by OpenMVS and nerfstudio, and lets us drop
    # Refractive-COLMAP-specific fields (its binary format is incompatible upstream).
    tmp = dense / "sparse_txt"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    ctx.run([colmap, "model_converter", "--input_path", dsp, "--output_path", tmp,
             "--output_type", "TXT"])
    shutil.rmtree(dsp)
    tmp.rename(dsp)
    n = strip_refraction_from_cameras_txt(dsp / "cameras.txt")
    if n:
        ctx.log(f"stripped refraction parameters from {n} camera(s) in dense model")
    dst = model_stats(dsp)
    ctx.log(f"dense workspace ready: {dense} ({dst['images']} images)")
