"""Gaussian Splatting stage via nerfstudio (splatfacto / splatfacto-big / water-splatting)."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import sys

from ..context import StageContext, StageError
from .mvs import place

_RE_STEP = re.compile(r"^\s*(\d+)\s+\((\d+(?:\.\d+)?)%\)")


def _progress(line: str):
    m = _RE_STEP.search(line)
    return float(m.group(2)) / 100.0 if m else None


def run(ctx: StageContext) -> None:
    p = ctx.params["splat"]
    dense = ctx.work / "dense"
    if not (dense / "sparse" / "cameras.txt").exists():
        raise StageError(f"no undistorted COLMAP workspace at {dense} (did the sfm stage run?)")
    ns_train = os.environ.get("PG_NS_TRAIN", "ns-train")
    ns_export = os.environ.get("PG_NS_EXPORT", "ns-export")
    wd = ctx.job_dir / "splat"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    method = p["method"]

    ctx.step(0.0, 0.95, f"training {method} ({p['iterations']} iterations)")
    cmd = [ns_train, method,
           "--data", dense,
           "--output-dir", wd / "runs",
           "--experiment-name", "splat",
           "--timestamp", "run",
           "--max-num-iterations", p["iterations"],
           "--vis", "tensorboard",
           "--viewer.quit-on-train-completion", "True"]
    if p.get("extra_args"):
        cmd += shlex.split(p["extra_args"])
    cmd += ["colmap", "--colmap-path", "sparse", "--images-path", "images"]
    if int(p.get("downscale_factor", 1)) > 0:
        cmd += ["--downscale-factor", p["downscale_factor"]]
    ctx.run(cmd, cwd=wd, progress=_progress)

    configs = sorted((wd / "runs").rglob("config.yml"))
    if not configs:
        raise StageError("ns-train finished but no config.yml was found")
    cfg = configs[-1]
    ctx.log(f"trained config: {cfg}")

    ctx.step(0.95, 1.0, "exporting gaussian splat PLY")
    exp = wd / "export"
    exp.mkdir(exist_ok=True)
    rc = ctx.run([ns_export, "gaussian-splat", "--load-config", cfg, "--output-dir", exp],
                 cwd=wd, check=False)
    ply = exp / "splat.ply"
    if rc != 0 or not ply.exists():
        ctx.log("ns-export failed or unsupported for this model; using built-in exporter")
        ctx.run([sys.executable, "-m", "pgworker.export_splat", "--config", cfg,
                 "--output", ply], cwd=wd)
    if not ply.exists():
        raise StageError("splat export produced no file")
    place(ply, ctx.outputs / "splat.ply")
    ctx.summary = f"{method}: splat.ply ready"
