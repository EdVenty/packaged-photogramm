"""Dense reconstruction + meshing + texturing with OpenMVS."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from ..context import StageContext, StageError

_RE_PCT = re.compile(r"\((\d+(?:\.\d+)?)%")


def _pct(line: str):
    m = _RE_PCT.search(line)
    return float(m.group(1)) / 100.0 if m else None


def place(src: Path, dst: Path) -> None:
    """Hard-link (or copy) an artifact into outputs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def run(ctx: StageContext) -> None:
    p = ctx.params["mvs"]
    dense = ctx.work / "dense"
    if not (dense / "sparse" / "cameras.txt").exists():
        raise StageError(f"no undistorted COLMAP workspace at {dense} (did the sfm stage run?)")
    bindir = os.environ.get("PG_OPENMVS_BIN", "")
    tool = (lambda name: str(Path(bindir) / name)) if bindir else (lambda name: name)
    wd = ctx.job_dir / "mvs"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    helps: dict[str, str] = {}

    def supports(name: str, flag: str) -> bool:
        if name not in helps:
            helps[name] = ctx.capture([tool(name), "--help"])
        h = helps[name]
        return (not h.strip()) or (flag in h)

    common = ["-w", wd, "--verbosity", "2"]

    ctx.step(0.0, 0.05, "OpenMVS: import COLMAP scene")
    ctx.run([tool("InterfaceCOLMAP"), "-i", dense, "-o", wd / "scene.mvs",
             "--image-folder", dense / "images", *common])

    ctx.step(0.05, 0.55, "OpenMVS: densify point cloud")
    cmd = [tool("DensifyPointCloud"), "-i", wd / "scene.mvs", "-o", wd / "scene_dense.mvs",
           "--resolution-level", p["resolution_level"], *common]
    if p.get("max_resolution") and supports("DensifyPointCloud", "--max-resolution"):
        cmd += ["--max-resolution", p["max_resolution"]]
    if supports("DensifyPointCloud", "--number-views-fuse"):
        cmd += ["--number-views-fuse", p["number_views_fuse"]]
    ctx.run(cmd, progress=_pct)
    dense_ply = wd / "scene_dense.ply"
    if dense_ply.exists():
        place(dense_ply, ctx.outputs / "dense_points.ply")

    ctx.step(0.55, 0.7, "OpenMVS: reconstruct mesh")
    cmd = [tool("ReconstructMesh"), "-i", wd / "scene_dense.mvs", "-o", wd / "scene_mesh.mvs",
           *common]
    if dense_ply.exists() and supports("ReconstructMesh", "--pointcloud-file"):
        cmd += ["--pointcloud-file", dense_ply]
    if p.get("decimate", 1.0) < 1.0:
        cmd += ["--decimate", p["decimate"]]
    ctx.run(cmd, progress=_pct)
    mesh_ply = wd / "scene_mesh.ply"
    if not mesh_ply.exists():
        raise StageError("ReconstructMesh produced no mesh")

    if p.get("refine"):
        ctx.step(0.7, 0.85, "OpenMVS: refine mesh")
        if supports("RefineMesh", "--mesh-file"):
            cmd = [tool("RefineMesh"), "-i", wd / "scene_dense.mvs", "--mesh-file", mesh_ply]
        else:
            cmd = [tool("RefineMesh"), "-i", wd / "scene_mesh.mvs"]
        cmd += ["-o", wd / "scene_refined.mvs", "--resolution-level",
                max(1, int(p["resolution_level"])), *common]
        ctx.run(cmd, progress=_pct)
        if (wd / "scene_refined.ply").exists():
            mesh_ply = wd / "scene_refined.ply"
    place(mesh_ply, ctx.outputs / "mesh.ply")

    if p.get("texture", True):
        ctx.step(0.85, 1.0, "OpenMVS: texture mesh")
        if supports("TextureMesh", "--mesh-file"):
            cmd = [tool("TextureMesh"), "-i", wd / "scene_dense.mvs", "--mesh-file", mesh_ply]
        else:
            cmd = [tool("TextureMesh"), "-i", wd / mesh_ply.with_suffix(".mvs").name]
        cmd += ["-o", wd / "scene_textured.mvs", "--resolution-level",
                p.get("texture_resolution_level", 0), *common]
        if supports("TextureMesh", "--export-type"):
            cmd += ["--export-type", "obj"]
        ctx.run(cmd, progress=_pct)
        tex_dir = ctx.outputs / "mesh_textured"
        shutil.rmtree(tex_dir, ignore_errors=True)
        files = [f for f in wd.glob("scene_textured*")
                 if f.suffix.lower() in (".obj", ".mtl", ".png", ".jpg", ".jpeg", ".ply")]
        if not files:
            raise StageError("TextureMesh produced no output")
        for f in files:
            place(f, tex_dir / f.name)
    ctx.summary = "mesh ready"
