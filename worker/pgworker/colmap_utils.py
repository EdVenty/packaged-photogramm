"""Helpers for COLMAP: version-tolerant option names and model file handling."""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Number of intrinsic parameters per COLMAP camera model.
CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8,
    "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12, "FOV": 5, "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12, "RAD_TAN_THIN_PRISM_FISHEYE": 16,
}


class OptionResolver:
    """Maps logical options to whatever names the installed COLMAP build accepts.

    COLMAP renamed several options across versions (e.g. ``SiftExtraction.use_gpu`` became
    ``FeatureExtraction.use_gpu`` in 3.12). We parse ``colmap <command> -h`` once per command
    and pick the first candidate name that is present.
    """

    def __init__(self, capture: Callable[[Sequence[str]], str], binary: str = "colmap",
                 log: Callable[[str], None] | None = None):
        self._capture = capture
        self.binary = binary
        self._log = log or (lambda s: None)
        self._help: dict[str, str] = {}

    def help(self, command: str) -> str:
        if command not in self._help:
            self._help[command] = self._capture([self.binary, command, "-h"])
        return self._help[command]

    def has_command(self, command: str) -> bool:
        txt = self.help(command)
        return bool(txt) and "--" in txt and "Unknown command" not in txt

    def pick(self, command: str, candidates: Iterable[str]) -> str | None:
        cands = list(candidates)
        txt = self.help(command)
        if not txt.strip():
            return cands[0]  # cannot introspect: assume the first (newest) spelling
        for c in cands:
            if re.search(r"--" + re.escape(c) + r"(?![\w.])", txt):
                return c
        return None

    def args(self, command: str, options: Iterable[tuple[Sequence[str], Any]],
             required: bool = False) -> list[str]:
        out: list[str] = []
        for cands, value in options:
            if value is None:
                continue
            name = self.pick(command, cands)
            if name is None:
                msg = f"[colmap] option {'/'.join(cands)} not supported by '{command}', skipped"
                if required:
                    raise ValueError(msg)
                self._log(msg)
                continue
            if isinstance(value, bool):
                value = 1 if value else 0
            out += [f"--{name}", str(value)]
        return out


def _read_u64(path: Path) -> int:
    with open(path, "rb") as f:
        b = f.read(8)
    return struct.unpack("<Q", b)[0] if len(b) == 8 else 0


def model_stats(model_dir: Path) -> dict[str, int]:
    """Registered images / points of a COLMAP model (bin or txt)."""
    st = {"images": 0, "points": 0}
    if (model_dir / "images.bin").exists():
        st["images"] = _read_u64(model_dir / "images.bin")
        if (model_dir / "points3D.bin").exists():
            st["points"] = _read_u64(model_dir / "points3D.bin")
    elif (model_dir / "images.txt").exists():
        lines = [ln for ln in (model_dir / "images.txt").read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        st["images"] = len(lines) // 2
        if (model_dir / "points3D.txt").exists():
            st["points"] = sum(1 for ln in (model_dir / "points3D.txt").read_text().splitlines()
                               if ln.strip() and not ln.startswith("#"))
    return st


def largest_model(sparse_dir: Path) -> tuple[Path | None, dict[str, int]]:
    best: Path | None = None
    best_st = {"images": 0, "points": 0}
    if not sparse_dir.is_dir():
        return None, best_st
    for d in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
        st = model_stats(d)
        if st["images"] > best_st["images"]:
            best, best_st = d, st
    return best, best_st


def strip_refraction_from_cameras_txt(path: Path) -> int:
    """Remove the optional ``REFRAC_MODEL REFRAC_PARAMS[]`` tail written by Refractive COLMAP.

    Returns the number of camera lines that were modified.
    """
    out_lines, changed = [], 0
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            out_lines.append(ln)
            continue
        parts = ln.split()
        model = parts[1] if len(parts) > 1 else ""
        n = CAMERA_MODEL_NUM_PARAMS.get(model)
        if n is None:
            out_lines.append(ln)
            continue
        keep = parts[: 4 + n]
        if len(parts) > len(keep):
            changed += 1
        out_lines.append(" ".join(keep))
    path.write_text("\n".join(out_lines) + "\n")
    return changed
