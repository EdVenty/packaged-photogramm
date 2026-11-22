"""Filesystem layout helpers."""
from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class Storage:
    def __init__(self, root: Path):
        self.root = root
        (root / "datasets").mkdir(parents=True, exist_ok=True)
        (root / "jobs").mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, did: str) -> Path:
        return self.root / "datasets" / did

    def dataset_images(self, did: str) -> Path:
        return self.dataset_dir(did) / "images"

    def job_dir(self, jid: str) -> Path:
        return self.root / "jobs" / jid

    def outputs_dir(self, jid: str) -> Path:
        return self.job_dir(jid) / "outputs"

    def log_path(self, jid: str, stage: str) -> Path:
        return self.job_dir(jid) / "logs" / f"{safe_name(stage)}.log"

    def list_images(self, did: str) -> list[dict]:
        d = self.dataset_images(did)
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append({"name": p.name, "size": p.stat().st_size})
        return out

    def remove_tree(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def safe_name(name: str) -> str:
    base = PurePosixPath(name.replace("\\", "/")).name
    base = _SAFE.sub("_", base).strip("._") or "file"
    return base[:200]


def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def save_stream(src: BinaryIO, dst: Path, chunk: int = 4 << 20) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    n = 0
    with open(tmp, "wb") as f:
        while True:
            b = src.read(chunk)
            if not b:
                break
            f.write(b)
            n += len(b)
    os.replace(tmp, dst)
    return n


def extract_images_from_zip(zip_path: Path, dst_dir: Path) -> list[str]:
    """Extract image members (flattened, sanitized names) from a zip. Returns saved names."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and is_image(m.filename)
                   and not PurePosixPath(m.filename).name.startswith(".")
                   and "__MACOSX" not in m.filename]
        names = [PurePosixPath(m.filename.replace("\\", "/")).name for m in members]
        dup = {n for n in names if names.count(n) > 1}
        for m, n in zip(members, names):
            target_name = safe_name(m.filename.replace("/", "_") if n in dup else n)
            with zf.open(m) as src:
                save_stream(src, dst_dir / target_name)
            saved.append(target_name)
    return saved


def list_tree(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append({"path": p.relative_to(root).as_posix(), "size": p.stat().st_size})
    return out


def resolve_inside(root: Path, rel: str) -> Path | None:
    try:
        p = (root / rel).resolve()
        p.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return p


class _ZipStream:
    """File-like sink that collects written bytes for a streaming generator."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.pos = 0

    def write(self, b: bytes) -> int:
        self.buf += b
        self.pos += len(b)
        return len(b)

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        b = bytes(self.buf)
        self.buf.clear()
        return b


def stream_zip(root: Path, arc_prefix: str = "") -> Iterator[bytes]:
    for chunk in _stream_zip(root, arc_prefix):
        if chunk:
            yield chunk


def _stream_zip(root: Path, arc_prefix: str) -> Iterator[bytes]:
    sink = _ZipStream()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True) as zf:
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            arc = (PurePosixPath(arc_prefix) / p.relative_to(root).as_posix()).as_posix()
            with open(p, "rb") as src, zf.open(arc, "w", force_zip64=True) as dst:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    dst.write(b)
                    if len(sink.buf) >= (1 << 20):
                        yield sink.take()
            yield sink.take()
    yield sink.take()
