"""HTTP client for the packaged-photogramm API."""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ProgressCb = Callable[[int, int], None]  # (done_bytes, total_bytes)


class ApiError(Exception):
    def __init__(self, message: str, status: int = 0, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


class _ProgressReader:
    """File wrapper that reports read progress (used for multipart uploads)."""

    def __init__(self, fh, cb: Callable[[int], None]):
        self._fh = fh
        self._cb = cb

    def read(self, n: int = -1) -> bytes:
        b = self._fh.read(n)
        if b:
            self._cb(len(b))
        return b

    def __getattr__(self, item):
        return getattr(self._fh, item)


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and not p.name.startswith("."))


class ApiClient:
    def __init__(self, base_url: str, token: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._c = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)
        self._upload_timeout = httpx.Timeout(timeout, read=600.0, write=600.0)

    def close(self) -> None:
        self._c.close()

    # ------------------------------------------------------------- basics
    def _handle(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:  # noqa: BLE001
                detail = r.text
            msg = detail.get("message", detail) if isinstance(detail, dict) else detail
            raise ApiError(f"HTTP {r.status_code}: {msg}", r.status_code, detail)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def get(self, path: str, **params: Any) -> Any:
        try:
            return self._handle(self._c.get(path, params=params or None))
        except httpx.HTTPError as e:
            raise ApiError(f"{self.base_url}: {e}") from e

    def post(self, path: str, json: Any = None) -> Any:
        try:
            return self._handle(self._c.post(path, json=json))
        except httpx.HTTPError as e:
            raise ApiError(f"{self.base_url}: {e}") from e

    def delete(self, path: str) -> Any:
        try:
            return self._handle(self._c.delete(path))
        except httpx.HTTPError as e:
            raise ApiError(f"{self.base_url}: {e}") from e

    # ------------------------------------------------------------- domain
    def info(self) -> dict:
        return self.get("/api/info")

    def presets(self) -> dict:
        return self.get("/api/presets")

    def datasets(self) -> list[dict]:
        return self.get("/api/datasets")

    def create_dataset(self, name: str) -> dict:
        return self.post("/api/datasets", {"name": name})

    def delete_dataset(self, did: str) -> Any:
        return self.delete(f"/api/datasets/{did}")

    def jobs(self) -> list[dict]:
        return self.get("/api/jobs")

    def job(self, jid: str) -> dict:
        return self.get(f"/api/jobs/{jid}")

    def create_job(self, payload: dict) -> dict:
        return self.post("/api/jobs", payload)

    def cancel_job(self, jid: str) -> dict:
        return self.post(f"/api/jobs/{jid}/cancel")

    def retry_job(self, jid: str) -> dict:
        return self.post(f"/api/jobs/{jid}/retry")

    def delete_job(self, jid: str) -> Any:
        return self.delete(f"/api/jobs/{jid}")

    def logs(self, jid: str, stage: str, offset: int) -> dict:
        return self.get(f"/api/jobs/{jid}/logs/{stage}", offset=offset)

    def artifacts(self, jid: str) -> list[dict]:
        return self.get(f"/api/jobs/{jid}/artifacts")

    def workers(self) -> list[dict]:
        return self.get("/api/workers")

    # ------------------------------------------------------------- transfers
    def upload_images(self, did: str, files: Iterable[Path], progress: ProgressCb | None = None,
                      batch_files: int = 32, batch_bytes: int = 256 << 20) -> int:
        files = list(files)
        counts: dict[str, int] = {}
        for f in files:
            counts[f.name] = counts.get(f.name, 0) + 1
        total = sum(f.stat().st_size for f in files)
        done = 0
        sent = 0

        def bump(n: int) -> None:
            nonlocal done
            done += n
            if progress:
                progress(min(done, total), total)

        batch: list[Path] = []
        size = 0

        def flush() -> None:
            nonlocal batch, size, sent
            if not batch:
                return
            handles = [open(p, "rb") for p in batch]
            try:
                parts = [("files", (p.name if counts[p.name] == 1 else f"{p.parent.name}_{p.name}", _ProgressReader(h, bump),
                                    mimetypes.guess_type(p.name)[0] or "application/octet-stream"))
                         for p, h in zip(batch, handles)]
                r = self._c.post(f"/api/datasets/{did}/images", files=parts,
                                 timeout=self._upload_timeout)
                res = self._handle(r)
                sent += len(res.get("saved", []))
            except httpx.HTTPError as e:
                raise ApiError(f"upload failed: {e}") from e
            finally:
                for h in handles:
                    h.close()
            batch, size = [], 0

        for f in files:
            batch.append(f)
            size += f.stat().st_size
            if len(batch) >= batch_files or size >= batch_bytes:
                flush()
        flush()
        return sent

    def upload_archive(self, did: str, path: Path, progress: ProgressCb | None = None) -> int:
        total = path.stat().st_size
        done = 0

        def bump(n: int) -> None:
            nonlocal done
            done += n
            if progress:
                progress(min(done, total), total)

        with open(path, "rb") as fh:
            try:
                r = self._c.post(f"/api/datasets/{did}/archive",
                                 files={"file": (path.name, _ProgressReader(fh, bump),
                                                 "application/zip")},
                                 timeout=httpx.Timeout(30.0, read=3600.0, write=3600.0))
            except httpx.HTTPError as e:
                raise ApiError(f"upload failed: {e}") from e
        return len(self._handle(r).get("saved", []))

    def download(self, path: str, dest: Path, progress: ProgressCb | None = None) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            with self._c.stream("GET", path, timeout=httpx.Timeout(30.0, read=3600.0)) as r:
                if r.status_code >= 400:
                    r.read()
                    self._handle(r)
                total = int(r.headers.get("content-length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
        except httpx.HTTPError as e:
            raise ApiError(f"download failed: {e}") from e
        os.replace(tmp, dest)
        return dest
