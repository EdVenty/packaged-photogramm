"""FastAPI application: public API for the TUI and internal API for workers."""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from . import __version__
from .db import ACTIVE, DB, job_progress, job_status
from .params import JobCreate, build_stages, presets_public, resolve_params
from .storage import (Storage, extract_images_from_zip, is_image, list_tree, resolve_inside,
                      safe_name, save_stream, stream_zip)

WORKER_ONLINE_SEC = 60.0


class DatasetCreate(BaseModel):
    name: str



class WorkerHeartbeat(BaseModel):
    worker_id: str
    kinds: list[str]
    hostname: str = ""
    info: dict[str, Any] = {}



class ClaimReq(BaseModel):
    worker_id: str
    kinds: list[str]



class StageHeartbeat(BaseModel):
    worker_id: str
    progress: float | None = None
    message: str | None = None



class StageFinish(BaseModel):
    worker_id: str
    status: str
    error: str = ""
    message: str | None = None



class Settings:
    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("PG_DATA_DIR", "/data"))
        self.api_token = os.environ.get("PG_API_TOKEN", "")
        self.lease_sec = float(os.environ.get("PG_STAGE_LEASE_SEC", "180"))
        self.watchdog_interval = float(os.environ.get("PG_WATCHDOG_INTERVAL", "15"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    db = DB(settings.data_dir / "db.sqlite")
    storage = Storage(settings.data_dir)
    stop = threading.Event()

    def watchdog() -> None:
        while not stop.wait(settings.watchdog_interval):
            try:
                db.expire_stale(settings.lease_sec)
            except Exception:  # pragma: no cover - keep the thread alive
                pass

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        t = threading.Thread(target=watchdog, name="watchdog", daemon=True)
        t.start()
        yield
        stop.set()

    app = FastAPI(title="packaged-photogramm", version=__version__, lifespan=lifespan)
    app.state.db = db
    app.state.storage = storage
    app.state.settings = settings

    def auth(authorization: str | None = Header(None),
             x_api_token: str | None = Header(None)) -> None:
        if not settings.api_token:
            return
        tok = x_api_token or ""
        if authorization and authorization.lower().startswith("bearer "):
            tok = authorization[7:].strip()
        if not secrets.compare_digest(tok, settings.api_token):
            raise HTTPException(401, "invalid or missing API token")

    # ------------------------------------------------------------------ helpers
    def get_dataset(did: str) -> dict[str, Any]:
        r = db.one("SELECT * FROM datasets WHERE id=?", (did,))
        if r is None:
            raise HTTPException(404, "dataset not found")
        d = dict(r)
        imgs = storage.list_images(did)
        d["image_count"] = len(imgs)
        d["size"] = sum(i["size"] for i in imgs)
        return d

    def job_view(r: Any, with_params: bool = True) -> dict[str, Any]:
        j = dict(r)
        stages = db.job_stages(j["id"])
        j["params"] = json.loads(j["params"])
        if not with_params:
            uw = j["params"].get("underwater", {})
            j["mode"] = ("uw-refractive" if uw.get("enabled") and uw.get("camera") == "refractive"
                         else "underwater" if uw.get("enabled") else "standard")
            j.pop("params")
        j["status"] = job_status(stages)
        j["progress"] = job_progress(stages)
        j["stages"] = stages
        ds = db.one("SELECT name FROM datasets WHERE id=?", (j["dataset_id"],))
        j["dataset_name"] = ds["name"] if ds else "?"
        return j

    def get_job_row(jid: str) -> Any:
        r = db.one("SELECT * FROM jobs WHERE id=?", (jid,))
        if r is None:
            raise HTTPException(404, "job not found")
        return r

    def online_workers() -> list[dict[str, Any]]:
        now = time.time()
        out = []
        for r in db.query("SELECT * FROM workers ORDER BY id"):
            w = dict(r)
            w["kinds"] = json.loads(w["kinds"])
            w["info"] = json.loads(w["info"])
            w["online"] = now - w["last_seen"] < WORKER_ONLINE_SEC
            w["age"] = round(now - w["last_seen"], 1)
            out.append(w)
        return out

    # ------------------------------------------------------------------ public
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/info", dependencies=[Depends(auth)])
    def info() -> dict[str, Any]:
        ws = online_workers()
        kinds = sorted({k for w in ws if w["online"] for k in w["kinds"]})
        return {"version": __version__, "presets": list(presets_public()),
                "workers_online": sum(1 for w in ws if w["online"]), "kinds_online": kinds}

    @app.get("/api/presets", dependencies=[Depends(auth)])
    def presets() -> dict[str, Any]:
        return {"presets": presets_public(), "defaults": resolve_params(None, {}).model_dump()}

    @app.get("/api/datasets", dependencies=[Depends(auth)])
    def list_datasets() -> list[dict[str, Any]]:
        return [get_dataset(r["id"]) for r in
                db.query("SELECT id FROM datasets ORDER BY created_at DESC")]

    @app.post("/api/datasets", dependencies=[Depends(auth)], status_code=201)
    def create_dataset(body: DatasetCreate) -> dict[str, Any]:
        name = body.name.strip() or "dataset"
        did = db.create_dataset(name)
        storage.dataset_images(did).mkdir(parents=True, exist_ok=True)
        return get_dataset(did)

    @app.get("/api/datasets/{did}", dependencies=[Depends(auth)])
    def dataset_detail(did: str) -> dict[str, Any]:
        d = get_dataset(did)
        d["images"] = storage.list_images(did)
        return d

    @app.delete("/api/datasets/{did}", dependencies=[Depends(auth)])
    def delete_dataset(did: str) -> dict[str, Any]:
        get_dataset(did)
        jobs = db.query("SELECT id FROM jobs WHERE dataset_id=?", (did,))
        for j in jobs:
            if any(s["status"] in ACTIVE for s in db.job_stages(j["id"])):
                raise HTTPException(409, "dataset has active jobs; cancel them first")
        db.delete_dataset(did)
        for j in jobs:
            storage.remove_tree(storage.job_dir(j["id"]))
        storage.remove_tree(storage.dataset_dir(did))
        return {"deleted": did}

    @app.post("/api/datasets/{did}/images", dependencies=[Depends(auth)])
    def upload_images(did: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
        get_dataset(did)
        saved, skipped = [], []
        for f in files:
            name = safe_name(f.filename or "")
            if not is_image(name):
                skipped.append(f.filename)
                continue
            save_stream(f.file, storage.dataset_images(did) / name)
            saved.append(name)
        return {"saved": saved, "skipped": skipped, "dataset": get_dataset(did)}

    @app.post("/api/datasets/{did}/archive", dependencies=[Depends(auth)])
    def upload_archive(did: str, file: UploadFile = File(...)) -> dict[str, Any]:
        get_dataset(did)
        tmp = storage.dataset_dir(did) / f".upload-{os.getpid()}-{time.time_ns()}.zip"
        try:
            save_stream(file.file, tmp)
            try:
                saved = extract_images_from_zip(tmp, storage.dataset_images(did))
            except Exception as e:
                raise HTTPException(400, f"cannot read zip: {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
        return {"saved": saved, "skipped": [], "dataset": get_dataset(did)}

    @app.get("/api/jobs", dependencies=[Depends(auth)])
    def list_jobs() -> list[dict[str, Any]]:
        return [job_view(r, with_params=False)
                for r in db.query("SELECT * FROM jobs ORDER BY created_at DESC")]

    @app.post("/api/jobs", dependencies=[Depends(auth)], status_code=201)
    def create_job(body: JobCreate) -> dict[str, Any]:
        ds = get_dataset(body.dataset_id)
        if ds["image_count"] < 2:
            raise HTTPException(400, "dataset needs at least 2 images")
        try:
            params = resolve_params(body.preset, body.params)
        except (ValidationError, ValueError) as e:
            raise HTTPException(422, str(e)) from e
        stages = build_stages(body.pipeline, params)
        name = body.name.strip() or f"{ds['name']}-{body.pipeline}"
        jid = db.create_job(name, body.dataset_id, body.pipeline, body.preset,
                            params.model_dump(), stages)
        for sub in ("logs", "outputs", "work"):
            (storage.job_dir(jid) / sub).mkdir(parents=True, exist_ok=True)
        return job_view(get_job_row(jid))

    @app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
    def job_detail(jid: str) -> dict[str, Any]:
        return job_view(get_job_row(jid))

    @app.post("/api/jobs/{jid}/cancel", dependencies=[Depends(auth)])
    def cancel_job(jid: str) -> dict[str, Any]:
        get_job_row(jid)
        db.cancel_job(jid)
        return job_view(get_job_row(jid))

    @app.post("/api/jobs/{jid}/retry", dependencies=[Depends(auth)])
    def retry_job(jid: str) -> dict[str, Any]:
        get_job_row(jid)
        if any(s["status"] == "running" for s in db.job_stages(jid)):
            raise HTTPException(409, "job is still running")
        db.retry_job(jid)
        return job_view(get_job_row(jid))

    @app.delete("/api/jobs/{jid}", dependencies=[Depends(auth)])
    def delete_job(jid: str) -> dict[str, Any]:
        get_job_row(jid)
        if any(s["status"] == "running" for s in db.job_stages(jid)):
            raise HTTPException(409, "job is running; cancel it first")
        db.delete_job(jid)
        storage.remove_tree(storage.job_dir(jid))
        return {"deleted": jid}

    @app.get("/api/jobs/{jid}/logs/{stage}", dependencies=[Depends(auth)])
    def job_logs(jid: str, stage: str, offset: int = Query(0),
                 limit: int = Query(256 * 1024, ge=1, le=4 << 20)) -> dict[str, Any]:
        get_job_row(jid)
        p = storage.log_path(jid, stage)
        if not p.exists():
            return {"offset": 0, "next_offset": 0, "size": 0, "text": ""}
        size = p.stat().st_size
        if offset < 0:
            offset = max(0, size - 64 * 1024)
        offset = min(offset, size)
        with open(p, "rb") as f:
            f.seek(offset)
            data = f.read(limit)
        return {"offset": offset, "next_offset": offset + len(data), "size": size,
                "text": data.decode("utf-8", errors="replace")}

    @app.get("/api/jobs/{jid}/artifacts", dependencies=[Depends(auth)])
    def artifacts(jid: str) -> list[dict[str, Any]]:
        get_job_row(jid)
        return list_tree(storage.outputs_dir(jid))

    @app.get("/api/jobs/{jid}/artifacts/{path:path}", dependencies=[Depends(auth)])
    def artifact(jid: str, path: str) -> FileResponse:
        get_job_row(jid)
        p = resolve_inside(storage.outputs_dir(jid), path)
        if p is None or not p.is_file():
            raise HTTPException(404, "artifact not found")
        return FileResponse(p, filename=p.name)

    @app.get("/api/jobs/{jid}/archive", dependencies=[Depends(auth)], response_model=None)
    def archive(jid: str) -> StreamingResponse:
        r = get_job_row(jid)
        root = storage.outputs_dir(jid)
        if not list_tree(root):
            raise HTTPException(404, "no artifacts yet")
        fname = f"{safe_name(r['name'])}-{jid}.zip"
        return StreamingResponse(stream_zip(root, f"{safe_name(r['name'])}-{jid}"),
                                 media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    @app.get("/api/workers", dependencies=[Depends(auth)])
    def workers() -> list[dict[str, Any]]:
        return online_workers()

    # ------------------------------------------------------------------ internal
    @app.post("/internal/workers/heartbeat", dependencies=[Depends(auth)])
    def worker_hb(body: WorkerHeartbeat) -> dict[str, Any]:
        db.worker_heartbeat(body.worker_id, body.kinds, body.hostname, body.info)
        return {"ok": True}

    @app.post("/internal/claim", dependencies=[Depends(auth)], response_model=None)
    def claim(body: ClaimReq) -> Response | dict[str, Any]:
        if not db.one("SELECT 1 FROM workers WHERE id=?", (body.worker_id,)):
            db.worker_heartbeat(body.worker_id, body.kinds, "", {})
        s = db.claim(body.worker_id, body.kinds)
        if s is None:
            return Response(status_code=204)
        j = get_job_row(s["job_id"])
        return {
            "stage": {"id": s["id"], "name": s["name"], "kind": s["kind"]},
            "job": {"id": j["id"], "name": j["name"], "pipeline": j["pipeline"],
                    "params": json.loads(j["params"])},
            "paths": {
                "data": str(settings.data_dir),
                "dataset_images": str(storage.dataset_images(j["dataset_id"])),
                "job": str(storage.job_dir(j["id"])),
                "log": str(storage.log_path(j["id"], s["name"])),
                "outputs": str(storage.outputs_dir(j["id"])),
            },
        }

    @app.post("/internal/stages/{sid}/heartbeat", dependencies=[Depends(auth)])
    def stage_hb(sid: str, body: StageHeartbeat) -> dict[str, Any]:
        r = db.stage_heartbeat(sid, body.worker_id, body.progress, body.message)
        if r is None:
            return {"cancel": True, "reason": "stage not found"}
        lost = r["status"] != "running" or r["worker_id"] != body.worker_id
        return {"cancel": bool(r["cancel_requested"]) or lost,
                "reason": "stage no longer assigned" if lost else
                ("cancel requested" if r["cancel_requested"] else "")}

    @app.post("/internal/stages/{sid}/finish", dependencies=[Depends(auth)])
    def stage_finish(sid: str, body: StageFinish) -> dict[str, Any]:
        if body.status not in ("done", "failed", "cancelled"):
            raise HTTPException(422, "bad status")
        ok = db.finish_stage(sid, body.worker_id, body.status, body.error[-8000:], body.message)
        return {"accepted": ok}

    return app

