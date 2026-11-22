"""Worker agent: registers with the API, claims stages and runs them."""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import socket
import threading
import time
import traceback
from typing import Any

from .client import ApiError, Client
from .context import Cancelled, StageContext, StageError

log = logging.getLogger("pgworker")

HANDLERS = {
    "preprocess": "pgworker.stages.preprocess",
    "sfm": "pgworker.stages.sfm",
    "mvs": "pgworker.stages.mvs",
    "splat": "pgworker.stages.splat",
}


class Agent:
    def __init__(self, client: Client, kinds: list[str], worker_id: str,
                 poll_interval: float = 3.0, heartbeat_interval: float = 10.0):
        self.client = client
        self.kinds = kinds
        self.worker_id = worker_id
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.stop = threading.Event()
        self.info = {"platform": platform.platform(), "python": platform.python_version(),
                     "gpu": os.environ.get("NVIDIA_VISIBLE_DEVICES", "")}

    def register(self) -> None:
        self.client.post("/internal/workers/heartbeat",
                         {"worker_id": self.worker_id, "kinds": self.kinds,
                          "hostname": socket.gethostname(), "info": self.info})

    def _bg_heartbeat(self) -> None:
        while not self.stop.wait(self.heartbeat_interval):
            try:
                self.register()
            except ApiError as e:
                log.warning("heartbeat failed: %s", e)

    def run_forever(self) -> None:
        threading.Thread(target=self._bg_heartbeat, daemon=True).start()
        backoff = self.poll_interval
        while not self.stop.is_set():
            try:
                self.register()
                ran = self.run_once()
                backoff = self.poll_interval
                if not ran:
                    self.stop.wait(self.poll_interval)
            except ApiError as e:
                log.warning("API unavailable: %s (retry in %.0fs)", e, backoff)
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 60)

    def run_once(self) -> bool:
        status, claim = self.client.post("/internal/claim",
                                         {"worker_id": self.worker_id, "kinds": self.kinds})
        if status == 204 or not claim:
            return False
        self.execute(claim)
        return True

    def execute(self, claim: dict[str, Any]) -> str:
        stage = claim["stage"]
        ctx = StageContext(stage, claim["job"], claim["paths"])
        log.info("running stage %s/%s (%s)", claim["job"]["id"], stage["name"], stage["kind"])
        ctx.log(f"=== worker {self.worker_id} ({socket.gethostname()}) "
                f"starts stage '{stage['name']}' of job {claim['job']['id']} "
                f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        done = threading.Event()
        hb = threading.Thread(target=self._stage_heartbeat, args=(stage["id"], ctx, done),
                              daemon=True)
        hb.start()
        status, error = "done", ""
        try:
            mod = importlib.import_module(HANDLERS[stage["name"]])
            mod.run(ctx)
            ctx.set_progress(1.0, ctx.summary or "done")
        except Cancelled:
            status, error = "cancelled", ""
            ctx.message = "cancelled"
            ctx.log("=== stage cancelled ===")
        except StageError as e:
            status, error = "failed", str(e)
            ctx.log(f"=== stage failed: {e} ===")
        except Exception as e:  # noqa: BLE001
            status, error = "failed", f"{type(e).__name__}: {e}"
            ctx.log(traceback.format_exc())
        finally:
            done.set()
            hb.join(timeout=5)
        ctx.log(f"=== finished with status '{status}' at {time.strftime('%H:%M:%S')} ===")
        ctx.close()
        for attempt in range(10):
            try:
                self.client.post(f"/internal/stages/{stage['id']}/finish",
                                 {"worker_id": self.worker_id, "status": status,
                                  "error": error,
                                  "message": ctx.summary if status == "done" and ctx.summary
                                  else ctx.message})
                break
            except ApiError as e:
                log.warning("finish report failed (%s), retrying", e)
                time.sleep(min(2 ** attempt, 30))
        return status

    def _stage_heartbeat(self, sid: str, ctx: StageContext, done: threading.Event) -> None:
        interval = min(self.heartbeat_interval, 5.0)
        while not done.wait(interval):
            try:
                _, resp = self.client.post(f"/internal/stages/{sid}/heartbeat",
                                           {"worker_id": self.worker_id,
                                            "progress": round(ctx.progress, 4),
                                            "message": ctx.message})
                if resp and resp.get("cancel"):
                    ctx.log(f"[server: {resp.get('reason') or 'cancel'}]")
                    ctx.cancel_event.set()
            except ApiError as e:
                log.warning("stage heartbeat failed: %s", e)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="packaged-photogramm worker")
    ap.add_argument("--api", default=os.environ.get("PG_API_URL", "http://api:8000"))
    ap.add_argument("--token", default=os.environ.get("PG_API_TOKEN", ""))
    ap.add_argument("--kinds", default=os.environ.get("PG_WORKER_KINDS", ""),
                    help="comma-separated stage kinds: cpu,sfm,sfm_refrac,mvs,splat")
    ap.add_argument("--id", default=os.environ.get("PG_WORKER_ID", ""))
    ap.add_argument("--poll", type=float, default=float(os.environ.get("PG_POLL_INTERVAL", "3")))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    # Images can advertise optional capabilities (e.g. the splat image built with
    # WaterSplatting adds "splat_uw") via a file baked in at build time.
    extra = os.environ.get("PG_EXTRA_KINDS_FILE", "/etc/pgworker/extra-kinds")
    if os.path.isfile(extra):
        with open(extra) as f:
            kinds += [k.strip() for k in f.read().replace("\n", ",").split(",")
                      if k.strip() and k.strip() not in kinds]
    if not kinds:
        raise SystemExit("no kinds configured (PG_WORKER_KINDS)")
    wid = args.id or f"{'+'.join(kinds)}@{socket.gethostname()}"
    log.info("worker %s kinds=%s api=%s", wid, kinds, args.api)
    Agent(Client(args.api, args.token), kinds, wid, poll_interval=args.poll).run_forever()


if __name__ == "__main__":
    main()
