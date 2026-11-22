"""Execution context passed to stage handlers."""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


class Cancelled(Exception):
    pass


class StageError(Exception):
    pass


ProgressParser = Callable[[str], Optional[float]]


class StageContext:
    def __init__(self, stage: dict[str, Any], job: dict[str, Any], paths: dict[str, str]):
        self.stage = stage
        self.job = job
        self.params: dict[str, Any] = job["params"]
        self.data_dir = Path(paths["data"])
        self.dataset_images = Path(paths["dataset_images"])
        self.job_dir = Path(paths["job"])
        self.outputs = Path(paths["outputs"])
        self.work = self.job_dir / "work"
        self.log_path = Path(paths["log"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        self._lock = threading.Lock()
        self.progress = 0.0
        self.message = "started"
        self.summary = ""  # final human-readable result, reported on finish
        self.cancel_event = threading.Event()
        self._proc: subprocess.Popen | None = None
        # progress window of the currently running step: [lo, hi]
        self._win = (0.0, 1.0)

    # ---------------------------------------------------------------- logging
    def log(self, text: str) -> None:
        with self._lock:
            self._log.write(text if text.endswith("\n") else text + "\n")

    def close(self) -> None:
        with self._lock:
            self._log.close()

    def set_progress(self, p: float, message: str | None = None) -> None:
        self.progress = max(0.0, min(1.0, p))
        if message is not None:
            self.message = message[:300]

    def step(self, lo: float, hi: float, message: str) -> None:
        """Begin a step occupying the [lo, hi] slice of overall progress."""
        self._win = (lo, hi)
        self.set_progress(lo, message)
        self.log(f"\n===== [{time.strftime('%H:%M:%S')}] {message} =====")

    def sub_progress(self, frac: float) -> None:
        lo, hi = self._win
        self.set_progress(lo + (hi - lo) * max(0.0, min(1.0, frac)))

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    # ---------------------------------------------------------------- processes
    def run(self, cmd: Sequence[str], cwd: Path | None = None,
            progress: ProgressParser | None = None, env: dict[str, str] | None = None,
            check: bool = True) -> int:
        self.check_cancel()
        cmd = [str(c) for c in cmd]
        self.log(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
        full_env = dict(os.environ)
        full_env.setdefault("PYTHONUNBUFFERED", "1")
        if env:
            full_env.update(env)
        try:
            proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, env=full_env,
                                    start_new_session=True)
        except FileNotFoundError as e:
            raise StageError(f"executable not found: {cmd[0]}") from e
        self._proc = proc
        killer = threading.Thread(target=self._watch_cancel, args=(proc,), daemon=True)
        killer.start()
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # split on \n and \r (progress bars use \r)
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                self.log(line)
                if progress:
                    try:
                        f = progress(line)
                    except Exception:
                        f = None
                    if f is not None:
                        self.sub_progress(f)
        if buf:
            self.log(buf.decode("utf-8", errors="replace"))
        rc = proc.wait()
        self._proc = None
        if self.cancel_event.is_set():
            raise Cancelled()
        self.log(f"[exit code {rc}]")
        if check and rc != 0:
            raise StageError(f"command failed (exit {rc}): {cmd[0]} {' '.join(cmd[1:3])}")
        return rc

    def capture(self, cmd: Sequence[str], timeout: float = 120) -> str:
        """Run a short command and return combined output (not logged)."""
        try:
            r = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=timeout)
            return r.stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired):
            return ""

    def _watch_cancel(self, proc: subprocess.Popen) -> None:
        while proc.poll() is None:
            if self.cancel_event.wait(0.5):
                self.log("[cancel requested: terminating process group]")
                self._kill(proc)
                return

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            return
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(pgid, sig)
            except OSError:
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue
