"""Formatting helpers."""
from __future__ import annotations

from rich.text import Text

STATUS_STYLE = {
    "queued": "yellow", "pending": "dim", "running": "bold cyan", "done": "green",
    "failed": "bold red", "cancelled": "magenta", "skipped": "dim",
}
STATUS_ICON = {
    "queued": "◷", "pending": "·", "running": "▶", "done": "✔", "failed": "✘",
    "cancelled": "■", "skipped": "↷",
}


def status_text(status: str) -> Text:
    return Text(f"{STATUS_ICON.get(status, '?')} {status}", style=STATUS_STYLE.get(status, ""))


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def duration(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def pct(x: float) -> str:
    return f"{100 * float(x or 0):.0f}%"


def progress_bar(x: float, width: int = 12) -> str:
    x = max(0.0, min(1.0, float(x or 0)))
    full = int(round(x * width))
    return "█" * full + "░" * (width - full)
