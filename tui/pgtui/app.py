"""pgtui — terminal UI for the packaged-photogramm reconstruction server."""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import (Button, DataTable, Footer, Header, Static, TabbedContent,
                             TabPane)

from . import __version__
from .client import ApiClient, ApiError, find_images
from .fmt import human_size, pct, progress_bar, status_text
from .screens import ConfirmScreen, JobScreen, NewDatasetScreen, NewJobScreen, TransferScreen

CSS = """
Screen { background: $surface; }
#status { height: 1; padding: 0 1; background: $panel; }
.toolbar { height: 3; padding: 0 1; }
.toolbar Button { margin-right: 1; min-width: 10; }
DataTable { height: 1fr; }
.dialog { width: 80; height: auto; max-height: 90%; padding: 1 2; border: thick $primary;
          background: $boost; }
.dialog.small { width: 60; }
.dialog.wide { width: 110; height: 90%; }
ModalScreen { align: center middle; }
.title { text-style: bold; margin-bottom: 1; }
.hint { color: $text-muted; margin: 1 0 0 0; }
.buttons { height: 3; margin-top: 1; align-horizontal: right; }
.buttons Button { margin-left: 1; }
.row { height: 3; }
.form { grid-size: 2; grid-columns: 26 1fr; grid-gutter: 0 1; height: auto; }
.form Label { padding: 1 0 0 0; }
.section { text-style: bold; color: $accent; padding: 1 0 0 0; }
#job-head { height: 3; padding: 1 1 0 1; }
#job-body { height: 1fr; }
#job-left { width: 1fr; }
#job-right { width: 1fr; }
#stages { height: 1fr; }
#artifacts { height: 1fr; }
#log { height: 1fr; border: round $panel-lighten-2; }
"""


class PhotogrammApp(App):
    TITLE = "packaged-photogramm"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Выход"),
        Binding("n", "new_job", "Новая задача"),
        Binding("u", "upload", "Загрузить фото"),
        Binding("c", "cancel", "Отменить"),
        Binding("r", "retry", "Повторить"),
        Binding("d", "delete", "Удалить"),
        Binding("f5", "refresh", "Обновить"),
        Binding("1", "tab('jobs')", "Задачи", show=False),
        Binding("2", "tab('datasets')", "Датасеты", show=False),
        Binding("3", "tab('workers')", "Воркеры", show=False),
    ]

    def __init__(self, server: str, token: str = "", refresh: float = 2.0):
        super().__init__()
        self.client = ApiClient(server, token)
        self.refresh_interval = refresh
        self.job_rows: list[dict] = []
        self.dataset_rows: list[dict] = []
        self.worker_rows: list[dict] = []
        self.sub_title = server

    MAIN_ONLY = {"new_job", "upload", "cancel", "retry", "delete", "tab", "refresh"}

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        # Main-screen actions are disabled (and hidden) while a job screen or dialog is open.
        if action in self.MAIN_ONLY and len(self.screen_stack) > 1:
            return False
        return True

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="jobs", id="tabs"):
            with TabPane("Задачи [1]", id="jobs"):
                with Horizontal(classes="toolbar"):
                    yield Button("Новая (n)", id="btn-new-job", variant="primary")
                    yield Button("Открыть (⏎)", id="btn-open")
                    yield Button("Отменить (c)", id="btn-cancel")
                    yield Button("Повторить (r)", id="btn-retry")
                    yield Button("Удалить (d)", id="btn-del-job", variant="error")
                yield DataTable(id="jobs-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Датасеты [2]", id="datasets"):
                with Horizontal(classes="toolbar"):
                    yield Button("Новый датасет (u)", id="btn-new-ds", variant="primary")
                    yield Button("Догрузить фото", id="btn-add-ds")
                    yield Button("Задача из датасета", id="btn-job-ds")
                    yield Button("Удалить (d)", id="btn-del-ds", variant="error")
                yield DataTable(id="ds-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Воркеры [3]", id="workers"):
                yield DataTable(id="workers-table", cursor_type="row", zebra_stripes=True)
        yield Static("подключение…", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#jobs-table", DataTable).add_columns(
            "id", "имя", "датасет", "pipeline", "режим", "статус", "прогресс", "стадии", "создана")
        self.query_one("#ds-table", DataTable).add_columns("id", "имя", "фото", "размер", "создан")
        self.query_one("#workers-table", DataTable).add_columns(
            "воркер", "очереди", "онлайн", "посл. пульс", "текущая стадия")
        self.refresh_data()
        self.set_interval(self.refresh_interval, self.refresh_data)

    def action_refresh(self) -> None:
        self.refresh_data()

    # ------------------------------------------------------------------ data refresh
    @work(thread=True, exclusive=True, group="refresh")
    def refresh_data(self) -> None:
        try:
            info = self.client.info()
            jobs = self.client.jobs()
            datasets = self.client.datasets()
            workers = self.client.workers()
        except ApiError as e:
            self.call_from_thread(self._set_status, Text(f"✘ {e}", style="red"))
            return
        self.call_from_thread(self._apply, info, jobs, datasets, workers)

    def _set_status(self, t: Text) -> None:
        self.query_one("#status", Static).update(t)

    @staticmethod
    def _refill(table: DataTable, rows: list[tuple[str, tuple]]) -> None:
        key = None
        if table.row_count:
            try:
                key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
            except Exception:  # noqa: BLE001
                key = None
        table.clear()
        for k, row in rows:
            table.add_row(*row, key=k)
        if key is not None:
            for i, (k, _) in enumerate(rows):
                if k == key:
                    table.move_cursor(row=i)
                    break

    def _apply(self, info: dict, jobs: list, datasets: list, workers: list) -> None:
        self.job_rows, self.dataset_rows, self.worker_rows = jobs, datasets, workers
        kinds = set(info.get("kinds_online", []))
        st = Text()
        st.append("● ", style="green")
        st.append(f"сервер v{info['version']}  воркеров онлайн: {info['workers_online']}  ")
        for k in ("cpu", "sfm", "sfm_refrac", "mvs", "splat"):
            st.append(f" {k}", style="green" if k in kinds else "red dim")
        self._set_status(st)

        jrows = []
        for j in jobs:
            stages = " ".join(f"{s['name']}:{s['status'][0]}" for s in j["stages"])
            mode = {"standard": "обычный", "underwater": "🌊 подводный",
                    "uw-refractive": "🌊 рефракц."}.get(j.get("mode", ""), j.get("mode", ""))
            jrows.append((j["id"], (j["id"], j["name"][:32], j["dataset_name"][:20], j["pipeline"],
                                    mode, status_text(j["status"]),
                                    f"{progress_bar(j['progress'])} {pct(j['progress'])}",
                                    stages, _ts(j["created_at"]))))
        self._refill(self.query_one("#jobs-table", DataTable), jrows)

        drows = [(d["id"], (d["id"], d["name"], str(d["image_count"]), human_size(d["size"]),
                            _ts(d["created_at"]))) for d in datasets]
        self._refill(self.query_one("#ds-table", DataTable), drows)

        wrows = [(w["id"], (w["id"], ",".join(w["kinds"]),
                            Text("да", style="green") if w["online"] else Text("нет", style="red"),
                            f"{w['age']:.0f} с назад", w.get("current_stage") or "—"))
                 for w in workers]
        self._refill(self.query_one("#workers-table", DataTable), wrows)

    # ------------------------------------------------------------------ helpers
    def _active_tab(self) -> str:
        return self.query_one("#tabs", TabbedContent).active

    def _selected(self, table_id: str) -> str | None:
        t = self.query_one(table_id, DataTable)
        if not t.row_count:
            return None
        try:
            return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001
            return None

    def _selected_job(self) -> str | None:
        jid = self._selected("#jobs-table")
        if not jid:
            self.notify("Не выбрана задача", severity="warning")
        return jid

    def _selected_ds(self) -> dict | None:
        did = self._selected("#ds-table")
        ds = next((d for d in self.dataset_rows if d["id"] == did), None)
        if not ds:
            self.notify("Не выбран датасет", severity="warning")
        return ds

    @work(thread=True)
    def _call(self, fn, *args, ok: str = "") -> None:
        try:
            fn(*args)
            if ok:
                self.call_from_thread(self.notify, ok)
        except ApiError as e:
            self.call_from_thread(self.notify, str(e), severity="error", timeout=8)
        self.call_from_thread(self.refresh_data)

    def action_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    # ------------------------------------------------------------------ actions
    def action_open(self) -> None:
        if self._active_tab() == "jobs":
            jid = self._selected_job()
            if jid:
                self.push_screen(JobScreen(self.client, jid))

    @on(DataTable.RowSelected, "#jobs-table")
    def _job_row(self, ev: DataTable.RowSelected) -> None:
        self.push_screen(JobScreen(self.client, ev.row_key.value))

    @on(Button.Pressed, "#btn-open")
    def _b_open(self) -> None:
        jid = self._selected_job()
        if jid:
            self.push_screen(JobScreen(self.client, jid))

    @on(Button.Pressed, "#btn-cancel")
    def action_cancel(self) -> None:
        jid = self._selected_job()
        if not jid:
            return
        self.push_screen(ConfirmScreen(f"Отменить задачу {jid}?"),
                         lambda ok: ok and self._call(self.client.cancel_job, jid,
                                                      ok="Отмена запрошена"))

    @on(Button.Pressed, "#btn-retry")
    def action_retry(self) -> None:
        jid = self._selected_job()
        if jid:
            self._call(self.client.retry_job, jid, ok="Стадии поставлены в очередь")

    @on(Button.Pressed, "#btn-del-job")
    def _del_job(self) -> None:
        jid = self._selected_job()
        if jid:
            self.push_screen(ConfirmScreen(f"Удалить задачу {jid} и её результаты?"),
                             lambda ok: ok and self._call(self.client.delete_job, jid,
                                                          ok="Задача удалена"))

    @on(Button.Pressed, "#btn-del-ds")
    def _del_ds(self) -> None:
        ds = self._selected_ds()
        if ds:
            self.push_screen(
                ConfirmScreen(f"Удалить датасет «{ds['name']}» со всеми фото и задачами?"),
                lambda ok: ok and self._call(self.client.delete_dataset, ds["id"],
                                             ok="Датасет удалён"))

    def action_delete(self) -> None:
        if self._active_tab() == "datasets":
            self._del_ds()
        elif self._active_tab() == "jobs":
            self._del_job()

    @on(Button.Pressed, "#btn-new-ds")
    def action_upload(self) -> None:
        self.push_screen(NewDatasetScreen(), self._do_upload)

    @on(Button.Pressed, "#btn-add-ds")
    def _add_ds(self) -> None:
        ds = self._selected_ds()
        if ds:
            self.push_screen(NewDatasetScreen(ds), self._do_upload)

    def _do_upload(self, res: dict | None) -> None:
        if not res:
            return
        path = Path(res["path"])

        def task(progress, log) -> str:
            did = res["dataset_id"]
            if not did:
                did = self.client.create_dataset(res["name"])["id"]
            if path.is_dir():
                n = self.client.upload_images(did, find_images(path), progress)
            else:
                log("загрузка архива…")
                n = self.client.upload_archive(did, path, progress)
            return f"Загружено изображений: {n} (датасет {did})"

        self.push_screen(TransferScreen(f"Загрузка {path.name}", task),
                         lambda _ok: self.refresh_data())

    @on(Button.Pressed, "#btn-job-ds")
    def _job_from_ds(self) -> None:
        ds = self._selected_ds()
        if ds:
            self._open_new_job(ds["id"])

    @on(Button.Pressed, "#btn-new-job")
    def action_new_job(self) -> None:
        did = self._selected("#ds-table") if self._active_tab() == "datasets" else None
        self._open_new_job(did)

    @work(thread=True)
    def _open_new_job(self, dataset_id: str | None) -> None:
        try:
            presets = self.client.presets()
            datasets = self.client.datasets()
        except ApiError as e:
            self.call_from_thread(self.notify, str(e), severity="error")
            return
        if not datasets:
            self.call_from_thread(self.notify, "Сначала создайте датасет (u)", severity="warning")
            return
        self.call_from_thread(self.push_screen, NewJobScreen(datasets, presets, dataset_id),
                              self._submit_job)

    def _submit_job(self, payload: dict | None) -> None:
        if payload:
            self._create_job(payload)

    @work(thread=True)
    def _create_job(self, payload: dict) -> None:
        try:
            job = self.client.create_job(payload)
        except ApiError as e:
            self.call_from_thread(self.notify, str(e), severity="error", timeout=12)
            return
        self.call_from_thread(self.notify, f"Задача {job['id']} создана: "
                                           f"{', '.join(s['name'] for s in job['stages'])}")
        self.call_from_thread(self.action_tab, "jobs")
        self.call_from_thread(self.refresh_data)

    def on_unmount(self) -> None:
        self.client.close()


def _ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pgtui", description="TUI for packaged-photogramm")
    ap.add_argument("--server", "-s", default=os.environ.get("PG_SERVER", "http://localhost:8000"),
                    help="API URL (env PG_SERVER)")
    ap.add_argument("--token", "-t", default=os.environ.get("PG_TOKEN", ""),
                    help="API token (env PG_TOKEN)")
    ap.add_argument("--refresh", type=float, default=2.0, help="refresh interval, seconds")
    ap.add_argument("--version", action="version", version=f"pgtui {__version__}")
    a = ap.parse_args(argv)
    PhotogrammApp(a.server, a.token, a.refresh).run()


if __name__ == "__main__":
    main()
