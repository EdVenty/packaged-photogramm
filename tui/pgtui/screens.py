"""Modal dialogs and the job detail screen."""
from __future__ import annotations

import copy
import time
import zipfile
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, ProgressBar,
                             RichLog, Select, Static, Switch)

from .client import ApiClient, ApiError, find_images
from .fmt import STATUS_STYLE, duration, human_size, pct, status_text


# =============================================================================== confirm
class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "no", "Cancel"), Binding("y", "yes", "Yes")]

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog small"):
            yield Label(self.question, classes="title")
            with Horizontal(classes="buttons"):
                yield Button("Да (y)", id="yes", variant="error")
                yield Button("Отмена (esc)", id="no")

    @on(Button.Pressed, "#yes")
    def action_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def action_no(self) -> None:
        self.dismiss(False)


# =============================================================================== dataset
class NewDatasetScreen(ModalScreen[dict | None]):
    """Ask for a dataset name and a local folder / zip to upload."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, dataset: dict | None = None):
        super().__init__()
        self.dataset = dataset  # upload into an existing dataset when given

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            if self.dataset:
                yield Label(f"Загрузить фото в датасет «{self.dataset['name']}»", classes="title")
            else:
                yield Label("Новый датасет", classes="title")
                yield Label("Имя")
                yield Input(placeholder="my-scene", id="name")
            yield Label("Путь к папке с фото или .zip (на этой машине)")
            yield Input(placeholder="/path/to/photos  или  photos.zip", id="path")
            yield Static("", id="hint", classes="hint")
            with Horizontal(classes="buttons"):
                yield Button("Загрузить", id="ok", variant="primary")
                yield Button("Отмена", id="cancel")

    @on(Input.Changed, "#path")
    def _preview(self, ev: Input.Changed) -> None:
        hint = self.query_one("#hint", Static)
        p = Path(ev.value).expanduser()
        if p.is_dir():
            imgs = find_images(p)
            hint.update(f"найдено изображений: {len(imgs)} "
                        f"({human_size(sum(i.stat().st_size for i in imgs))})")
        elif p.is_file() and p.suffix.lower() == ".zip":
            hint.update(f"zip-архив, {human_size(p.stat().st_size)}")
        else:
            hint.update("")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def _ok(self) -> None:
        path = Path(self.query_one("#path", Input).value.strip()).expanduser()
        if not (path.is_dir() or (path.is_file() and path.suffix.lower() == ".zip")):
            self.notify("Укажите существующую папку или .zip", severity="error")
            return
        if path.is_dir() and not find_images(path):
            self.notify("В папке нет изображений (jpg/png/tif/bmp)", severity="error")
            return
        name = "" if self.dataset else self.query_one("#name", Input).value.strip()
        self.dismiss({"name": name or path.stem, "path": str(path),
                      "dataset_id": self.dataset["id"] if self.dataset else None})

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class TransferScreen(ModalScreen[bool]):
    """Runs an upload or download in a background thread with a progress bar."""

    def __init__(self, title: str, task):
        super().__init__()
        self.title_text = title
        self.task = task  # callable(progress_cb, log_cb) -> str (result message)
        self._last = 0.0

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="title")
            yield ProgressBar(total=100, show_eta=True, id="bar")
            yield Static("", id="msg", classes="hint")
            with Horizontal(classes="buttons"):
                yield Button("Закрыть", id="close", disabled=True)

    def on_mount(self) -> None:
        self._run()

    def _progress(self, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last < 0.1 and done < total:
            return
        self._last = now
        self.app.call_from_thread(self._set, done, total)

    def _set(self, done: int, total: int) -> None:
        bar = self.query_one("#bar", ProgressBar)
        bar.update(total=max(total, 1), progress=done)
        self.query_one("#msg", Static).update(f"{human_size(done)} / {human_size(total)}")

    def _log(self, text: str) -> None:
        self.app.call_from_thread(self.query_one("#msg", Static).update, text)

    @work(thread=True)
    def _run(self) -> None:
        ok = True
        try:
            msg = self.task(self._progress, self._log)
        except (ApiError, OSError, zipfile.BadZipFile) as e:
            ok, msg = False, f"Ошибка: {e}"
        self.app.call_from_thread(self._finish, ok, msg)

    def _finish(self, ok: bool, msg: str) -> None:
        self.query_one("#msg", Static).update(Text(msg, style="green" if ok else "red"))
        btn = self.query_one("#close", Button)
        btn.disabled = False
        btn.focus()
        self._ok = ok

    @on(Button.Pressed, "#close")
    def _close(self) -> None:
        self.dismiss(getattr(self, "_ok", False))


# =============================================================================== new job
CAMERA_MODELS = ["SIMPLE_RADIAL", "OPENCV", "FULL_OPENCV", "RADIAL", "PINHOLE",
                 "SIMPLE_PINHOLE", "OPENCV_FISHEYE"]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class NewJobScreen(ModalScreen[dict | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("ctrl+s", "submit", "Start")]

    def __init__(self, datasets: list[dict], presets: dict, dataset_id: str | None = None,
                 modules: dict | None = None):
        super().__init__()
        self.modules = modules or {}
        self.datasets = datasets
        self.presets = presets["presets"]
        self.defaults = presets["defaults"]
        self.dataset_id = dataset_id or (datasets[0]["id"] if datasets else None)

    # field id -> (section, key, kind)
    FIELDS = {
        "pre_max": ("preprocess", "max_image_size", int),
        "cam_model": ("sfm", "camera_model", str),
        "cam_params": ("sfm", "camera_params", str),
        "matcher": ("sfm", "matcher", str),
        "robust": ("sfm", "robust_features", bool),
        "mvs_res": ("mvs", "resolution_level", int),
        "mvs_refine": ("mvs", "refine", bool),
        "mvs_texture": ("mvs", "texture", bool),
        "splat_method": ("splat", "method", str),
        "splat_iters": ("splat", "iterations", int),
        "splat_down": ("splat", "downscale_factor", int),
        "uw_on": ("underwater", "enabled", bool),
        "uw_enhance": ("underwater", "enhance", str),
        "uw_for": ("underwater", "enhance_for", str),
        "uw_camera": ("underwater", "camera", str),
        "uw_port": ("underwater", "port", str),
        "uw_refrac": ("underwater", "refrac_params", str),
    }

    def compose(self) -> ComposeResult:
        ds_opts = [(f"{d['name']} ({d['image_count']} фото)", d["id"]) for d in self.datasets]
        preset_opts = [(f"{k} — {v['description']}"[:90], k) for k, v in self.presets.items()]
        with Vertical(classes="dialog wide"):
            yield Label("Новая задача реконструкции", classes="title")
            mod = Text("Модули на сервере: ")
            for key, label in (("mesh", "меш"), ("splat", "сплаты"),
                               ("water_splatting", "water-splatting"), ("refrac", "рефракц. COLMAP")):
                ok = self.modules.get(key)
                mod.append(f"{'✔' if ok else '✘'} {label}  ", style="green" if ok else "red")
            yield Static(mod, id="modules")
            with VerticalScroll():
                with Grid(classes="form"):
                    yield Label("Имя задачи")
                    yield Input(placeholder="(по умолчанию: датасет-pipeline)", id="name")
                    yield Label("Датасет")
                    yield Select(ds_opts, value=self.dataset_id, allow_blank=False, id="dataset")
                    yield Label("Результат")
                    yield Select([("меш + сплаты (full)", "full"), ("только меш", "mesh"),
                                  ("только сплаты", "splat")], value="full", allow_blank=False,
                                 id="pipeline")
                    yield Label("Пресет")
                    yield Select(preset_opts, value="standard", allow_blank=False, id="preset")

                    yield Static("— Предобработка / SfM —", classes="section")
                    yield Static("")
                    yield Label("Ресайз до (px, 0=нет)")
                    yield Input(id="pre_max", type="integer")
                    yield Label("Модель камеры")
                    yield Select([(m, m) for m in CAMERA_MODELS], allow_blank=False,
                                 id="cam_model")
                    yield Label("Параметры камеры")
                    yield Input(placeholder="fx,fy,cx,cy,k1,k2,p1,p2 (необяз.)", id="cam_params")
                    yield Label("Матчер")
                    yield Select([("exhaustive (до ~500 фото)", "exhaustive"),
                                  ("sequential (видео/трансекты)", "sequential"),
                                  ("spatial (по GPS)", "spatial")], allow_blank=False,
                                 id="matcher")
                    yield Label("Устойчивые признаки")
                    yield Switch(id="robust")

                    yield Static("— Меш (OpenMVS) —", classes="section")
                    yield Static("")
                    yield Label("Resolution level")
                    yield Input(id="mvs_res", type="integer")
                    yield Label("RefineMesh")
                    yield Switch(id="mvs_refine")
                    yield Label("Текстура")
                    yield Switch(id="mvs_texture")

                    yield Static("— Сплаты (nerfstudio) —", classes="section")
                    yield Static("")
                    yield Label("Метод")
                    yield Select([("splatfacto", "splatfacto"),
                                  ("splatfacto-big", "splatfacto-big"),
                                  ("water-splatting (подводный)", "water-splatting")],
                                 allow_blank=False, id="splat_method")
                    yield Label("Итерации")
                    yield Input(id="splat_iters", type="integer")
                    yield Label("Downscale (0=авто)")
                    yield Input(id="splat_down", type="integer")

                    yield Static("— Подводный режим —", classes="section")
                    yield Static("")
                    yield Label("Включён")
                    yield Switch(id="uw_on")
                    yield Label("Цветокоррекция")
                    yield Select([("нет", "none"), ("redcomp (рекомендуется)", "redcomp"),
                                  ("UDCP dehaze", "udcp"), ("gray-world", "grayworld"),
                                  ("CLAHE", "clahe")], allow_blank=False, id="uw_enhance")
                    yield Label("Коррекция для")
                    yield Select([("только SfM", "sfm"), ("SfM + текстуры/сплаты", "all")],
                                 allow_blank=False, id="uw_for")
                    yield Label("Модель камеры")
                    yield Select([("pinhole-приближение", "pinhole"),
                                  ("рефракционная (Refractive COLMAP)", "refractive")],
                                 allow_blank=False, id="uw_camera")
                    yield Label("Иллюминатор")
                    yield Select([("flat port", "flat"), ("dome port", "dome")],
                                 allow_blank=False, id="uw_port")
                    yield Label("Рефракц. параметры")
                    yield Input(placeholder="8 чисел через запятую", id="uw_refrac")
                yield Static(
                    "FLATPORT: nx,ny,nz,d_int[m],thickness[m],n_air,n_glass,n_water\n"
                    "DOMEPORT: cx,cy,cz[m],radius[m],thickness[m],n_air,n_glass,n_water",
                    classes="hint")
            with Horizontal(classes="buttons"):
                yield Button("Запустить (ctrl+s)", id="ok", variant="primary")
                yield Button("Отмена", id="cancel")

    def on_mount(self) -> None:
        self._apply_preset("standard")

    @on(Select.Changed, "#preset")
    def _preset_changed(self, ev: Select.Changed) -> None:
        if isinstance(ev.value, str):
            self._apply_preset(ev.value)

    def _apply_preset(self, name: str) -> None:
        params = _merge(self.defaults, self.presets.get(name, {}).get("params", {}))
        for fid, (sec, key, kind) in self.FIELDS.items():
            val = params[sec][key]
            w = self.query_one(f"#{fid}")
            if isinstance(w, Switch):
                w.value = bool(val)
            elif isinstance(w, Select):
                w.value = val
            else:
                w.value = str(val)

    def _collect(self) -> dict:
        params: dict[str, dict[str, Any]] = {}
        for fid, (sec, key, kind) in self.FIELDS.items():
            w = self.query_one(f"#{fid}")
            v: Any = w.value
            if kind is int:
                v = int(v or 0)
            elif kind is bool:
                v = bool(v)
            else:
                v = "" if v is None or v == Select.BLANK else str(v)
            params.setdefault(sec, {})[key] = v
        return {
            "name": self.query_one("#name", Input).value.strip(),
            "dataset_id": self.query_one("#dataset", Select).value,
            "pipeline": self.query_one("#pipeline", Select).value,
            "preset": self.query_one("#preset", Select).value,
            "params": params,
        }

    @on(Button.Pressed, "#ok")
    def action_submit(self) -> None:
        try:
            payload = self._collect()
        except ValueError as e:
            self.notify(f"Неверное значение: {e}", severity="error")
            return
        if not payload["dataset_id"] or payload["dataset_id"] == Select.BLANK:
            self.notify("Нет датасета — сначала загрузите фото", severity="error")
            return
        self.dismiss(payload)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


# =============================================================================== download
class DownloadScreen(ModalScreen[dict | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, default_dir: str, what: str):
        super().__init__()
        self.default_dir = default_dir
        self.what = what

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(f"Скачать: {self.what}", classes="title")
            yield Label("Папка назначения")
            yield Input(value=self.default_dir, id="dir")
            with Horizontal(classes="row"):
                yield Switch(value=True, id="extract")
                yield Label(" распаковать zip")
            with Horizontal(classes="buttons"):
                yield Button("Скачать", id="ok", variant="primary")
                yield Button("Отмена", id="cancel")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def _ok(self) -> None:
        self.dismiss({"dir": self.query_one("#dir", Input).value.strip() or ".",
                      "extract": self.query_one("#extract", Switch).value})

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


# =============================================================================== job detail
class JobScreen(Screen):
    BINDINGS = [
        Binding("escape,b", "back", "Назад"),
        Binding("c", "cancel_job", "Отменить"),
        Binding("r", "retry_job", "Повторить"),
        Binding("s", "download_all", "Скачать всё"),
        Binding("a", "download_one", "Скачать файл"),
        Binding("f", "toggle_follow", "Автопрокрутка"),
    ]

    def __init__(self, client: ApiClient, job_id: str):
        super().__init__()
        self.client = client
        self.job_id = job_id
        self.job: dict | None = None
        self.stage: str | None = None
        self.log_offset = -1
        self.follow = True
        self._artifacts_at = 0.0
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("загрузка…", id="job-head")
        with Horizontal(id="job-body"):
            with Vertical(id="job-left"):
                yield Label("Стадии", classes="section")
                yield DataTable(id="stages", cursor_type="row", zebra_stripes=True)
                yield Label("Артефакты", classes="section")
                yield DataTable(id="artifacts", cursor_type="row", zebra_stripes=True)
            with Vertical(id="job-right"):
                yield Label("Лог", id="log-title", classes="section")
                yield RichLog(id="log", max_lines=20000, wrap=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        st = self.query_one("#stages", DataTable)
        st.add_columns("стадия", "очередь", "статус", "прогресс", "время", "воркер", "сообщение")
        at = self.query_one("#artifacts", DataTable)
        at.add_columns("файл", "размер")
        self.poll()
        self.set_interval(1.5, self.poll)

    @work(thread=True, exclusive=True, group="job-poll")
    def poll(self) -> None:
        try:
            job = self.client.job(self.job_id)
            stage = self.stage or self._default_stage(job)
            logs = self.client.logs(self.job_id, stage, self.log_offset) if stage else None
            arts = None
            if time.monotonic() - self._artifacts_at > 4:
                arts = self.client.artifacts(self.job_id)
        except ApiError as e:
            self.app.call_from_thread(self._error, str(e))
            return
        self.app.call_from_thread(self._apply, job, stage, logs, arts)

    @staticmethod
    def _default_stage(job: dict) -> str | None:
        stages = job.get("stages", [])
        for s in stages:
            if s["status"] == "running":
                return s["name"]
        for s in reversed(stages):
            if s["status"] in ("failed",):
                return s["name"]
        started = [s for s in stages if s["status"] != "pending"]
        return (started[-1] if started else stages[0])["name"] if stages else None

    def _error(self, msg: str) -> None:
        self.query_one("#job-head", Static).update(Text(f"Ошибка связи: {msg}", style="red"))

    def _apply(self, job: dict, stage: str | None, logs: dict | None,
               arts: list | None) -> None:
        self.job = job
        if self.stage is None and stage:
            self.stage = stage
        head = Text()
        head.append(f" {job['name']} ", style="bold reverse")
        head.append(f"  id {job['id']}  датасет {job['dataset_name']}  pipeline {job['pipeline']}"
                    f"  пресет {job.get('preset') or '-'}  ")
        head.append_text(status_text(job["status"]))
        head.append(f"  {pct(job['progress'])}")
        uw = job.get("params", {}).get("underwater", {})
        if uw.get("enabled"):
            head.append(f"  🌊 подводный: {uw.get('camera')}"
                        f"{'/' + uw.get('port') if uw.get('camera') == 'refractive' else ''}"
                        f", коррекция {uw.get('enhance')}", style="cyan")
        self.query_one("#job-head", Static).update(head)

        st = self.query_one("#stages", DataTable)
        cur = st.cursor_row
        st.clear()
        now = time.time()
        for s in job["stages"]:
            end = s.get("finished_at") or (now if s["status"] == "running" else None)
            took = duration(end - s["started_at"]) if s.get("started_at") and end else ""
            msg = s.get("error") or s.get("message") or ""
            st.add_row(s["name"], s["kind"], status_text(s["status"]),
                       pct(s["progress"]) if s["status"] in ("running", "done") else "",
                       took, (s.get("worker_id") or "")[:28],
                       Text(msg[:120], style="red" if s.get("error") else ""), key=s["name"])
        if job["stages"]:
            st.move_cursor(row=min(cur, len(job["stages"]) - 1))

        if arts is not None:
            self._artifacts_at = time.monotonic()
            at = self.query_one("#artifacts", DataTable)
            cur_a = at.cursor_row
            at.clear()
            for a in arts:
                at.add_row(a["path"], human_size(a["size"]), key=a["path"])
            if arts:
                at.move_cursor(row=min(cur_a, len(arts) - 1))

        if logs is not None and stage == self.stage:
            log = self.query_one("#log", RichLog)
            if logs["text"]:
                if self.log_offset == -1 and logs["offset"] > 0:
                    log.write(Text("… (начало лога пропущено)", style="dim"))
                for line in logs["text"].splitlines():
                    log.write(line, scroll_end=self.follow)
            self.log_offset = logs["next_offset"]
            self.query_one("#log-title", Label).update(
                f"Лог: {stage}  ({human_size(logs['size'])})"
                f"{'' if self.follow else '  [автопрокрутка выкл.]'}")

    @on(DataTable.RowSelected, "#stages")
    def _stage_selected(self, ev: DataTable.RowSelected) -> None:
        name = ev.row_key.value
        if name and name != self.stage:
            self.stage = name
            self.log_offset = -1
            self.query_one("#log", RichLog).clear()
            self.poll()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        self.notify(f"Автопрокрутка {'вкл' if self.follow else 'выкл'}")

    def action_cancel_job(self) -> None:
        def go(ok: bool | None) -> None:
            if ok:
                self._call(self.client.cancel_job, "Отмена запрошена")
        self.app.push_screen(ConfirmScreen("Отменить задачу?"), go)

    def action_retry_job(self) -> None:
        self._call(self.client.retry_job, "Упавшие стадии поставлены в очередь")

    @work(thread=True)
    def _call(self, fn, ok_msg: str) -> None:
        try:
            fn(self.job_id)
            self.app.call_from_thread(self.notify, ok_msg)
        except ApiError as e:
            self.app.call_from_thread(self.notify, str(e), severity="error")

    def _download(self, server_path: str, filename: str, is_zip: bool) -> None:
        name = self.job["name"] if self.job else self.job_id
        default = str(Path.cwd() / "downloads" / f"{name}-{self.job_id}")

        def go(res: dict | None) -> None:
            if not res:
                return
            dest_dir = Path(res["dir"]).expanduser()
            dest = dest_dir / filename

            def task(progress, log) -> str:
                self.client.download(server_path, dest, progress)
                if is_zip and res["extract"]:
                    log("распаковка…")
                    with zipfile.ZipFile(dest) as zf:
                        zf.extractall(dest_dir)
                    dest.unlink()
                    return f"Готово: {dest_dir}"
                return f"Готово: {dest}"

            self.app.push_screen(TransferScreen(f"Скачивание {filename}", task))

        self.app.push_screen(DownloadScreen(default, filename), go)

    def action_download_all(self) -> None:
        self._download(f"/api/jobs/{self.job_id}/archive", f"{self.job_id}.zip", True)

    def action_download_one(self) -> None:
        at = self.query_one("#artifacts", DataTable)
        if at.row_count == 0:
            self.notify("Артефактов пока нет", severity="warning")
            return
        path = at.coordinate_to_cell_key(at.cursor_coordinate).row_key.value
        self._download(f"/api/jobs/{self.job_id}/artifacts/{path}", path,
                       path.lower().endswith(".zip"))


__all__ = ["ConfirmScreen", "NewDatasetScreen", "TransferScreen", "NewJobScreen",
           "DownloadScreen", "JobScreen", "STATUS_STYLE"]
