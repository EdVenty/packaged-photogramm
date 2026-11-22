# Спецификация: packaged-photogramm

Сервер для восстановления **текстурированного меша** и **3D Gaussian Splatting** по фотодатасету.
Разворачивается одной командой `docker compose up -d`, управляется через TUI (`pgtui`) по REST API.
Есть отдельный **подводный режим** (цветокоррекция, устойчивые к низкому контрасту признаки,
рефракционная модель камеры, подводные сплаты).

## 1. Требования

| # | Требование |
|---|-----------|
| R1 | Хост Linux + NVIDIA GPU + `nvidia-container-toolkit`, Docker Compose v2. |
| R2 | Использовать готовые open-source решения, без собственной реализации алгоритмов реконструкции. |
| R3 | Пайплайны: `mesh` (меш), `splat` (сплаты), `full` (оба, общий SfM). |
| R4 | Подводный режим: (а) pinhole-приближение с цветокоррекцией и подстройкой параметров SfM; (б) рефракционный SfM (flat port / dome port); (в) сплаты с моделью водной среды. |
| R5 | Загрузка фото через API/TUI (отдельные файлы или zip), без доступа к ФС сервера. |
| R6 | Очередь задач, параллельное выполнение независимых стадий, отмена, повтор, живые логи, прогресс. |
| R7 | Скачивание результатов: OBJ (+MTL+текстуры), PLY (меш, плотное облако, разреженное облако), PLY 3DGS. |
| R8 | Опциональная авторизация по токену (для доступа к серверу по сети). |

## 2. Используемые решения

| Задача | Решение | Образ |
|-------|---------|-------|
| SfM (обычный и подводный pinhole) | **COLMAP** (CUDA) | официальный `colmap/colmap` + python3 |
| Рефракционный SfM | **Refractive COLMAP** — форк [MDSKiel/colmap_underwater](https://github.com/MDSKiel/colmap_underwater) (IROS 2024), модели `FLATPORT` / `DOMEPORT` | собирается из исходников (CUDA) |
| Плотное облако, меш, текстура | **OpenMVS** (`InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh → [RefineMesh] → TextureMesh`) | собирается официальным `docker/buildInDocker.sh` с CUDA |
| Сплаты | **nerfstudio** `splatfacto` / `splatfacto-big` (gsplat) | CUDA 11.8 + torch 2.1.2 + nerfstudio 1.1.4 |
| Подводные сплаты | **WaterSplatting** ([water-splatting/water-splatting](https://github.com/water-splatting/water-splatting), 3DV 2025) — плагин nerfstudio, метод `water-splatting` | тот же образ |
| Цветокоррекция подводных фото | OpenCV/NumPy: компенсация красного канала (Ancuti et al.), gray-world, CLAHE, UDCP-дехейз | `python:3.12-slim` |
| API | FastAPI + SQLite | `python:3.12-slim` |
| TUI | Textual + httpx | pip-пакет / контейнер |

## 3. Архитектура

```
             ┌────────── TUI (pgtui) ──────────┐
             │ HTTP  (Bearer token, опц.)       │
             ▼                                  │
     ┌──────────────┐   SQLite (jobs, stages)   │
     │   api        │◄──────────────┐           │
     │  FastAPI     │               │           │
     └──────┬───────┘               │           │
   /internal│claim / heartbeat / finish (pull)  │
   ┌────────┼──────────┬───────────┬──────────┐ │
   ▼        ▼          ▼           ▼          ▼
worker-cpu worker-sfm worker-sfm-  worker-mvs worker-splat
preprocess  COLMAP    refrac       OpenMVS    nerfstudio +
                      (форк)                  WaterSplatting
   └──────────── общий том /data ─────────────┘
```

* Воркеры **сами забирают** стадии (pull-модель) — API не знает о контейнерах, воркеры масштабируются `--scale`.
* Все контейнеры монтируют один том `/data` — тяжёлые данные не ходят через HTTP между сервисами.
* Агент воркера — Python stdlib-only (работает в любом базовом образе с python3 ≥ 3.8).

### 3.1 Раскладка `/data`

```
/data/db.sqlite
/data/datasets/<dataset_id>/images/*.jpg
/data/jobs/<job_id>/
    work/images/            # уменьшенные оригиналы (если задан preprocess.max_image_size)
    work/images_enhanced/   # цветокоррекция (подводный режим)
    work/database.db, work/sparse/<n>/   # COLMAP
    work/dense/{images,sparse}           # undistorted-сцена (TXT-модель, PINHOLE) — общий вход для MVS и сплатов
    mvs/                    # рабочие файлы OpenMVS
    splat/                  # рабочие файлы nerfstudio
    logs/<stage>.log
    outputs/                # итоговые артефакты (отдаются через API)
```

### 3.2 Граф стадий задачи

| Стадия | kind (очередь) | Когда | Зависит от |
|--------|---------------|-------|-----------|
| `preprocess` | `cpu` | `underwater.enhance != none` или `preprocess.max_image_size > 0` | — |
| `sfm` | `sfm` / `sfm_refrac` (если `underwater.camera = refractive`) | всегда | preprocess (если есть) |
| `mvs` | `mvs` | pipeline ∈ {mesh, full} | sfm |
| `splat` | `splat` / `splat_uw` (если `splat.method = water-splatting`) | pipeline ∈ {splat, full} | sfm |

`mvs` и `splat` в режиме `full` идут параллельно на разных воркерах.

### 3.3 Состояния

Стадия: `pending → running → done | failed | cancelled`, а также `skipped` (упала зависимость).
Задача (вычисляется): `queued` (ничего не стартовало), `running`, `done`, `failed`, `cancelled`.

* Отмена: `pending` → `cancelled`; `running` → флаг `cancel_requested`, воркер видит его в ответе heartbeat, убивает группу процессов, сообщает `cancelled`.
* Падение: зависимые стадии → `skipped`. Независимые продолжают работу.
* Повтор (`retry`): `failed|cancelled|skipped` → `pending`.
* Lease: стадия `running` без heartbeat дольше `PG_STAGE_LEASE_SEC` (по умолчанию 180 с) → `failed` («воркер потерян»).

### 3.4 Модули (выборочная установка)

| Модуль (compose-профиль) | Сервисы | kinds |
|--------------------------|---------|-------|
| core (всегда) | `api`, `worker-cpu`, `worker-sfm` | `cpu`, `sfm` |
| `mesh` | `worker-mvs` | `mvs` |
| `splat` | `worker-splat` | `splat`, плюс `splat_uw` при `INSTALL_WATERSPLATTING=1` |
| `refrac` | `worker-sfm-refrac` | `sfm_refrac` |
| `all` | все перечисленные | |

* Модули выбираются через `COMPOSE_PROFILES` в `.env`; `install.sh` делает это интерактивно или по флагам.
* Опциональные возможности образа (`splat_uw`) записываются при сборке в `/etc/pgworker/extra-kinds`,
  и агент воркера добавляет их к своим kinds.
* `GET /api/info` → `modules: {core, mesh, splat, water_splatting, refrac}` строится по онлайн-воркерам.
* `POST /api/jobs` отвечает `409 {code: modules_missing, missing_kinds, modules, message}`, если у какой-то
  стадии нет онлайн-воркера. С `queue_if_unavailable: true` задача ставится в очередь и ждёт.

## 4. Параметры задачи

`POST /api/jobs` принимает `preset` (необязательно) и `params` (частичные переопределения), сервер выполняет deep-merge `defaults ← preset ← params` и валидацию.

```jsonc
{
  "preprocess": { "max_image_size": 0 },          // 0 = без ресайза
  "sfm": {
    "camera_model": "SIMPLE_RADIAL",               // PINHOLE|SIMPLE_RADIAL|RADIAL|OPENCV|FULL_OPENCV|...
    "single_camera": true,
    "camera_params": "",                           // "fx,fy,cx,cy,..." — для рефракционного режима (in-air калибровка)
    "matcher": "exhaustive",                       // exhaustive|sequential|spatial
    "sequential_overlap": 10,
    "mapper": "incremental",                       // incremental|hierarchical
    "max_image_size": 3200,                        // для извлечения признаков
    "max_num_features": 8192,
    "robust_features": false,                      // DSP-SIFT + affine shape + низкий порог пиков (низкоконтрастные сцены)
    "guided_matching": false,
    "undistort_max_image_size": 3200
  },
  "mvs": {
    "resolution_level": 1, "max_resolution": 2560, "number_views_fuse": 3,
    "refine": false, "decimate": 1.0, "texture": true, "texture_resolution_level": 0
  },
  "splat": {
    "method": "splatfacto",                        // splatfacto|splatfacto-big|water-splatting
    "iterations": 30000,
    "downscale_factor": 1,                         // 0 = авто (nerfstudio)
    "extra_args": ""                               // доп. аргументы ns-train (до сабкоманды датапарсера)
  },
  "underwater": {
    "enabled": false,
    "enhance": "none",                             // none|grayworld|clahe|redcomp|udcp
    "enhance_for": "sfm",                          // sfm (только для поиска признаков) | all (и для текстур/сплатов)
    "camera": "pinhole",                           // pinhole (UW-pinhole приближение) | refractive
    "port": "flat",                                // flat|dome
    "refrac_params": "",                           // FLATPORT: nx,ny,nz,d_int,thickness,n_air,n_glass,n_water
                                                   // DOMEPORT: cx,cy,cz,radius,thickness,n_air,n_glass,n_water
    "refine_refrac_params": false
  }
}
```

### 4.1 Пресеты

| Пресет | Суть |
|--------|------|
| `standard` | Обычная съёмка: SIMPLE_RADIAL, exhaustive, splatfacto. |
| `fast` | Превью: ресайз до 1600, resolution_level 2, 7000 итераций сплатов. |
| `underwater` | UW-pinhole: OPENCV-модель (поглощает часть рефракции), `redcomp`-цветокоррекция для SfM, robust_features, guided matching, `water-splatting`. |
| `underwater-refractive-flat` | Refractive COLMAP, FLATPORT, требуется `refrac_params` (+ желательно `camera_params` in-air). |
| `underwater-refractive-dome` | То же для DOMEPORT. |

### 4.2 Ограничения подводного режима

* Refractive COLMAP даёт корректные позы и разреженное облако с учётом рефракции. OpenMVS и nerfstudio
  рефракцию не поддерживают, поэтому плотная реконструкция и сплаты строятся по undistorted pinhole-изображениям
  (приближение; для dome port, близкого к центрированному, погрешность мала; для flat port — заметнее на краях кадра).
* Формат БД/моделей форка несовместим с оригинальным COLMAP. Воркер конвертирует модель в TXT и
  удаляет рефракционные поля из `cameras.txt` перед передачей дальше.
* WaterSplatting обучает отдельную модель среды: экспортированный `splat.ply` содержит сцену «без воды».

## 5. REST API

Все `/api/*` и `/internal/*` требуют `Authorization: Bearer <PG_API_TOKEN>`, если токен задан.

| Метод | Путь | Описание |
|------|------|----------|
| GET | `/health` | Живость (без авторизации). |
| GET | `/api/info` | Версия, пресеты, онлайн-воркеры, доступные kinds и модули. |
| GET | `/api/presets` | Пресеты и параметры по умолчанию. |
| GET/POST | `/api/datasets` | Список / создать `{name}`. |
| GET/DELETE | `/api/datasets/{id}` | Детали (со списком изображений) / удалить (409 при активных задачах). |
| POST | `/api/datasets/{id}/images` | multipart `files[]` — загрузка изображений. |
| POST | `/api/datasets/{id}/archive` | multipart `file` — zip с изображениями. |
| GET/POST | `/api/jobs` | Список / создать `{dataset_id, name, pipeline, preset, params, queue_if_unavailable}`; 409 `modules_missing`. |
| GET/DELETE | `/api/jobs/{id}` | Детали со стадиями / удалить (409 если выполняется). |
| POST | `/api/jobs/{id}/cancel` \| `/retry` | Отмена / повтор. |
| GET | `/api/jobs/{id}/logs/{stage}?offset=N` | Инкрементальный лог; `offset=-1` — хвост 64 КБ. |
| GET | `/api/jobs/{id}/artifacts` | Список файлов в `outputs/`. |
| GET | `/api/jobs/{id}/artifacts/{path}` | Скачать файл. |
| GET | `/api/jobs/{id}/archive` | zip всех артефактов (потоково). |
| GET | `/api/workers` | Воркеры (последний heartbeat, kinds, текущая стадия). |
| POST | `/internal/workers/heartbeat` | Регистрация/пульс воркера. |
| POST | `/internal/claim` | `{worker_id, kinds}` → стадия или 204. |
| POST | `/internal/stages/{id}/heartbeat` | `{progress, message}` → `{cancel: bool}`. |
| POST | `/internal/stages/{id}/finish` | `{status: done|failed|cancelled, error?, message?}`. |

## 6. Артефакты (`outputs/`)

| Файл | Источник |
|------|----------|
| `sparse/{cameras,images,points3D}.txt`, `sparse_points.ply` | sfm |
| `dense_points.ply` | mvs (DensifyPointCloud) |
| `mesh.ply` | mvs (ReconstructMesh / RefineMesh) |
| `mesh_textured/scene_textured.obj` + `.mtl` + текстуры | mvs (TextureMesh) |
| `splat.ply` | splat (формат 3DGS INRIA, открывается в SuperSplat и др.) |

## 7. TUI

`pgtui --server http://host:8000 [--token ...]` (или `PG_SERVER`, `PG_TOKEN`).

* Вкладки **Datasets / Jobs / Workers**, автообновление.
* Создание датасета и загрузка папки/zip с прогрессом (батчами).
* Мастер новой задачи: датасет, pipeline, пресет, ключевые параметры (в т.ч. подводные).
* Экран задачи: стадии с прогрессом, живой лог выбранной стадии, список артефактов, скачивание zip.
* Горячие клавиши: `n` новая задача, `u` загрузить, `enter` детали, `c` отмена, `r` повтор, `d` удалить, `1/2/3` вкладки, `q` выход;
  на экране задачи: `s` скачать всё, `a` скачать выбранный файл, `f` автопрокрутка лога, `esc` назад.

## 8. Конфигурация (`.env`)

| Переменная | По умолчанию | |
|-----------|--------------|--|
| `PG_API_TOKEN` | пусто | токен API (пусто — без авторизации) |
| `PG_PORT` | 8000 | порт API на хосте |
| `PG_DATA_DIR` | `./data` | каталог данных на хосте |
| `CUDA_ARCHITECTURES` | `75;80;86;89` | архитектуры при сборке COLMAP-форка/OpenMVS |
| `TORCH_CUDA_ARCH_LIST` | `7.5;8.0;8.6;8.9` | для сборки CUDA-расширений WaterSplatting |
| `PG_STAGE_LEASE_SEC` | 180 | таймаут heartbeat стадии |
