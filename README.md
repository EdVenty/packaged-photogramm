# packaged-photogramm

Сервер для восстановления **текстурированного меша** и **3D Gaussian Splatting** по набору фотографий.
Собран из готовых open-source инструментов и упакован в Docker Compose; управляется через TUI.

* **SfM:** COLMAP (CUDA); для подводной съёмки ещё и [Refractive COLMAP](https://github.com/MDSKiel/colmap_underwater) (модели иллюминаторов flat port / dome port)
* **Меш:** OpenMVS (плотное облако → меш → опционально refine → текстура, экспорт в OBJ/PLY)
* **Сплаты:** nerfstudio `splatfacto` / `splatfacto-big`, для подводных сцен [WaterSplatting](https://github.com/water-splatting/water-splatting)
* **Подводная цветокоррекция:** компенсация красного канала (Ancuti), gray-world, CLAHE, UDCP
* **Управление:** REST API (FastAPI) и TUI `pgtui` (Textual)

Подробности: [docs/SPEC.md](docs/SPEC.md) (спецификация), [docs/PLAN.md](docs/PLAN.md) (план и риски).

## Требования

* Linux, NVIDIA GPU (рекомендуется ≥ 8 ГБ VRAM; на 24 ГБ спокойно работают и сплаты, и MVS), драйвер ≥ 525
* Docker Engine ≥ 24 с плагином Compose v2 и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
* Около 40 ГБ места под образы; первая сборка занимает 30–90 минут (из исходников собираются COLMAP-форк, OpenMVS и CUDA-расширения)

Проверка GPU в Docker: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

## Установка и выбор модулей

Систему можно ставить частично. **Ядро** (API, предобработка фото, COLMAP SfM) ставится всегда, остальное — модули:

| Модуль | Сервис | Что даёт | Сборка |
|--------|--------|----------|--------|
| *core* | `api`, `worker-cpu`, `worker-sfm` | API, очередь, ресайз и цветокоррекция, позы камер и разреженное облако | быстро (готовый образ COLMAP) |
| `mesh` | `worker-mvs` | плотное облако, меш, текстура (OpenMVS) | ~15–30 мин |
| `splat` | `worker-splat` | Gaussian Splatting (nerfstudio splatfacto) | ~15–40 мин |
| ↳ WaterSplatting | внутри `worker-splat` | подводные сплаты (`INSTALL_WATERSPLATTING=1`) | +5–15 мин |
| `refrac` | `worker-sfm-refrac` | Refractive COLMAP для подводных корпусов | ~20–40 мин |

Интерактивная установка спросит, какие модули нужны, определит архитектуру GPU (чтобы сборка шла быстрее),
сгенерирует API-токен, запишет `.env`, затем соберёт и запустит выбранное:

```bash
./install.sh
```

Неинтерактивно:

```bash
./install.sh -y --modules mesh,splat --water-splatting no   # без рефракции и WaterSplatting
./install.sh -y --modules all                               # всё
./install.sh -y --modules none                              # только ядро
./install.sh --modules splat --no-build --no-start          # только переписать .env
```

Повторный запуск `install.sh` меняет набор модулей: новые соберутся, а снятые будут остановлены.

Вручную то же самое делается через профили Docker Compose (`mesh`, `splat`, `refrac`, `all`) в `.env`:

```bash
cp .env.example .env
# COMPOSE_PROFILES=mesh,splat      ← выбранные модули
# INSTALL_WATERSPLATTING=1         ← WaterSplatting внутри модуля splat
docker compose build && docker compose up -d
curl http://localhost:8000/health
```

Сервер знает, какие модули запущены (воркеры регистрируются сами). Если задача требует отсутствующего модуля,
например пресет `underwater` без WaterSplatting или `full` без `mesh`, API отвечает `409 modules_missing`.
TUI в этом случае предлагает всё равно поставить задачу в очередь (`queue_if_unavailable: true`): стадия подождёт,
пока модуль не появится. Состав модулей виден в строке статуса TUI и в `GET /api/info`.

Воркеры масштабируются через `--scale`, например при нескольких GPU: `docker compose up -d --scale worker-splat=2`
(чтобы назначить GPU конкретно, задайте `device_ids` в секции `deploy` нужного сервиса).

Данные лежат в `./data` (настраивается через `PG_DATA_DIR`): `datasets/`, `jobs/<id>/outputs/`, `db.sqlite`.

## TUI

```bash
pip install ./tui                    # или: pipx install ./tui
pgtui --server http://SERVER:8000 --token <PG_API_TOKEN>
# переменные окружения: PG_SERVER, PG_TOKEN
```

Или прямо на сервере в контейнере (текущая папка монтируется в `/host`):

```bash
docker compose run --rm tui
```

Типичный сценарий:

1. `u`: новый датасет, указать папку с фото или `.zip` (загрузка идёт батчами с прогрессом).
2. `n`: новая задача, выбрать датасет, результат (`full` / `mesh` / `splat`) и пресет; при необходимости поправить параметры.
3. `Enter` на задаче: стадии, живой лог, артефакты; `s` скачать всё архивом (с распаковкой), `a` скачать выбранный файл.
4. `c` отменить, `r` перезапустить упавшие стадии, `d` удалить.

## Пресеты

| Пресет | Когда использовать |
|--------|-------------------|
| `standard` | обычная съёмка объекта или сцены |
| `fast` | быстрое превью (ресайз до 1600 px, грубый MVS, 7k итераций сплатов) |
| `underwater` | подводная съёмка без калибровки корпуса: OPENCV-модель поглощает часть рефракции, цветокоррекция для поиска признаков, DSP-SIFT, guided matching, WaterSplatting |
| `underwater-refractive-flat` | известна калибровка плоского иллюминатора: Refractive COLMAP (FLATPORT) |
| `underwater-refractive-dome` | то же для купольного иллюминатора (DOMEPORT) |

### Подводный режим: на что обратить внимание

* **Цветокоррекция** (`underwater.enhance`): `redcomp` подходит как вариант по умолчанию; для мутной воды попробуйте `udcp`.
  При `enhance_for = sfm` исправленные снимки нужны только для поиска признаков, а в текстуры и сплаты идут оригиналы.
  При `all` откорректированные снимки используются везде.
* **Рефракционная модель** требует параметров корпуса (8 чисел):
  * FLATPORT: `nx,ny,nz,d_int,thickness,n_air,n_glass,n_water`, где нормаль к стеклу задаётся в системе координат камеры, а расстояния в метрах
  * DOMEPORT: `cx,cy,cz,radius,thickness,n_air,n_glass,n_water`, где смещение центра купола и радиус задаются в метрах

  Желательно также указать in-air калибровку в `sfm.camera_params` (например, `fx,fy,cx,cy,k1,k2,p1,p2` для OPENCV).
  Калибровку корпуса можно получить, например, в [CalibMar](https://cau-git.rz.uni-kiel.de/inf-ag-koeser/calibmar).
  Флаг `refine_refrac_params` уточняет параметры корпуса в bundle adjustment, но без гарантии сходимости.
* **Ограничение:** OpenMVS и nerfstudio не поддерживают рефракцию. Позы и разреженное облако считаются с её учётом, а плотная реконструкция и сплаты строятся по undistorted pinhole-изображениям.
* **WaterSplatting** отдельно моделирует водную среду, поэтому в `splat.ply` остаётся сцена «без воды».
* Для трансект и видео-раскадровок используйте `matcher = sequential`.

## Результаты (`outputs/`)

| Файл | Описание |
|------|----------|
| `sparse/*.txt`, `sparse_points.ply` | модель COLMAP (позы камер) и разреженное облако |
| `dense_points.ply` | плотное облако OpenMVS |
| `mesh.ply` | меш без текстуры |
| `mesh_textured/scene_textured.obj` (+`.mtl`, текстуры) | текстурированный меш |
| `splat.ply` | Gaussian Splatting в формате INRIA 3DGS (SuperSplat, gsplat-вьюеры и т.п.) |

## API

Документация OpenAPI доступна по адресу `http://SERVER:8000/docs`. Пример без TUI:

```bash
H="Authorization: Bearer $PG_API_TOKEN"
DS=$(curl -s -H "$H" -X POST localhost:8000/api/datasets -H 'content-type: application/json' -d '{"name":"reef"}' | jq -r .id)
curl -s -H "$H" -F file=@photos.zip localhost:8000/api/datasets/$DS/archive
curl -s -H "$H" -X POST localhost:8000/api/jobs -H 'content-type: application/json' \
     -d "{\"dataset_id\":\"$DS\",\"pipeline\":\"full\",\"preset\":\"underwater\"}"
```

## Разработка и тесты

```bash
pip install fastapi uvicorn python-multipart textual httpx pytest numpy pillow opencv-python-headless
python -m pytest -q tests
```

Тесты поднимают настоящий API-сервер и агент воркера, а вместо `colmap`, OpenMVS и `ns-train`/`ns-export` используют фейки
(`tests/fakebin`). Так проверяются граф стадий, отмена, повтор, lease, имена опций COLMAP (обычный и рефракционный),
формирование команд, обработка артефактов, предобработка и TUI.

## Траблшутинг

* **Воркер не берёт задачи:** проверьте вкладку «Воркеры» в TUI и `docker compose logs worker-sfm`. В строке статуса красным отмечены модули без онлайн-воркера; если модуль не установлен, добавьте его через `./install.sh`.
* **`could not select device driver "nvidia"`:** не установлен или не настроен NVIDIA Container Toolkit.
* **Мало зарегистрированных снимков:** включите `robust_features`, используйте `sequential` для съёмки с последовательным перекрытием, уменьшите `max_image_size`.
* **Нехватка VRAM в сплатах:** увеличьте `splat.downscale_factor` (2–4) или используйте пресет `fast`.
* **OpenMVS долго работает:** увеличьте `mvs.resolution_level` (2 = вдвое меньшее разрешение) и уменьшите `max_resolution`.
* **Стадия упала с «worker lost»:** воркер перезапустился или завис, задачу можно перезапустить клавишей `r`.
