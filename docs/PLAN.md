# План реализации

Решения по уточняющим вопросам: NVIDIA GPU (CUDA); COLMAP + OpenMVS / nerfstudio (gsplat);
REST API + загрузка через TUI; Python (FastAPI + Textual); выход OBJ/PLY; плюс подводный режим.

## Этап 1. Спецификация и каркас
- [x] `docs/SPEC.md`, `docs/PLAN.md`, структура репозитория.

## Этап 2. API-сервер (`server/`)
- [x] SQLite-схема: datasets, jobs, stages, workers (WAL, атомарный claim через `BEGIN IMMEDIATE`).
- [x] Pydantic-модели параметров, пресеты, deep-merge.
- [x] Построение графа стадий по pipeline / подводному режиму.
- [x] Эндпоинты датасетов: создание, загрузка файлов и zip (защита от path traversal), удаление.
- [x] Эндпоинты задач: создание, список, отмена, повтор, удаление, логи с offset, артефакты, потоковый zip.
- [x] Внутренний API для воркеров: heartbeat, claim, stage heartbeat (с флагом отмены), finish.
- [x] Watchdog просроченных стадий; каскадный skip/cancel зависимых стадий.
- [x] Авторизация Bearer-токеном.

## Этап 3. Агент воркера (`worker/pgworker`, stdlib-only)
- [x] Цикл: регистрация → claim → выполнение → finish; поток heartbeat; отмена через kill группы процессов.
- [x] Запуск внешних команд с записью лога в `/data/jobs/<id>/logs/<stage>.log` и парсингом прогресса.
- [x] Стадия `preprocess`: ресайз, цветокоррекция (redcomp / grayworld / clahe / udcp), сохранение EXIF.
- [x] Стадия `sfm`: feature_extractor → matcher → mapper → выбор крупнейшей модели → image_undistorter → TXT;
      автоопределение имён опций по `colmap <cmd> -h` (совместимость COLMAP 3.10 форка и 3.12+);
      рефракционные опции (`ImageReader.camera_refrac_*`, `*.enable_refraction`), очистка cameras.txt.
- [x] Стадия `mvs`: OpenMVS-цепочка с автоопределением опций (`-p`, `-m`, `--export-type`).
- [x] Стадия `splat`: `ns-train` (splatfacto / splatfacto-big / water-splatting) → `ns-export gaussian-splat`
      с фолбэком на собственный экспорт PLY из чекпойнта.

## Этап 4. Docker
- [x] `docker/api.Dockerfile`, `cpu.Dockerfile`, `colmap.Dockerfile`, `colmap-refrac.Dockerfile`,
      `openmvs.Dockerfile`, `splat.Dockerfile`, `tui.Dockerfile`.
- [x] `docker-compose.yml`: общий том, GPU-резервации, healthcheck API, профиль `tui`, `.env.example`.

## Этап 5. TUI (`tui/`)
- [x] Клиент httpx, вкладки Datasets/Jobs/Workers, автообновление в фоне.
- [x] Модалки: новый датасет + загрузка (прогресс), новая задача (пресеты, подводные параметры), подтверждение.
- [x] Экран задачи: стадии, живой лог, артефакты, скачивание zip.

## Этап 6. Тесты и документация
- [x] Тесты API (state machine, отмена, повтор, lease, артефакты, загрузка zip).
- [x] Интеграционный тест: реальный uvicorn + агент воркера с фейковыми `colmap`/OpenMVS/`ns-train` в PATH.
- [x] Тест preprocess (OpenCV) и smoke-тест TUI (Textual pilot).
- [x] README: установка, запуск, подводный режим, траблшутинг.

## Риски
| Риск | Митигация |
|------|-----------|
| Сборка образов (COLMAP-форк, OpenMVS, WaterSplatting) долгая и зависит от сети | Явные build-args, кэш слоёв; сборка один раз. |
| Имена опций COLMAP меняются между версиями | Автоопределение по help-выводу, список кандидатов. |
| WaterSplatting закреплён на nerfstudio 1.1.4 / torch 2.1.2 / CUDA 11.8 | Один образ сплатов на этих версиях; splatfacto из того же nerfstudio. |
| `ns-export` может не поддерживать модель WaterSplatting | Собственный экспорт 3DGS PLY из чекпойнта. |
| Реальная GPU-сборка не проверялась в CI (нет GPU/Docker-демона) | Логика проверена на фейковых бинарях; команды соответствуют документации инструментов. |
