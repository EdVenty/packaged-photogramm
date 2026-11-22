import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("server", "worker", "tui"):
    sys.path.insert(0, str(ROOT / p))


def make_jpeg(color=(30, 120, 160), size=(64, 48)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def app(data_dir, monkeypatch):
    from photogramm_api.main import Settings, create_app
    monkeypatch.setenv("PG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PG_API_TOKEN", "")
    s = Settings()
    s.watchdog_interval = 3600
    return create_app(s)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


ALL_KINDS = ["cpu", "sfm", "sfm_refrac", "mvs", "splat", "splat_uw"]


@pytest.fixture
def dataset(client):
    # a worker advertising every module, so job creation is not rejected as "modules_missing"
    client.post("/internal/workers/heartbeat", json={"worker_id": "registry", "kinds": ALL_KINDS})
    r = client.post("/api/datasets", json={"name": "scene"})
    did = r.json()["id"]
    files = [("files", (f"img{i:02d}.jpg", make_jpeg((20 * i % 255, 100, 150)), "image/jpeg"))
             for i in range(4)]
    r = client.post(f"/api/datasets/{did}/images", files=files)
    assert r.status_code == 200, r.text
    return did
