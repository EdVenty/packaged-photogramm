"""End-to-end: real uvicorn server + worker agent with fake COLMAP/OpenMVS/nerfstudio."""
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from conftest import make_jpeg

FAKEBIN = Path(__file__).parent / "fakebin"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server(data_dir, monkeypatch, tmp_path):
    from photogramm_api.main import Settings, create_app
    monkeypatch.setenv("PG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PG_API_TOKEN", "tok")
    monkeypatch.setenv("PATH", f"{FAKEBIN}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.log"))
    monkeypatch.setenv("PG_PREPROCESS_WORKERS", "2")
    port = _free_port()
    cfg = uvicorn.Config(create_app(Settings()), host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(url + "/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield url, tmp_path / "calls.log"
    srv.should_exit = True
    t.join(timeout=5)


def _agent(url, kinds, wid):
    from pgworker.agent import Agent
    from pgworker.client import Client
    return Agent(Client(url, "tok"), kinds, wid, heartbeat_interval=0.2)


def _setup(url, preset=None, params=None, pipeline="full"):
    from pgtui.client import ApiClient
    api = ApiClient(url, "tok")
    ds = api.create_dataset("reef")
    folder = Path(os.environ["FAKE_CALLS"]).parent / "photos"
    folder.mkdir(exist_ok=True)
    for i in range(5):
        (folder / f"IMG_{i}.jpg").write_bytes(make_jpeg((10 + 40 * i, 120, 170), (320, 240)))
    assert api.upload_images(ds["id"], sorted(folder.glob("*.jpg"))) == 5
    job = api.create_job({"dataset_id": ds["id"], "pipeline": pipeline, "preset": preset,
                          "params": params or {}})
    return api, job


def _run_all(url, job_id, api, agents):
    for _ in range(10):
        ran = False
        for a in agents:
            ran |= a.run_once()
        if not ran:
            break
    return api.job(job_id)


def test_standard_full_pipeline(server, monkeypatch):
    url, calls = server
    monkeypatch.setenv("FAKE_COLMAP_FLAVOR", "new")
    api, job = _setup(url)
    agents = [_agent(url, ["sfm"], "sfm"), _agent(url, ["mvs"], "mvs"),
              _agent(url, ["splat"], "splat")]
    j = _run_all(url, job["id"], api, agents)
    assert j["status"] == "done", [(s["name"], s["status"], s["error"]) for s in j["stages"]]
    arts = {a["path"] for a in api.artifacts(job["id"])}
    assert {"sparse/cameras.txt", "sparse_points.ply", "dense_points.ply", "mesh.ply",
            "splat.ply"} <= arts
    assert any(a.startswith("mesh_textured/") and a.endswith(".obj") for a in arts)
    log = calls.read_text()
    assert "--FeatureExtraction.use_gpu 1" in log           # new COLMAP option names
    assert "refrac" not in log
    assert "ns-train splatfacto" in log
    assert "--pointcloud-file" in log and "--export-type obj" in log
    sfm_log = api.logs(job["id"], "sfm", 0)["text"]
    assert "registered 5/5" in api.job(job["id"])["stages"][0]["message"]
    assert "Processed file [5/5]" in sfm_log


def test_underwater_refractive_pipeline(server, monkeypatch):
    url, calls = server
    monkeypatch.setenv("FAKE_COLMAP_FLAVOR", "refrac")
    api, job = _setup(url, preset="underwater-refractive-flat",
                      params={"preprocess": {"max_image_size": 160},
                              "underwater": {"enhance_for": "all"}})
    assert [s["kind"] for s in job["stages"]] == ["cpu", "sfm_refrac", "mvs", "splat"]
    # a standard sfm worker must not pick refractive work
    assert _agent(url, ["sfm"], "plain").run_once() is False
    agents = [_agent(url, ["cpu"], "cpu"), _agent(url, ["sfm_refrac"], "refrac"),
              _agent(url, ["mvs"], "mvs"), _agent(url, ["splat"], "splat")]
    j = _run_all(url, job["id"], api, agents)
    assert j["status"] == "done", [(s["name"], s["status"], s["error"]) for s in j["stages"]]

    from PIL import Image
    jobdir = Path(os.environ["PG_DATA_DIR"]) / "jobs" / job["id"]
    enh = sorted((jobdir / "work" / "images_enhanced").glob("*.jpg"))
    assert len(enh) == 5 and max(Image.open(enh[0]).size) == 160
    log = calls.read_text()
    assert "--ImageReader.camera_refrac_model FLATPORT" in log
    assert "--ImageReader.camera_refrac_params 0,0,1,0.01,0.01,1.0,1.49,1.334" in log
    assert "--TwoViewGeometry.enable_refraction 1" in log
    assert "--Mapper.enable_refraction 1" in log
    assert "--SiftExtraction.use_gpu 0" in log          # robust features force CPU SIFT
    assert "--SiftExtraction.domain_size_pooling 1" in log
    assert "--SiftMatching.guided_matching 1" in log
    assert "ns-train water-splatting" in log
    assert f"--image_path {jobdir}/work/images_enhanced" in log
    # refraction tail removed from the model handed to MVS / splats
    cams = (jobdir / "work" / "dense" / "sparse" / "cameras.txt").read_text()
    assert "FLATPORT" not in cams and "PINHOLE" in cams
    assert "FLATPORT" not in (jobdir / "outputs" / "sparse" / "cameras.txt").read_text()


def test_refractive_job_fails_cleanly_on_plain_colmap(server, monkeypatch):
    url, _ = server
    monkeypatch.setenv("FAKE_COLMAP_FLAVOR", "new")
    api, job = _setup(url, preset="underwater-refractive-dome",
                      params={"underwater": {"enhance": "none"}}, pipeline="mesh")
    st = _agent(url, ["sfm_refrac"], "misconfigured").run_once()
    assert st
    j = api.job(job["id"])
    assert j["stages"][0]["status"] == "failed"
    assert "Refractive COLMAP" in j["stages"][0]["error"]
    assert j["stages"][1]["status"] == "skipped"


def test_cancel_kills_running_process(server, monkeypatch, tmp_path):
    url, _ = server
    slow = tmp_path / "slowbin"
    slow.mkdir()
    (slow / "colmap").write_text("#!/bin/sh\n[ \"$2\" = \"-h\" ] && exit 0\nsleep 60\n")
    (slow / "colmap").chmod(0o755)
    monkeypatch.setenv("PATH", f"{slow}{os.pathsep}{os.environ['PATH']}")
    api, job = _setup(url, pipeline="mesh")
    agent = _agent(url, ["sfm"], "sfm")
    result = {}
    t = threading.Thread(target=lambda: result.update(ran=agent.run_once()))
    t0 = time.time()
    t.start()
    for _ in range(100):
        if api.job(job["id"])["stages"][0]["status"] == "running":
            break
        time.sleep(0.05)
    api.cancel_job(job["id"])
    t.join(timeout=30)
    assert not t.is_alive() and time.time() - t0 < 30
    j = api.job(job["id"])
    assert j["status"] == "cancelled"
    assert [s["status"] for s in j["stages"]] == ["cancelled", "cancelled"]
