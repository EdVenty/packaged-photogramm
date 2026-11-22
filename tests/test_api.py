import io
import time
import zipfile

from conftest import make_jpeg

W = "w1"


def claim(client, kinds, wid=W):
    r = client.post("/internal/claim", json={"worker_id": wid, "kinds": kinds})
    assert r.status_code in (200, 204), r.text
    return None if r.status_code == 204 else r.json()


def finish(client, sid, status="done", error="", wid=W):
    r = client.post(f"/internal/stages/{sid}/finish",
                    json={"worker_id": wid, "status": status, "error": error})
    assert r.json()["accepted"], r.text


def test_health_and_presets(client):
    assert client.get("/health").json() == {"status": "ok"}
    p = client.get("/api/presets").json()
    assert "underwater" in p["presets"] and "underwater-refractive-flat" in p["presets"]
    assert p["defaults"]["sfm"]["camera_model"] == "SIMPLE_RADIAL"


def test_dataset_upload_and_zip(client, dataset):
    d = client.get(f"/api/datasets/{dataset}").json()
    assert d["image_count"] == 4
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a/x.jpg", make_jpeg())
        zf.writestr("b/x.jpg", make_jpeg())
        zf.writestr("../../evil.jpg", make_jpeg())
        zf.writestr("notes.txt", "hello")
        zf.writestr("__MACOSX/a/._x.jpg", b"junk")
    r = client.post(f"/api/datasets/{dataset}/archive",
                    files={"file": ("p.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 200, r.text
    saved = r.json()["saved"]
    assert len(saved) == 3 and all("/" not in s and ".." not in s for s in saved)
    assert client.get(f"/api/datasets/{dataset}").json()["image_count"] == 7
    # non-image skipped, traversal sanitised
    r = client.post(f"/api/datasets/{dataset}/images",
                    files=[("files", ("../../../etc/passwd.jpg", make_jpeg(), "image/jpeg")),
                           ("files", ("readme.md", b"x", "text/plain"))])
    assert r.json()["saved"] == ["passwd.jpg"] and r.json()["skipped"] == ["readme.md"]


def test_full_pipeline_ordering(client, dataset):
    r = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "full"})
    assert r.status_code == 201, r.text
    job = r.json()
    assert [s["name"] for s in job["stages"]] == ["sfm", "mvs", "splat"]
    assert job["status"] == "queued"
    # mvs cannot start before sfm
    assert claim(client, ["mvs", "splat"]) is None
    c = claim(client, ["sfm"])
    assert c["stage"]["name"] == "sfm"
    assert c["paths"]["dataset_images"].endswith(f"datasets/{dataset}/images")
    assert c["job"]["params"]["sfm"]["matcher"] == "exhaustive"
    hb = client.post(f"/internal/stages/{c['stage']['id']}/heartbeat",
                     json={"worker_id": W, "progress": 0.5, "message": "matching"}).json()
    assert hb["cancel"] is False
    j = client.get(f"/api/jobs/{job['id']}").json()
    assert j["status"] == "running" and 0 < j["progress"] < 1
    finish(client, c["stage"]["id"])
    a = claim(client, ["mvs"], "w-mvs")
    b = claim(client, ["splat"], "w-splat")
    assert a["stage"]["name"] == "mvs" and b["stage"]["name"] == "splat"
    finish(client, a["stage"]["id"], wid="w-mvs")
    finish(client, b["stage"]["id"], wid="w-splat")
    j = client.get(f"/api/jobs/{job['id']}").json()
    assert j["status"] == "done" and j["progress"] == 1


def test_underwater_presets_build_stages(client, dataset):
    r = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "splat",
                                       "preset": "underwater"})
    job = r.json()
    assert [(s["name"], s["kind"]) for s in job["stages"]] == [
        ("preprocess", "cpu"), ("sfm", "sfm"), ("splat", "splat_uw")]
    assert job["params"]["splat"]["method"] == "water-splatting"
    assert job["params"]["underwater"]["enhance"] == "redcomp"

    r = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "mesh",
                                       "preset": "underwater-refractive-dome",
                                       "params": {"underwater": {"enhance": "none"}}})
    job = r.json()
    assert [(s["name"], s["kind"]) for s in job["stages"]] == [("sfm", "sfm_refrac"),
                                                              ("mvs", "mvs")]
    assert job["params"]["underwater"]["port"] == "dome"

    # invalid refraction params rejected
    r = client.post("/api/jobs", json={"dataset_id": dataset, "preset": "underwater-refractive-flat",
                                       "params": {"underwater": {"refrac_params": "1,2,3"}}})
    assert r.status_code == 422
    r = client.post("/api/jobs", json={"dataset_id": dataset, "params": {"sfm": {"bogus": 1}}})
    assert r.status_code == 422
    r = client.post("/api/jobs", json={"dataset_id": dataset, "preset": "nope"})
    assert r.status_code == 422


def test_failure_skips_dependents_and_retry(client, dataset):
    job = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "full"}).json()
    c = claim(client, ["sfm"])
    finish(client, c["stage"]["id"], "failed", "boom")
    j = client.get(f"/api/jobs/{job['id']}").json()
    assert {s["name"]: s["status"] for s in j["stages"]} == {
        "sfm": "failed", "mvs": "skipped", "splat": "skipped"}
    assert j["status"] == "failed"
    r = client.post(f"/api/jobs/{job['id']}/retry").json()
    assert all(s["status"] == "pending" for s in r["stages"])
    assert claim(client, ["sfm"])["stage"]["name"] == "sfm"


def test_cancel_running_and_pending(client, dataset):
    job = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "mesh"}).json()
    c = claim(client, ["sfm"])
    client.post(f"/api/jobs/{job['id']}/cancel")
    j = client.get(f"/api/jobs/{job['id']}").json()
    st = {s["name"]: s for s in j["stages"]}
    assert st["mvs"]["status"] == "cancelled" and st["sfm"]["cancel_requested"]
    hb = client.post(f"/internal/stages/{c['stage']['id']}/heartbeat",
                     json={"worker_id": W}).json()
    assert hb["cancel"] is True
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 409
    finish(client, c["stage"]["id"], "cancelled")
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200


def test_lease_expiry_and_worker_restart(client, app, dataset):
    job = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "mesh"}).json()
    c = claim(client, ["sfm"])
    db = app.state.db
    db._conn.execute("UPDATE stages SET heartbeat_at=? WHERE id=?",
                     (time.time() - 1000, c["stage"]["id"]))
    assert db.expire_stale(180) == 1
    j = client.get(f"/api/jobs/{job['id']}").json()
    assert j["stages"][0]["error"].startswith("worker lost")
    # late finish from the lost worker is rejected
    r = client.post(f"/internal/stages/{c['stage']['id']}/finish",
                    json={"worker_id": W, "status": "done"})
    assert r.json()["accepted"] is False

    client.post(f"/api/jobs/{job['id']}/retry")
    claim(client, ["sfm"], "wx")
    # same worker claims again => its previous running stage is marked failed
    assert claim(client, ["sfm"], "wx") is None
    j = client.get(f"/api/jobs/{job['id']}").json()
    assert j["stages"][0]["error"] == "worker restarted"


def test_logs_artifacts_archive(client, app, dataset):
    job = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "mesh"}).json()
    st = app.state.storage
    st.log_path(job["id"], "sfm").write_text("line1\nline2\n")
    r = client.get(f"/api/jobs/{job['id']}/logs/sfm", params={"offset": 0}).json()
    assert r["text"] == "line1\nline2\n" and r["next_offset"] == 12
    r = client.get(f"/api/jobs/{job['id']}/logs/sfm", params={"offset": 12}).json()
    assert r["text"] == ""
    out = st.outputs_dir(job["id"])
    (out / "mesh_textured").mkdir(parents=True)
    (out / "mesh.ply").write_bytes(b"ply\n" + b"0" * 5000)
    (out / "mesh_textured" / "scene_textured.obj").write_text("v 0 0 0\n")
    arts = client.get(f"/api/jobs/{job['id']}/artifacts").json()
    assert {a["path"] for a in arts} == {"mesh.ply", "mesh_textured/scene_textured.obj"}
    assert client.get(f"/api/jobs/{job['id']}/artifacts/mesh.ply").content.startswith(b"ply")
    assert client.get(f"/api/jobs/{job['id']}/artifacts/../../db.sqlite").status_code == 404
    z = client.get(f"/api/jobs/{job['id']}/archive")
    assert z.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert any(n.endswith("mesh_textured/scene_textured.obj") for n in names)


def test_auth(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from photogramm_api.main import Settings, create_app
    monkeypatch.setenv("PG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PG_API_TOKEN", "s3cret")
    with TestClient(create_app(Settings())) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/api/jobs").status_code == 401
        assert c.get("/api/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/api/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.post("/internal/claim", json={"worker_id": "x", "kinds": ["sfm"]}).status_code == 401


def test_delete_dataset_guard(client, dataset):
    job = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "mesh"}).json()
    assert client.delete(f"/api/datasets/{dataset}").status_code == 409
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert client.delete(f"/api/datasets/{dataset}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_modules_missing_and_info(client, app, dataset):
    db = app.state.db
    db._conn.execute("DELETE FROM workers")
    client.post("/internal/workers/heartbeat", json={"worker_id": "core", "kinds": ["cpu", "sfm"]})
    client.post("/internal/workers/heartbeat", json={"worker_id": "spl", "kinds": ["splat"]})
    mods = client.get("/api/info").json()["modules"]
    assert mods == {"core": True, "mesh": False, "splat": True, "water_splatting": False,
                    "refrac": False}
    # splatfacto-only job is fine
    assert client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "splat"}).status_code == 201
    # full job needs the mesh module
    r = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "full"})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "modules_missing" and d["missing_kinds"] == ["mvs"] and d["modules"] == ["mesh"]
    # underwater preset needs WaterSplatting
    r = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "splat", "preset": "underwater"})
    assert r.json()["detail"]["missing_kinds"] == ["splat_uw"]
    # ...unless explicitly queued anyway
    r = client.post("/api/jobs", json={"dataset_id": dataset, "pipeline": "full",
                                       "queue_if_unavailable": True})
    assert r.status_code == 201
    # stale workers do not count
    db._conn.execute("UPDATE workers SET last_seen=0")
    assert client.get("/api/info").json()["modules"]["core"] is False
