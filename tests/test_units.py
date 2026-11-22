import struct

import numpy as np
from PIL import Image


def test_enhancement_restores_red():
    from pgworker.stages.preprocess import enhance
    rng = np.random.default_rng(0)
    # bluish-green underwater-like image with weak red channel
    img = np.stack([rng.uniform(0.02, 0.1, (64, 64)), rng.uniform(0.4, 0.7, (64, 64)),
                    rng.uniform(0.5, 0.8, (64, 64))], axis=-1).astype(np.float32)
    for m in ("grayworld", "clahe", "redcomp", "udcp"):
        out = enhance(img, m)
        assert out.shape == img.shape and np.isfinite(out).all()
        assert 0 <= out.min() and out.max() <= 1
    r0 = img[..., 0].mean() / img[..., 1].mean()
    r1 = enhance(img, "redcomp")[..., 0].mean() / enhance(img, "redcomp")[..., 1].mean()
    assert r1 > 2 * r0


def test_process_one_keeps_exif_and_resizes(tmp_path):
    from pgworker.stages.preprocess import process_one
    src = tmp_path / "a.jpg"
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make
    Image.new("RGB", (400, 300), (20, 120, 160)).save(src, exif=exif.tobytes())
    r, e = tmp_path / "r.jpg", tmp_path / "e.jpg"
    process_one(str(src), str(r), str(e), 200, "redcomp")
    for p in (r, e):
        im = Image.open(p)
        assert im.size == (200, 150)
        assert im.getexif().get(0x010F) == "TestCam"


def test_colmap_option_resolver():
    from pgworker.colmap_utils import OptionResolver
    help_new = "  --FeatureExtraction.use_gpu arg\n  --SiftExtraction.max_num_features arg\n"
    help_old = "  --SiftExtraction.use_gpu arg (=1)\n  --SiftExtraction.use_gpu_extra arg\n"
    logs = []
    for txt, expect in ((help_new, "FeatureExtraction.use_gpu"), (help_old, "SiftExtraction.use_gpu")):
        R = OptionResolver(lambda cmd, t=txt: t, log=logs.append)
        args = R.args("feature_extractor", [
            (["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"], True),
            (["Missing.option"], 3), (["Skipped.none"], None)])
        assert args == [f"--{expect}", "1"]
    assert any("Missing.option" in l for l in logs)
    # no help available -> first candidate
    assert OptionResolver(lambda c: "").pick("x", ["A.b", "C.d"]) == "A.b"


def test_strip_refraction_and_model_stats(tmp_path):
    from pgworker.colmap_utils import largest_model, strip_refraction_from_cameras_txt
    cams = tmp_path / "cameras.txt"
    cams.write_text("# header\n1 OPENCV 100 80 50 50 50 40 0.1 0.01 0 0 DOMEPORT 0 0 0.005 0.05 0.005 1 1.49 1.334\n"
                    "2 PINHOLE 100 80 50 50 50 40\n")
    assert strip_refraction_from_cameras_txt(cams) == 1
    lines = cams.read_text().splitlines()
    assert lines[1].split() == "1 OPENCV 100 80 50 50 50 40 0.1 0.01 0 0".split()
    assert lines[2] == "2 PINHOLE 100 80 50 50 50 40"
    sp = tmp_path / "sparse"
    for name, n in (("0", 3), ("1", 10), ("2", 7)):
        (sp / name).mkdir(parents=True)
        (sp / name / "images.bin").write_bytes(struct.pack("<Q", n))
    best, st = largest_model(sp)
    assert best.name == "1" and st["images"] == 10


def test_write_splat_ply(tmp_path):
    from pgworker.export_splat import write_ply
    n = 5
    means = np.random.rand(n, 3)
    means[2, 0] = np.nan  # dropped
    p = tmp_path / "s.ply"
    cnt = write_ply(p, means, np.zeros((n, 3)), np.tile([1, 0, 0, 0], (n, 1)),
                    np.zeros((n, 1)), np.zeros((n, 3)), np.zeros((n, 15, 3)))
    assert cnt == 4
    raw = p.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    props = [l for l in head.decode().splitlines() if l.startswith("property")]
    assert len(props) == 3 + 3 + 3 + 45 + 1 + 3 + 4
    assert len(body) == cnt * len(props) * 4


def test_params_merge_and_stages():
    from photogramm_api.params import build_stages, resolve_params
    p = resolve_params("fast", {"splat": {"iterations": 500}})
    assert p.preprocess.max_image_size == 1600 and p.splat.iterations == 500
    assert [s["name"] for s in build_stages("splat", p)] == ["preprocess", "sfm", "splat"]
    p = resolve_params(None, {})
    assert [s["name"] for s in build_stages("mesh", p)] == ["sfm", "mvs"]
