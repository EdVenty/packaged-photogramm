"""Fallback exporter: nerfstudio checkpoint -> 3DGS (INRIA-format) PLY.

Used when ``ns-export gaussian-splat`` rejects a model (e.g. third-party methods such as
WaterSplatting). Runs inside the splat image (needs torch + nerfstudio).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def _get(model, name):
    if hasattr(model, name):
        try:
            return getattr(model, name)
        except Exception:  # noqa: BLE001
            pass
    gp = getattr(model, "gauss_params", None)
    if gp is not None and name in gp:
        return gp[name]
    return None


def _np(t):
    return t.detach().float().cpu().numpy() if t is not None else None


def write_ply(path: Path, means, scales, quats, opacities, f_dc, f_rest) -> int:
    n = means.shape[0]
    f_rest = f_rest.reshape(n, -1) if f_rest is not None and f_rest.size else np.zeros((n, 0))
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([
        means, np.zeros((n, 3)), f_dc.reshape(n, 3), f_rest,
        opacities.reshape(n, 1), scales.reshape(n, 3), quats.reshape(n, 4),
    ], axis=1).astype(np.float32)
    ok = np.isfinite(data).all(axis=1)
    data = data[ok]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {data.shape[0]}"]
    header += [f"property float {f}" for f in fields]
    header += ["end_header"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())
    return int(data.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(Path(a.config), test_mode="inference")
    model = pipeline.model
    with torch.no_grad():
        means = _np(_get(model, "means"))
        scales = _np(_get(model, "scales"))
        quats = _np(_get(model, "quats"))
        opac = _np(_get(model, "opacities"))
        f_dc = _np(_get(model, "features_dc"))
        f_rest = _get(model, "features_rest")
        if f_rest is not None:
            # (N, K, 3) -> channel-major (N, 3*K) as in the reference 3DGS exporter
            f_rest = _np(f_rest.transpose(1, 2).contiguous())
        if f_dc is None:
            colors = _np(_get(model, "colors"))
            if colors is None:
                raise SystemExit("model has neither features_dc nor colors")
            f_dc = (np.clip(colors, 0, 1) - 0.5) / SH_C0
    if any(x is None for x in (means, scales, quats, opac)):
        raise SystemExit(f"unsupported model type {type(model).__name__}: missing gaussian params")
    n = write_ply(Path(a.output), means, scales, quats, opac, f_dc, f_rest)
    print(f"wrote {n} gaussians to {a.output}")


if __name__ == "__main__":
    main()
