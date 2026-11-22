"""Job parameter models, presets and stage-graph construction."""
from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CAMERA_MODELS = (
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV",
    "FULL_OPENCV", "OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreprocessParams(_Strict):
    max_image_size: int = Field(0, ge=0, description="Downscale longest side; 0 = keep")


class SfmParams(_Strict):
    camera_model: Literal[CAMERA_MODELS] = "SIMPLE_RADIAL"  # type: ignore[valid-type]
    single_camera: bool = True
    camera_params: str = ""
    matcher: Literal["exhaustive", "sequential", "spatial"] = "exhaustive"
    sequential_overlap: int = Field(10, ge=1)
    mapper: Literal["incremental", "hierarchical"] = "incremental"
    max_image_size: int = Field(3200, ge=0)
    max_num_features: int = Field(8192, ge=256)
    robust_features: bool = False
    guided_matching: bool = False
    undistort_max_image_size: int = Field(3200, ge=0)
    use_gpu: bool = True


class MvsParams(_Strict):
    resolution_level: int = Field(1, ge=0, le=6)
    max_resolution: int = Field(2560, ge=0)
    number_views_fuse: int = Field(3, ge=1)
    refine: bool = False
    decimate: float = Field(1.0, gt=0, le=1)
    texture: bool = True
    texture_resolution_level: int = Field(0, ge=0, le=6)


class SplatParams(_Strict):
    method: Literal["splatfacto", "splatfacto-big", "water-splatting"] = "splatfacto"
    iterations: int = Field(30000, ge=100)
    downscale_factor: int = Field(1, ge=0, le=8, description="0 = auto")
    extra_args: str = ""


class UnderwaterParams(_Strict):
    enabled: bool = False
    enhance: Literal["none", "grayworld", "clahe", "redcomp", "udcp"] = "none"
    enhance_for: Literal["sfm", "all"] = "sfm"
    camera: Literal["pinhole", "refractive"] = "pinhole"
    port: Literal["flat", "dome"] = "flat"
    refrac_params: str = ""
    refine_refrac_params: bool = False

    @model_validator(mode="after")
    def _check(self) -> "UnderwaterParams":
        if self.enabled and self.camera == "refractive":
            vals = [v for v in self.refrac_params.replace(" ", "").split(",") if v]
            if len(vals) != 8:
                raise ValueError(
                    "underwater.refrac_params must contain 8 comma-separated numbers "
                    "(FLATPORT: nx,ny,nz,d_int,thickness,n_air,n_glass,n_water; "
                    "DOMEPORT: cx,cy,cz,radius,thickness,n_air,n_glass,n_water)"
                )
            try:
                [float(v) for v in vals]
            except ValueError as e:
                raise ValueError(f"underwater.refrac_params: {e}") from e
        return self


class JobParams(_Strict):
    preprocess: PreprocessParams = PreprocessParams()
    sfm: SfmParams = SfmParams()
    mvs: MvsParams = MvsParams()
    splat: SplatParams = SplatParams()
    underwater: UnderwaterParams = UnderwaterParams()


Pipeline = Literal["mesh", "splat", "full"]


class JobCreate(BaseModel):
    dataset_id: str
    name: str = ""
    pipeline: Pipeline = "full"
    preset: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    # Queue the job even if no online worker can run one of its stages (module not installed
    # or not started yet). Otherwise the API answers 409 "modules_missing".
    queue_if_unavailable: bool = False


# Stage kind -> installable module (docker compose profile) that provides a worker for it.
KIND_MODULE = {
    "cpu": "core",
    "sfm": "core",
    "sfm_refrac": "refrac",
    "mvs": "mesh",
    "splat": "splat",
    "splat_uw": "splat (+WaterSplatting)",
}


PRESETS: dict[str, dict[str, Any]] = {
    "standard": {
        "_description": "Обычная съёмка: SIMPLE_RADIAL, exhaustive matching, splatfacto",
    },
    "fast": {
        "_description": "Быстрое превью: ресайз до 1600px, грубый MVS, 7000 итераций сплатов",
        "preprocess": {"max_image_size": 1600},
        "sfm": {"max_num_features": 4096, "undistort_max_image_size": 1600},
        "mvs": {"resolution_level": 2, "max_resolution": 1600},
        "splat": {"iterations": 7000},
    },
    "underwater": {
        "_description": "Подводная съёмка (UW-pinhole): OPENCV-модель, цветокоррекция, "
                        "устойчивые признаки, WaterSplatting",
        "sfm": {"camera_model": "OPENCV", "robust_features": True, "guided_matching": True,
                "max_num_features": 16384},
        "mvs": {"resolution_level": 1},
        "splat": {"method": "water-splatting", "iterations": 15000},
        "underwater": {"enabled": True, "enhance": "redcomp", "enhance_for": "sfm",
                       "camera": "pinhole"},
    },
    "underwater-refractive-flat": {
        "_description": "Подводная съёмка, Refractive COLMAP, плоский иллюминатор (flat port). "
                        "Нужны refrac_params и желательно in-air camera_params",
        "sfm": {"camera_model": "OPENCV", "robust_features": True, "guided_matching": True,
                "max_num_features": 16384},
        "splat": {"method": "water-splatting", "iterations": 15000},
        "underwater": {"enabled": True, "enhance": "redcomp", "enhance_for": "sfm",
                       "camera": "refractive", "port": "flat",
                       "refrac_params": "0,0,1,0.01,0.01,1.0,1.49,1.334"},
    },
    "underwater-refractive-dome": {
        "_description": "Подводная съёмка, Refractive COLMAP, купольный иллюминатор (dome port)",
        "sfm": {"camera_model": "OPENCV", "robust_features": True, "guided_matching": True,
                "max_num_features": 16384},
        "splat": {"method": "water-splatting", "iterations": 15000},
        "underwater": {"enabled": True, "enhance": "redcomp", "enhance_for": "sfm",
                       "camera": "refractive", "port": "dome",
                       "refrac_params": "0,0,0.005,0.05,0.005,1.0,1.49,1.334"},
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_params(preset: str | None, overrides: dict[str, Any]) -> JobParams:
    base = JobParams().model_dump()
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset '{preset}', available: {', '.join(PRESETS)}")
        p = {k: v for k, v in PRESETS[preset].items() if not k.startswith("_")}
        base = deep_merge(base, p)
    merged = deep_merge(base, overrides or {})
    return JobParams.model_validate(merged)


def build_stages(pipeline: str, params: JobParams) -> list[dict[str, Any]]:
    """Return an ordered list of stage specs: {name, kind, deps}."""
    stages: list[dict[str, Any]] = []
    uw = params.underwater
    need_pre = params.preprocess.max_image_size > 0 or (uw.enabled and uw.enhance != "none")
    if need_pre:
        stages.append({"name": "preprocess", "kind": "cpu", "deps": []})
    sfm_kind = "sfm_refrac" if (uw.enabled and uw.camera == "refractive") else "sfm"
    stages.append({"name": "sfm", "kind": sfm_kind,
                   "deps": ["preprocess"] if need_pre else []})
    if pipeline in ("mesh", "full"):
        stages.append({"name": "mvs", "kind": "mvs", "deps": ["sfm"]})
    if pipeline in ("splat", "full"):
        kind = "splat_uw" if params.splat.method == "water-splatting" else "splat"
        stages.append({"name": "splat", "kind": kind, "deps": ["sfm"]})
    return stages


def presets_public() -> dict[str, Any]:
    return {
        name: {"description": p.get("_description", ""),
               "params": {k: v for k, v in p.items() if not k.startswith("_")}}
        for name, p in PRESETS.items()
    }
