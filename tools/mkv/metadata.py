import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pyk4a import (
    PyK4APlayback,
    CalibrationType,
    ImageFormat,
    ColorResolution,
    DepthMode,
    FPS,
    WiredSyncMode,
)

# Known Azure Kinect sizes (SDK)
COLOR_RESOLUTION_SIZES = {
    ColorResolution.RES_720P: (1280, 720),
    ColorResolution.RES_1080P: (1920, 1080),
    ColorResolution.RES_1440P: (2560, 1440),
    ColorResolution.RES_1536P: (2048, 1536),
    ColorResolution.RES_2160P: (3840, 2160),
    ColorResolution.RES_3072P: (4096, 3072),
    ColorResolution.OFF: None,
}
DEPTH_MODE_SIZES = {
    DepthMode.NFOV_UNBINNED: (640, 576),
    DepthMode.NFOV_2X2BINNED: (320, 288),
    DepthMode.WFOV_UNBINNED: (1024, 1024),
    DepthMode.WFOV_2X2BINNED: (512, 512),
    DepthMode.PASSIVE_IR: (1024, 1024),
    DepthMode.OFF: None,
}

ENUM_FIELDS = {
    "color_format": ImageFormat,
    "color_resolution": ColorResolution,
    "depth_mode": DepthMode,
    "camera_fps": FPS,
    "wired_sync_mode": WiredSyncMode,
}


def jdefault(o):
    """json.dumps/json.dump fallback for non-serializable objects."""
    if hasattr(o, "name") and hasattr(o, "value"):  # Enum-like
        return o.name
    if hasattr(o, "tolist"):  # numpy arrays
        return o.tolist()
    return str(o)


def enum_to_name(enum_cls, v):
    if v is None:
        return None
    if hasattr(v, "name") and hasattr(v, "value"):
        return v.name
    try:
        return enum_cls(int(v)).name
    except Exception:
        return str(v)


def decode_configuration(cfg):
    if cfg is None:
        return None
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    out = dict(cfg)
    for k, enum_cls in ENUM_FIELDS.items():
        if k in out:
            out[k] = enum_to_name(enum_cls, out[k])
    return out


def infer_sizes_from_config(cfg):
    if not isinstance(cfg, dict):
        return None, None
    cr = ColorResolution.__members__.get(cfg.get("color_resolution", "") or "", None)
    dm = DepthMode.__members__.get(cfg.get("depth_mode", "") or "", None)
    return COLOR_RESOLUTION_SIZES.get(cr), DEPTH_MODE_SIZES.get(dm)


def get_image_size(calib, ct):
    # Prefer API if present
    if hasattr(calib, "get_image_size"):
        try:
            return calib.get_image_size(ct)
        except Exception:
            pass

    # Fallback to calibration object fields
    attr = {CalibrationType.COLOR: "color_calibration", CalibrationType.DEPTH: "depth_calibration"}.get(ct)
    cam = getattr(calib, attr, None)
    w, h = getattr(cam, "resolution_width", None), getattr(cam, "resolution_height", None)
    return (int(w), int(h)) if w is not None and h is not None else None


def try_call(fn):
    try:
        return fn()
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="Dump Azure Kinect MKV metadata (pyk4a) to JSON.")
    ap.add_argument("mkv", help="Path to .mkv file")
    ap.add_argument("-o", "--out", default=None, help="Output .json path (default: <mkv_basename>.json)")
    ap.add_argument("--no-stdout", action="store_true", help="Don't print JSON; only write file.")
    args = ap.parse_args()

    mkv_path = Path(args.mkv)
    out_path = Path(args.out) if args.out else mkv_path.with_suffix(".json")

    pb = PyK4APlayback(str(mkv_path))
    pb.open()

    cfg = decode_configuration(getattr(pb, "configuration", None))
    cfg_color_size, cfg_depth_size = infer_sizes_from_config(cfg)

    data = {
        "file": str(mkv_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": cfg,
        "tracks": {k: getattr(pb, k) for k in ("has_color_track", "has_depth_track", "has_ir_track", "has_imu_track") if hasattr(pb, k)},
        "last_timestamp_usec": try_call(pb.get_last_timestamp_usec) if hasattr(pb, "get_last_timestamp_usec") else None,
        "calibration": {},
    }

    calib = getattr(pb, "calibration", None)
    if calib:
        for label, ct, fallback in (
            ("color", CalibrationType.COLOR, cfg_color_size),
            ("depth", CalibrationType.DEPTH, cfg_depth_size),
        ):
            size = get_image_size(calib, ct) or fallback or "(unknown)"
            data["calibration"][label] = {
                "image_size": size,
                "camera_matrix": try_call(lambda: calib.get_camera_matrix(ct)),
                "distortion": try_call(lambda: calib.get_distortion_coefficients(ct)),
            }

        if hasattr(calib, "get_extrinsics"):
            data["calibration"]["extrinsics_depth_to_color"] = try_call(
                lambda: calib.get_extrinsics(CalibrationType.DEPTH, CalibrationType.COLOR)
            )
            data["calibration"]["extrinsics_color_to_depth"] = try_call(
                lambda: calib.get_extrinsics(CalibrationType.COLOR, CalibrationType.DEPTH)
            )

    pb.close()

    out_path.write_text(json.dumps(data, indent=2, default=jdefault), encoding="utf-8")

    if not args.no_stdout:
        print(f"wrote_json: {out_path}")
        print(json.dumps(data, indent=2, default=jdefault))


if __name__ == "__main__":
    main()
