import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pyk4a import PyK4APlayback


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def decode_color_frame(color) -> np.ndarray:
    if color is None:
        raise ValueError("color frame is None")

    if isinstance(color, np.ndarray) and color.ndim == 1:
        buf = color
        if buf.dtype != np.uint8:
            buf = buf.astype(np.uint8, copy=False)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed (buffer not a valid JPEG/MJPG frame?)")
        return img

    if isinstance(color, np.ndarray) and color.ndim == 3:
        if color.shape[2] == 4:
            return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        if color.shape[2] == 3:
            return color

    raise ValueError(f"Unexpected color frame shape/type: {getattr(color, 'shape', type(color))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Split Azure Kinect MKV into JPG frames using pyk4a.")
    ap.add_argument("mkv", type=str, help="Path to input .mkv recorded by k4arecorder")
    args = ap.parse_args()

    mkv_path = Path(args.mkv).expanduser().resolve()
    if not mkv_path.exists():
        die(f"Input MKV not found: {mkv_path}")

    out_dir = Path(mkv_path.stem).resolve()
    ensure_dir(out_dir)

    pb = PyK4APlayback(str(mkv_path))
    try:
        pb.open()
    except Exception as e:
        die(f"Failed to open MKV with PyK4APlayback: {e}")

    idx = 0
    written = 0
    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    try:
        while True:
            try:
                cap = pb.get_next_capture()
            except EOFError:
                break
            except Exception as e:
                die(f"Playback read error at frame {idx}: {e}")

            if cap is None:
                break

            try:
                bgr = decode_color_frame(cap.color)
            except Exception as e:
                die(f"Frame decode failed at index {idx}: {e}")

            out_path = out_dir / f"frame_{written:06d}.jpg"
            ok = cv2.imwrite(str(out_path), bgr, jpg_params)
            if not ok:
                die(f"Failed to write JPG: {out_path}")

            written += 1
            idx += 1

    finally:
        try:
            pb.close()
        except Exception:
            pass

    print(f"Wrote {written} frames to: {out_dir}")


if __name__ == "__main__":
    main()