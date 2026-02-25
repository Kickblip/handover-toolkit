import argparse
import glob
import os
import pickle
import re
from itertools import combinations

import numpy as np
import open3d as o3d

CAM_ID_RE = re.compile(r"camera_(\d+)\.pkl$", re.IGNORECASE)


def load_extrinsics(extr_dir: str):
    T_by_id = {}
    for p in sorted(glob.glob(os.path.join(extr_dir, "*.pkl"))):
        m = CAM_ID_RE.search(os.path.basename(p))
        if not m:
            continue
        cam_id = int(m.group(1))
        with open(p, "rb") as f:
            T = np.asarray(pickle.load(f), dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"{p} expected 4x4, got {T.shape}")
        T_by_id[cam_id] = T
    return T_by_id


def make_frustum(T: np.ndarray, size: float = 0.15, aspect: float = 4 / 3):
    z = size
    w = size * aspect
    h = size

    pts_cam = np.array(
        [[0, 0, 0], [-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]],
        dtype=np.float64,
    )
    pts_h = np.c_[pts_cam, np.ones((5, 1))]
    pts_w = (T @ pts_h.T).T[:, :3]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_w),
        lines=o3d.utility.Vector2iVector(lines),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib_dir", help="Path to calib_(DATE) folder")
    args = ap.parse_args()

    extr_dir = os.path.join(args.calib_dir, "cam_extr")
    if not os.path.isdir(extr_dir):
        raise SystemExit(f"Missing extrinsics folder: {extr_dir}")

    T_by_id = load_extrinsics(extr_dir)
    if not T_by_id:
        raise SystemExit(f"No camera_*.pkl found in {extr_dir}")

    if 0 in T_by_id and not np.allclose(T_by_id[0], np.eye(4), atol=1e-6):
        print("Warning: camera_0 is not identity. Distances are still computed in camera_0 frame.")

    centers = {cid: T[:3, 3].copy() for cid, T in T_by_id.items()}

    print("\nPairwise distances (meters):")
    for a, b in combinations(sorted(centers.keys()), 2):
        d = float(np.linalg.norm(centers[a] - centers[b]))
        print(f"  camera_{a} <-> camera_{b}: {d:.6f} m")

    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)]
    for cid in sorted(T_by_id.keys()):
        geoms.append(make_frustum(T_by_id[cid]))

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Camera extrinsics (relative to camera_0) + distances in console",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()