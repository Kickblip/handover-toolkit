import argparse
import os
import pickle
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Load and print all .pkl files in a calibration folder."
    )
    parser.add_argument("calib_dir", type=str, help="Path to calib_(DATE) folder")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw repr() for non-numpy objects (otherwise pretty-print).",
    )
    args = parser.parse_args()

    base_dir = args.calib_dir
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' is not a valid directory.")
        sys.exit(1)

    pkl_files = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name.lower().endswith(".pkl"):
                pkl_files.append(os.path.join(root, name))

    if not pkl_files:
        print("No .pkl files found.")
        return

    for path in sorted(pkl_files):
        print("=" * 80)
        print(path)

        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"  ERROR loading pickle: {e}")
            continue

        if isinstance(obj, np.ndarray):
            print(f"  type: numpy.ndarray  dtype: {obj.dtype}  shape: {obj.shape}")
            with np.printoptions(suppress=True, precision=6):
                print(obj)
        else:
            print(f"  type: {type(obj)}")
            if args.raw:
                print(repr(obj))
            else:
                print(obj)

    print("=" * 80)


if __name__ == "__main__":
    main()