"""Minimal reader for ASCII PCD files as written by pcl::io::savePCDFileASCII.

Supports PointXYZ (FIELDS x y z) and PointXYZRGB (FIELDS x y z rgb, where rgb
is a packed float32). Returns (points, colors) with points float32 (N, 3) and
colors uint8 (N, 3) or None.
"""

import numpy as np


def read_pcd_ascii(path):
    fields = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF inside PCD header: %s" % path)
            line = line.decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            key = key.upper()
            vals = rest.split()
            if key == "FIELDS":
                fields = vals
            elif key == "DATA":
                if not vals or vals[0].lower() != "ascii":
                    raise ValueError("Only ascii PCD files are supported: %s" % path)
                break
        data = np.loadtxt(f, dtype=np.float64, ndmin=2)

    if data.size == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    points = np.stack(
        [data[:, fields.index(axis)] for axis in ("x", "y", "z")], axis=1
    ).astype(np.float32)

    colors = None
    if "rgb" in fields:
        packed = data[:, fields.index("rgb")].astype(np.float32).view(np.uint32)
        colors = np.stack(
            [(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF], axis=1
        ).astype(np.uint8)

    return points, colors
