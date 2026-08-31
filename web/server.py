#!/usr/bin/env python3
"""FAST-Calib web visualization & interaction platform (viser).

Runs the ROS `fast_calib` single-scene calibration node as a subprocess and
visualizes inputs, intermediate pipeline stages, and results in the browser.

Environment variables:
  CALIB_DATA_DIR             directory scanned for calibration scenes (default /data)
  CALIB_OUTPUT_DIR           directory the calibrator writes outputs to (default /output)
  FAST_CALIB_DEFAULT_PARAMS  YAML with default GUI parameter values
  FAST_CALIB_RUN_TIMEOUT     seconds to wait for the node to produce results (default 180)
  VISER_PORT                 web port (default 8080)
"""

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import viser

from pcd import read_pcd_ascii

DATA_DIR = Path(os.environ.get("CALIB_DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("CALIB_OUTPUT_DIR", "/output"))
DEFAULT_PARAMS_PATH = Path(
    os.environ.get("FAST_CALIB_DEFAULT_PARAMS", "/opt/fast_calib/config/qr_params.yaml")
)
RUN_TIMEOUT = int(os.environ.get("FAST_CALIB_RUN_TIMEOUT", "180"))
VISER_PORT = int(os.environ.get("VISER_PORT", "8080"))

OUTPUT_FILES = [
    "single_calib_result.txt",
    "colored_cloud.pcd",
    "qr_detect.png",
    "input_cloud.pcd",
    "filtered_cloud.pcd",
    "plane_cloud.pcd",
    "aligned_cloud.pcd",
    "edge_cloud.pcd",
    "lidar_centers.pcd",
    "qr_centers.pcd",
    "fast_calib_node.log",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_default_params():
    params = {}
    if DEFAULT_PARAMS_PATH.exists():
        with open(DEFAULT_PARAMS_PATH) as f:
            params = yaml.safe_load(f) or {}
    return params


def find_scenes():
    """Return {label: bag_path} for every rosbag under DATA_DIR."""
    scenes = {}
    if not DATA_DIR.exists():
        return scenes
    for bag in sorted(DATA_DIR.rglob("*.bag")):
        label = str(bag.relative_to(DATA_DIR))
        scenes[label] = bag
    return scenes


def find_image_for_bag(bag_path):
    """Pick an image file sitting next to the bag, if any (else "")."""
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images = sorted(bag_path.parent.glob(ext))
        if images:
            return str(images[0])
    return ""


def mat_to_wxyz(R):
    """Rotation matrix (3x3) -> quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def parse_result_txt(path):
    """Parse Rcl/Pcl from single_calib_result.txt -> 4x4 T_cam_lidar."""
    text = Path(path).read_text()
    rcl = re.search(r"Rcl:\s*\[([^\]]+)\]", text, re.S)
    pcl = re.search(r"Pcl:\s*\[([^\]]+)\]", text, re.S)
    if not rcl or not pcl:
        return None
    nums = [float(v) for v in re.split(r"[,\s]+", rcl.group(1).strip()) if v]
    trans = [float(v) for v in re.split(r"[,\s]+", pcl.group(1).strip()) if v]
    if len(nums) != 9 or len(trans) != 3:
        return None
    T = np.eye(4)
    T[:3, :3] = np.array(nums).reshape(3, 3)
    T[:3, 3] = trans
    return T


def parse_rmse(log_text):
    m = re.search(r"RMSE:\s*.*?([0-9]+\.[0-9]+)\s*m", log_text)
    return float(m.group(1)) if m else None


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def subsample(points, colors=None, max_points=250_000):
    if len(points) <= max_points:
        return points, colors
    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx], colors[idx] if colors is not None else None


def build_depth_image(points_cam, params, width, height):
    """Project camera-frame points into the (distorted) image plane and render
    a z-buffered, jet-colorized depth map. Returns (rgb uint8 HxWx3, mask bool)."""
    z = points_cam[:, 2]
    valid = z > 0.1
    pts, z = points_cam[valid], z[valid]

    x = pts[:, 0] / z
    y = pts[:, 1] / z
    r2 = x * x + y * y
    radial = 1.0 + params["k1"] * r2 + params["k2"] * r2 * r2
    xd = x * radial + 2.0 * params["p1"] * x * y + params["p2"] * (r2 + 2.0 * x * x)
    yd = y * radial + params["p1"] * (r2 + 2.0 * y * y) + 2.0 * params["p2"] * x * y
    u = params["fx"] * xd + params["cx"]
    v = params["fy"] * yd + params["cy"]

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ui = u[inside].astype(np.int64)
    vi = v[inside].astype(np.int64)
    zi = z[inside]

    depth = np.full((height, width), np.inf, dtype=np.float64)
    np.minimum.at(depth, (vi, ui), zi)
    mask = np.isfinite(depth)
    if not mask.any():
        return np.zeros((height, width, 3), dtype=np.uint8), mask

    # Contrast-stretch with percentiles, then apply a jet-like colormap.
    lo, hi = np.percentile(depth[mask], (2.0, 98.0))
    t = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    rgb[~mask] = 0
    return rgb, mask


# ---------------------------------------------------------------------------
# Web application
# ---------------------------------------------------------------------------


class FastCalibApp:
    def __init__(self):
        self.server = viser.ViserServer(host="0.0.0.0", port=VISER_PORT)
        try:
            self.server.scene.set_up_direction("+z")
        except Exception:
            pass
        self.defaults = load_default_params()
        self.scenes = find_scenes()
        self.run_lock = threading.Lock()
        self.last_result_path = None

        self._build_gui()

    # ------------------------------------------------------------------ GUI

    def _build_gui(self):
        gui = self.server.gui
        d = self.defaults

        gui.add_markdown("## FAST-Calib\nLiDAR–camera extrinsic calibration")

        # Depth-map panel pinned to the top of the left GUI column (i.e. the
        # top-left corner of the page); populated after each calibration run.
        self.depth_panel = gui.add_image(
            np.zeros((2, 2, 3), dtype=np.uint8),
            label="Reprojected LiDAR depth (camera view)",
            format="jpeg",
            order=-100,
            visible=False,
        )

        with gui.add_folder("Scene"):
            options = list(self.scenes.keys()) or ["(no scenes found under %s)" % DATA_DIR]
            self.scene_dropdown = gui.add_dropdown("Rosbag", options)
            self.image_text = gui.add_text(
                "Image path (blank = extract from bag)",
                initial_value="",
                hint="Optional image file next to the bag; leave blank to pull a frame from camera_topic.",
            )
            self.lidar_topic = gui.add_text(
                "LiDAR topic", initial_value=str(d.get("lidar_topic", "/livox/lidar"))
            )
            self.camera_topic = gui.add_text(
                "Camera topic", initial_value=str(d.get("camera_topic", "/camera/image_raw"))
            )

        with gui.add_folder("Camera intrinsics", expand_by_default=False):
            self.intrinsic_inputs = {}
            for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"):
                self.intrinsic_inputs[key] = gui.add_number(
                    key, initial_value=float(d.get(key, 0.0)), step=0.001
                )

        with gui.add_folder("Target geometry", expand_by_default=False):
            self.target_inputs = {}
            for key in (
                "marker_size",
                "delta_width_qr_center",
                "delta_height_qr_center",
                "delta_width_circles",
                "delta_height_circles",
                "circle_radius",
            ):
                self.target_inputs[key] = gui.add_number(
                    key, initial_value=float(d.get(key, 0.0)), step=0.005
                )

        with gui.add_folder("Distance filter (m)"):
            self.filter_inputs = {}
            for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
                self.filter_inputs[key] = gui.add_number(
                    key, initial_value=float(d.get(key, 0.0)), step=0.1
                )

        self.run_button = gui.add_button(
            "Run calibration",
            color="green",
            disabled=not self.scenes,
            hint=None if self.scenes else "Mount scene data (rosbag + image) into %s" % DATA_DIR,
        )
        self.status_md = gui.add_markdown("**Status:** idle")

        with gui.add_folder("Results"):
            self.result_md = gui.add_markdown("_No calibration run yet._")
            self.download_button = gui.add_button("Download result", disabled=True)

        with gui.add_folder("Layers", expand_by_default=False):
            self.layer_checkboxes = {}
            for key, label, default in (
                ("input", "Input cloud", True),
                ("filtered", "Filtered cloud", False),
                ("plane", "Plane (RANSAC)", False),
                ("edge", "Edge points", False),
                ("centers", "Circle centers", True),
                ("colored", "Colored cloud (result)", True),
                ("image", "Camera image", True),
                ("depth_inset", "Depth image panel (top-left)", True),
            ):
                self.layer_checkboxes[key] = gui.add_checkbox(label, initial_value=default)
                self.layer_checkboxes[key].on_update(self._on_layer_toggle)

        self.run_button.on_click(self._on_run_clicked)
        self.download_button.on_click(self._on_download)

    # ------------------------------------------------------------- callbacks

    def _on_layer_toggle(self, event):
        key = next(k for k, cb in self.layer_checkboxes.items() if cb is event.target)
        if key == "depth_inset":
            # Toggle the depth-map panel in the top-left corner of the page.
            self.depth_panel.visible = event.target.value
            return
        # Layer groups are frames named /layers/<key>; toggling the frame
        # visibility hides the whole subtree.
        frame = getattr(self, "_layer_frames", {}).get(key)
        if frame is not None:
            frame.visible = event.target.value

    def _on_download(self, event):
        if event.client is None or self.last_result_path is None:
            return
        path = Path(self.last_result_path)
        if path.exists():
            event.client.send_file_download(path.name, path.read_bytes())

    def _on_run_clicked(self, _):
        if self.run_lock.locked():
            return
        threading.Thread(target=self._run_calibration, daemon=True).start()

    # ----------------------------------------------------------- calibration

    def _collect_params(self):
        bag = self.scenes[self.scene_dropdown.value]
        params = {}
        params.update({k: float(v.value) for k, v in self.intrinsic_inputs.items()})
        params.update({k: float(v.value) for k, v in self.target_inputs.items()})
        params.update({k: float(v.value) for k, v in self.filter_inputs.items()})
        params["lidar_topic"] = self.lidar_topic.value
        params["camera_topic"] = self.camera_topic.value
        params["min_detected_markers"] = 3
        params["save_intermediate"] = True
        params["bag_path"] = str(bag)
        params["image_path"] = self.image_text.value.strip() or find_image_for_bag(bag)
        params["output_path"] = str(OUTPUT_DIR)
        self._last_params = params
        return params

    def _run_calibration(self):
        with self.run_lock:
            self.run_button.disabled = True
            try:
                self._run_calibration_impl()
            except Exception as exc:  # never leave the UI stuck
                self.status_md.content = "**Status:** error — `%s`" % exc
            finally:
                self.run_button.disabled = False

    def _run_calibration_impl(self):
        # Clean previous outputs so we can detect fresh results.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_FILES:
            try:
                (OUTPUT_DIR / name).unlink()
            except FileNotFoundError:
                pass

        params = self._collect_params()
        yaml_path = Path("/tmp/fast_calib_run.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(params, f)

        self.status_md.content = "**Status:** loading parameters…"
        subprocess.run(["rosparam", "load", str(yaml_path)], check=True)

        log_path = OUTPUT_DIR / "fast_calib_node.log"
        self.status_md.content = "**Status:** running fast_calib node…"
        logf = open(log_path, "w")
        proc = subprocess.Popen(
            ["rosrun", "fast_calib", "fast_calib"],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

        result_path = OUTPUT_DIR / "single_calib_result.txt"
        deadline = time.time() + RUN_TIMEOUT
        finished = False
        start = time.time()
        try:
            while time.time() < deadline:
                if result_path.exists() and (OUTPUT_DIR / "colored_cloud.pcd").exists():
                    finished = True
                    break
                if proc.poll() is not None:
                    break
                elapsed = int(time.time() - start)
                self.status_md.content = "**Status:** running fast_calib node… (%ds)" % elapsed
                time.sleep(1.0)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            logf.close()

        log_text = strip_ansi(log_path.read_text(errors="replace")) if log_path.exists() else ""

        if not finished:
            tail = "\n".join(log_text.strip().splitlines()[-15:])
            self.status_md.content = "**Status:** failed (timeout or node exited)"
            self.result_md.content = "**Node log (tail):**\n```\n%s\n```" % tail
            return

        rmse = parse_rmse(log_text)
        T = parse_result_txt(result_path)
        self.last_result_path = str(result_path)
        self.download_button.disabled = False
        self.status_md.content = "**Status:** done in %ds" % (time.time() - start)

        lines = []
        if rmse is not None:
            lines.append("**RMSE:** `%.4f m`" % rmse)
        if T is not None:
            lines.append("**T_cam_lidar:**")
            lines.append("```")
            for row in T:
                lines.append(" ".join("% .6f" % v for v in row))
            lines.append("```")
        self.result_md.content = "\n".join(lines) if lines else "_Result file written._"

        self._update_scene(T)

    # ------------------------------------------------------------ scene draw

    def _update_scene(self, T_cam_lidar):
        scene = self.server.scene
        self._layer_frames = {}

        def layer_frame(key):
            frame = scene.add_frame(
                "/layers/%s" % key,
                show_axes=False,
                visible=self.layer_checkboxes[key].value,
            )
            self._layer_frames[key] = frame
            return frame

        def add_cloud(key, filename, color, point_size, transform=None):
            path = OUTPUT_DIR / filename
            if not path.exists():
                return
            points, colors = read_pcd_ascii(path)
            if len(points) == 0:
                return
            if transform is not None:
                pts = np.concatenate(
                    [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
                )
                points = (transform @ pts.T).T[:, :3].astype(np.float32)
            points, colors = subsample(points, colors)
            if colors is None:
                colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
            layer_frame(key)
            scene.add_point_cloud(
                "/layers/%s/cloud" % key,
                points=points,
                colors=colors,
                point_size=point_size,
                point_shape="circle",
                precision="float32",
            )

        # LiDAR-frame layers
        add_cloud("input", "input_cloud.pcd", (140, 140, 140), 0.01)
        add_cloud("filtered", "filtered_cloud.pcd", (76, 154, 255), 0.01)
        add_cloud("plane", "plane_cloud.pcd", (54, 179, 126), 0.01)
        add_cloud("edge", "edge_cloud.pcd", (255, 86, 48), 0.02)
        add_cloud(
            "colored",
            "colored_cloud.pcd",
            (255, 255, 255),
            0.01,
            transform=np.linalg.inv(T_cam_lidar) if T_cam_lidar is not None else None,
        )

        # Circle centers: lidar centers (lidar frame) + qr centers (camera frame
        # -> transformed into lidar frame so both can be compared visually).
        layer_frame("centers")
        lidar_centers_path = OUTPUT_DIR / "lidar_centers.pcd"
        if lidar_centers_path.exists():
            pts, _ = read_pcd_ascii(lidar_centers_path)
            if len(pts):
                scene.add_point_cloud(
                    "/layers/centers/lidar",
                    points=pts,
                    colors=np.tile(np.array((255, 0, 0), dtype=np.uint8), (len(pts), 1)),
                    point_size=0.05,
                    point_shape="circle",
                    precision="float32",
                )
        qr_centers_path = OUTPUT_DIR / "qr_centers.pcd"
        if qr_centers_path.exists() and T_cam_lidar is not None:
            pts, _ = read_pcd_ascii(qr_centers_path)
            if len(pts):
                T_lidar_cam = np.linalg.inv(T_cam_lidar)
                pts_h = np.concatenate(
                    [pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1
                )
                pts = (T_lidar_cam @ pts_h.T).T[:, :3].astype(np.float32)
                scene.add_point_cloud(
                    "/layers/centers/qr",
                    points=pts,
                    colors=np.tile(np.array((0, 255, 0), dtype=np.uint8), (len(pts), 1)),
                    point_size=0.05,
                    point_shape="circle",
                    precision="float32",
                )

        # Sensor frames: LiDAR at origin, camera at inv(T_cam_lidar).
        scene.add_frame("/frames/lidar", axes_length=0.3, axes_radius=0.01)
        if T_cam_lidar is not None:
            T_lidar_cam = np.linalg.inv(T_cam_lidar)
            scene.add_frame(
                "/frames/camera",
                wxyz=mat_to_wxyz(T_lidar_cam[:3, :3]),
                position=T_lidar_cam[:3, 3],
                axes_length=0.3,
                axes_radius=0.01,
            )

            # QR detection image, placed just in front of the camera optical
            # frame (+z forward, +y down -> rotate 180 deg about x so the
            # texture faces the viewer standing behind the camera).
            image_path = OUTPUT_DIR / "qr_detect.png"
            if image_path.exists():
                image = np.asarray(Image.open(image_path).convert("RGB"))

                # Reprojected LiDAR depth map -> GUI panel at the top-left
                # corner of the page.
                colored_path = OUTPUT_DIR / "colored_cloud.pcd"
                if colored_path.exists() and getattr(self, "_last_params", None):
                    points_cam, _ = read_pcd_ascii(colored_path)
                    if len(points_cam):
                        h, w = image.shape[:2]
                        depth_rgb, mask = build_depth_image(
                            points_cam, self._last_params, w, h
                        )
                        if mask.any():
                            self.depth_panel.image = depth_rgb
                            self.depth_panel.visible = self.layer_checkboxes[
                                "depth_inset"
                            ].value

                h, w = image.shape[:2]
                render_w = 0.6
                R_img = T_lidar_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                fwd = T_lidar_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
                layer_frame("image")
                self._image_handle = scene.add_image(
                    "/layers/image/qr_detect",
                    image=image,
                    render_width=render_w,
                    render_height=render_w * h / w,
                    format="jpeg",
                    wxyz=mat_to_wxyz(R_img),
                    position=T_lidar_cam[:3, 3] + 0.3 * fwd,
                )
        else:
            layer_frame("image")


def main():
    app = FastCalibApp()
    print("FAST-Calib web UI: http://localhost:%d" % VISER_PORT)
    print("Scanning for scenes in: %s (%d found)" % (DATA_DIR, len(app.scenes)))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
