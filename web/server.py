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
from PIL import Image, ImageDraw

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
    "multi_calib_result.txt",
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


def parse_centers_line(line):
    """'lidar_centers: {x,y,z} {x,y,z} ...' -> list of (x, y, z) tuples."""
    out = []
    for group in re.findall(r"\{([^}]*)\}", line):
        vals = [float(v) for v in group.replace(" ", "").split(",") if v]
        if len(vals) != 3:
            return None
        out.append(tuple(vals))
    return out or None


def parse_center_records(path):
    """Parse output/circle_center_record.txt into a list of
    (time_line, lidar_pts[4], qr_pts[4]) blocks, mirroring multi_scene.cpp."""
    path = Path(path)
    if not path.exists():
        return []
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    blocks = []
    i = 0
    while i + 2 < len(lines):
        if (
            lines[i].startswith("time:")
            and "lidar_centers:" in lines[i + 1]
            and "qr_centers:" in lines[i + 2]
        ):
            lidar_pts = parse_centers_line(lines[i + 1])
            qr_pts = parse_centers_line(lines[i + 2])
            if lidar_pts and qr_pts and len(lidar_pts) == 4 and len(qr_pts) == 4:
                blocks.append((lines[i], lidar_pts, qr_pts))
            i += 3
        else:
            i += 1
    return blocks


def per_scene_rmse(blocks, T_cam_lidar):
    """RMSE of each recorded scene under the (multi-scene) extrinsic."""
    R, t = T_cam_lidar[:3, :3], T_cam_lidar[:3, 3]
    rmses = []
    for _, lidar_pts, qr_pts in blocks:
        L = np.asarray(lidar_pts)
        C = np.asarray(qr_pts)
        aligned = (R @ L.T).T + t
        rmses.append(float(np.sqrt(np.mean(np.sum((aligned - C) ** 2, axis=1)))))
    return rmses


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def subsample(points, colors=None, max_points=250_000):
    if len(points) <= max_points:
        return points, colors
    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx], colors[idx] if colors is not None else None


def project_points_distorted(points_cam, params):
    """Project camera-frame points into the distorted image plane.
    Returns (u, v, z) float arrays."""
    z = points_cam[:, 2]
    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z
    r2 = x * x + y * y
    radial = 1.0 + params["k1"] * r2 + params["k2"] * r2 * r2
    xd = x * radial + 2.0 * params["p1"] * x * y + params["p2"] * (r2 + 2.0 * x * x)
    yd = y * radial + params["p1"] * (r2 + 2.0 * y * y) + 2.0 * params["p2"] * x * y
    u = params["fx"] * xd + params["cx"]
    v = params["fy"] * yd + params["cy"]
    return u, v, z


def build_depth_image(points_cam, params, width, height):
    """Project camera-frame points into the (distorted) image plane and render
    a z-buffered, jet-colorized depth map. Returns (rgb uint8 HxWx3, mask bool)."""
    z = points_cam[:, 2]
    valid = z > 0.1
    pts, z = points_cam[valid], z[valid]

    u, v, z = project_points_distorted(pts, params)

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


def compose_depth_comparison(depth_single, depth_multi):
    """Side-by-side depth images: single-scene (left) vs multi-scene (right),
    with labels, for direct quality comparison."""
    h, w = depth_single.shape[:2]
    hw, hh = w // 2, h // 2
    label_h = 28
    out = Image.new("RGB", (hw * 2 + 6, hh + label_h), (24, 24, 28))
    out.paste(Image.fromarray(depth_single).resize((hw, hh), Image.BILINEAR), (0, label_h))
    out.paste(Image.fromarray(depth_multi).resize((hw, hh), Image.BILINEAR), (hw + 6, label_h))
    d = ImageDraw.Draw(out)
    d.text((8, 8), "single-scene", fill=(255, 200, 80))
    d.text((hw + 14, 8), "multi-scene", fill=(120, 220, 120))
    return np.asarray(out)


def colorize_cloud(points_lidar, T_cam_lidar, params, image_rgb):
    """Project LiDAR-frame points through the extrinsic onto the (distorted)
    camera image and sample pixel colors. Returns (points_lidar, colors)
    filtered to points landing inside the image."""
    pts_h = np.concatenate(
        [points_lidar, np.ones((len(points_lidar), 1), dtype=np.float32)], axis=1
    )
    pts_cam = (T_cam_lidar @ pts_h.T).T[:, :3].astype(np.float64)
    u, v, z = project_points_distorted(pts_cam, params)

    h, w = image_rgb.shape[:2]
    valid = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui = u[valid].astype(np.int64)
    vi = v[valid].astype(np.int64)
    colors = image_rgb[vi, ui]
    return points_lidar[valid], colors


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
            self.rescan_button = gui.add_button(
                "Rescan scenes",
                hint="Re-scan the data folder for newly added rosbags.",
            )
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

        with gui.add_folder("Multi-scene calibration", expand_by_default=False):
            gui.add_markdown(
                "Each single-scene run appends its 4 circle-center pairs to "
                "`circle_center_record.txt`. With **≥ 3 scenes** recorded, the "
                "joint solve pools the last 3 scenes (12 pairs) into one "
                "least-squares estimate."
            )
            self.multi_status_md = gui.add_markdown("")
            self.clear_records_button = gui.add_button(
                "Clear recorded scenes",
                hint="Truncate circle_center_record.txt to start a fresh multi-scene session.",
            )
            self.multi_run_button = gui.add_button(
                "Run multi-scene calibration", color="green", disabled=True
            )
            self.multi_result_md = gui.add_markdown("_No multi-scene run yet._")
            self.multi_download_button = gui.add_button(
                "Download multi-scene result", disabled=True
            )

        with gui.add_folder("Layers", expand_by_default=False):
            self.layer_checkboxes = {}
            for key, label, default in (
                ("input", "Input cloud", True),
                ("filtered", "Filtered cloud", False),
                ("plane", "Plane (RANSAC)", False),
                ("edge", "Edge points", False),
                ("centers", "Circle centers", True),
                ("colored", "Colored cloud (single-scene)", True),
                ("colored_multi", "Colored cloud (multi-scene)", True),
                ("image", "Camera image", True),
                ("depth_inset", "Depth image panel (top-left)", True),
            ):
                self.layer_checkboxes[key] = gui.add_checkbox(label, initial_value=default)
                self.layer_checkboxes[key].on_update(self._on_layer_toggle)
            self.isolate_checkbox = gui.add_checkbox(
                "Show multi-scene cloud only",
                initial_value=False,
                hint="Hide everything except the joint-extrinsic colored cloud.",
            )
            self.isolate_checkbox.on_update(self._on_isolate_toggle)

        self.run_button.on_click(self._on_run_clicked)
        self.rescan_button.on_click(self._on_rescan)
        self.download_button.on_click(self._on_download)
        self.clear_records_button.on_click(self._on_clear_records)
        self.multi_run_button.on_click(self._on_multi_run_clicked)
        self.multi_download_button.on_click(self._on_multi_download)
        self.last_multi_result_path = None
        self._refresh_multi_status()

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

    def _on_isolate_toggle(self, event):
        self._apply_isolation(event.target.value)

    def _apply_isolation(self, isolated):
        """Show only the multi-scene colored cloud; restore on uncheck."""
        for key, frame in getattr(self, "_layer_frames", {}).items():
            if isolated:
                frame.visible = key == "colored_multi"
            else:
                frame.visible = self.layer_checkboxes[key].value
        for frame in getattr(self, "_sensor_frames", {}).values():
            frame.visible = not isolated

    def _on_rescan(self, _):
        self.scenes = find_scenes()
        options = list(self.scenes.keys()) or ["(no scenes found under %s)" % DATA_DIR]
        self.scene_dropdown.options = options
        self.scene_dropdown.value = options[0]
        self.run_button.disabled = not self.scenes
        self.status_md.content = "**Status:** found %d scene(s)" % len(self.scenes)

    def _on_download(self, event):
        if event.client is None or self.last_result_path is None:
            return
        path = Path(self.last_result_path)
        if path.exists():
            event.client.send_file_download(path.name, path.read_bytes())

    # ------------------------------------------------------- multi-scene

    def _refresh_multi_status(self):
        blocks = parse_center_records(OUTPUT_DIR / "circle_center_record.txt")
        n = len(blocks)
        self.multi_status_md.content = (
            "**Recorded scenes:** %d (joint solve uses the last 3)" % n
        )
        self.multi_run_button.disabled = n < 3 or self.run_lock.locked()
        return n

    def _on_clear_records(self, _):
        try:
            (OUTPUT_DIR / "circle_center_record.txt").unlink()
        except FileNotFoundError:
            pass
        self.multi_result_md.content = "_No multi-scene run yet._"
        self.multi_download_button.disabled = True
        self._refresh_multi_status()

    def _on_multi_download(self, event):
        if event.client is None or self.last_multi_result_path is None:
            return
        path = Path(self.last_multi_result_path)
        if path.exists():
            event.client.send_file_download(path.name, path.read_bytes())

    def _on_multi_run_clicked(self, _):
        if self.run_lock.locked():
            return
        threading.Thread(target=self._run_multi_calibration, daemon=True).start()

    def _run_multi_calibration(self):
        with self.run_lock:
            self.multi_run_button.disabled = True
            self.run_button.disabled = True
            try:
                self._run_multi_calibration_impl()
            except Exception as exc:
                self.multi_status_md.content = "**Status:** error — `%s`" % exc
            finally:
                self.run_button.disabled = not self.scenes
                self._refresh_multi_status()

    def _run_multi_calibration_impl(self):
        yaml_path = Path("/tmp/fast_calib_multi.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump({"output_path": str(OUTPUT_DIR)}, f)
        subprocess.run(["rosparam", "load", str(yaml_path)], check=True)

        log_path = OUTPUT_DIR / "multi_fast_calib_node.log"
        self.multi_status_md.content = "**Status:** running multi-scene solve…"
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                ["rosrun", "fast_calib", "multi_fast_calib"],
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.multi_status_md.content = "**Status:** multi-scene solve timed out"
                return

        log_text = (
            strip_ansi(log_path.read_text(errors="replace")) if log_path.exists() else ""
        )
        result_path = OUTPUT_DIR / "multi_calib_result.txt"
        if proc.returncode != 0 or not result_path.exists():
            tail = "\n".join(log_text.strip().splitlines()[-15:])
            self.multi_status_md.content = "**Status:** multi-scene solve failed"
            self.multi_result_md.content = "**Node log (tail):**\n```\n%s\n```" % tail
            return

        T = parse_result_txt(result_path)
        rmse = parse_rmse(log_text)
        blocks = parse_center_records(OUTPUT_DIR / "circle_center_record.txt")[-3:]

        self.last_multi_result_path = str(result_path)
        self.multi_download_button.disabled = False
        self.multi_status_md.content = (
            "**Status:** multi-scene solve done (%d scenes pooled)" % len(blocks)
        )

        lines = []
        if rmse is not None:
            lines.append("**Joint RMSE (12 pairs):** `%.4f m`" % rmse)
        if T is not None:
            lines.append("**Per-scene RMSE under joint extrinsic:**")
            for k, scene_rmse in enumerate(per_scene_rmse(blocks, T)):
                lines.append("- scene %d: `%.4f m`" % (k + 1, scene_rmse))
            lines.append("")
            lines.append("**T_cam_lidar (multi-scene):**")
            lines.append("```")
            for row in T:
                lines.append(" ".join("% .6f" % v for v in row))
            lines.append("```")
        self.multi_result_md.content = (
            "\n".join(lines) if lines else "_Result file written._"
        )

        if T is not None:
            self._update_scene_multi(T)

    # ------------------------------------------- multi-scene visualization

    def _update_scene_multi(self, T_cam_lidar):
        """Re-render the 3D view with the joint (multi-scene) extrinsic:
        re-color the input cloud by projecting it through the joint T onto the
        camera image, move the camera frame/image, and show all recorded
        scenes' center pairs."""
        scene = self.server.scene
        params = getattr(self, "_last_params", None)
        blocks = parse_center_records(OUTPUT_DIR / "circle_center_record.txt")[-3:]

        # Camera frame + image at the joint pose.
        T_lidar_cam = np.linalg.inv(T_cam_lidar)
        self._sensor_frames["camera"] = scene.add_frame(
            "/frames/camera",
            wxyz=mat_to_wxyz(T_lidar_cam[:3, :3]),
            position=T_lidar_cam[:3, 3],
            axes_length=0.3,
            axes_radius=0.01,
        )
        image_handle = getattr(self, "_image_handle", None)
        if image_handle is not None:
            R_img = T_lidar_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            fwd = T_lidar_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
            image_handle.wxyz = mat_to_wxyz(R_img)
            image_handle.position = T_lidar_cam[:3, 3] + 0.3 * fwd

        # Center pairs of all pooled scenes: red = LiDAR frame detections,
        # green = camera detections mapped into the LiDAR frame via joint T.
        if blocks:
            lidar_all = np.array([p for _, lp, _ in blocks for p in lp], dtype=np.float32)
            qr_all = np.array([p for _, _, qp in blocks for p in qp], dtype=np.float32)
            qr_h = np.concatenate(
                [qr_all, np.ones((len(qr_all), 1), dtype=np.float32)], axis=1
            )
            qr_in_lidar = (T_lidar_cam @ qr_h.T).T[:, :3].astype(np.float32)
            scene.add_point_cloud(
                "/layers/centers/lidar",
                points=lidar_all,
                colors=np.tile(np.array((255, 0, 0), dtype=np.uint8), (len(lidar_all), 1)),
                point_size=0.05,
                point_shape="circle",
                precision="float32",
            )
            scene.add_point_cloud(
                "/layers/centers/qr",
                points=qr_in_lidar,
                colors=np.tile(np.array((0, 255, 0), dtype=np.uint8), (len(qr_in_lidar), 1)),
                point_size=0.05,
                point_shape="circle",
                precision="float32",
            )

        # Re-color the cloud with the joint extrinsic into its OWN layer
        # (/layers/colored_multi) so the single-scene result stays available
        # for toggled comparison. Prefer the dense colored cloud from the
        # last single-scene run (camera frame under T_single -> map back to
        # LiDAR frame); fall back to the downsampled input cloud.
        image_path = OUTPUT_DIR / "qr_detect.png"
        points_lidar = None
        colored_path = OUTPUT_DIR / "colored_cloud.pcd"
        T_single = parse_result_txt(OUTPUT_DIR / "single_calib_result.txt")
        if colored_path.exists() and T_single is not None:
            pts_cam, _ = read_pcd_ascii(colored_path)
            if len(pts_cam):
                pts_h = np.concatenate(
                    [pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)], axis=1
                )
                points_lidar = (np.linalg.inv(T_single) @ pts_h.T).T[:, :3].astype(
                    np.float32
                )
        if points_lidar is None and (OUTPUT_DIR / "input_cloud.pcd").exists():
            points_lidar, _ = read_pcd_ascii(OUTPUT_DIR / "input_cloud.pcd")

        if params and image_path.exists() and points_lidar is not None and len(points_lidar):
            image = np.asarray(Image.open(image_path).convert("RGB"))
            pts, colors = colorize_cloud(points_lidar, T_cam_lidar, params, image)
            if len(pts):
                pts, colors = subsample(pts, colors)
                frame = scene.add_frame(
                    "/layers/colored_multi",
                    show_axes=False,
                    visible=self.layer_checkboxes["colored_multi"].value,
                )
                self._layer_frames["colored_multi"] = frame
                self._colored_multi_handle = scene.add_point_cloud(
                    "/layers/colored_multi/cloud",
                    points=pts,
                    colors=colors,
                    point_size=0.01,
                    point_shape="circle",
                    precision="float32",
                )
                if self.isolate_checkbox.value:
                    self._apply_isolation(True)

            # Depth panel: render BOTH extrinsics from the same dense cloud
            # and show them side by side (single | multi) for comparison.
            pts_h = np.concatenate(
                [points_lidar, np.ones((len(points_lidar), 1), dtype=np.float32)],
                axis=1,
            )
            h, w = image.shape[:2]
            pts_cam_multi = (T_cam_lidar @ pts_h.T).T[:, :3]
            depth_multi, mask_multi = build_depth_image(pts_cam_multi, params, w, h)
            depth_single, mask_single = None, None
            if T_single is not None:
                pts_cam_single = (T_single @ pts_h.T).T[:, :3]
                depth_single, mask_single = build_depth_image(
                    pts_cam_single, params, w, h
                )
            if (
                depth_single is not None
                and mask_single.any()
                and mask_multi.any()
            ):
                self.depth_panel.image = compose_depth_comparison(
                    depth_single, depth_multi
                )
            elif mask_multi.any():
                self.depth_panel.image = depth_multi
            if (depth_single is not None and mask_single.any()) or mask_multi.any():
                self.depth_panel.visible = self.layer_checkboxes[
                    "depth_inset"
                ].value

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
            self.multi_run_button.disabled = True
            try:
                self._run_calibration_impl()
            except Exception as exc:  # never leave the UI stuck
                self.status_md.content = "**Status:** error — `%s`" % exc
            finally:
                self.run_button.disabled = False
        self._refresh_multi_status()

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
        self._sensor_frames = {}
        # A fresh single-scene run invalidates the isolated multi view.
        if self.isolate_checkbox.value:
            self.isolate_checkbox.value = False
        # Drop the previous multi-scene cloud; it belongs to an older scene.
        handle = getattr(self, "_colored_multi_handle", None)
        if handle is not None:
            handle.remove()
            self._colored_multi_handle = None

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
        self._sensor_frames["lidar"] = scene.add_frame(
            "/frames/lidar", axes_length=0.3, axes_radius=0.01
        )
        if T_cam_lidar is not None:
            T_lidar_cam = np.linalg.inv(T_cam_lidar)
            self._sensor_frames["camera"] = scene.add_frame(
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
