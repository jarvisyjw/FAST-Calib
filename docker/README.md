# FAST-Calib Docker + Web UI

A minimal Docker image for FAST-Calib with a browser-based visualization and
interaction platform built on [viser](https://viser.studio).

## Contents

- `Dockerfile` — multi-stage build on the official `ros:noetic-ros-base`
  (Ubuntu 20.04, amd64 + arm64). Stage 1 compiles the catkin package; stage 2
  keeps only the ROS runtime, the built workspace, and the viser web server.
- `entrypoint.sh` — starts `roscore`, waits for the master, then launches the
  web server.
- `../web/server.py` — the viser app (scene selection, parameter editing,
  calibration runner, 3D visualization, result download).

## Build

```bash
docker build -t fast-calib:web -f docker/Dockerfile .
```

## Run

```bash
# plain docker
docker run --rm -p 8080:8080 \
  -v $PWD/calib_data:/data:ro \
  -v $PWD/output:/output \
  fast-calib:web

# or docker compose
docker compose up --build
```

Then open **http://localhost:8080** in a browser.

## Using the web UI

1. **Scene** — pick a rosbag found under the mounted `calib_data/` folder. If
   an image file sits next to the bag it is used automatically; otherwise a
   frame is extracted from the bag's camera topic (set the topic names in the
   same panel).
2. **Camera intrinsics / Target geometry / Distance filter** — prefilled from
   `config/qr_params.yaml`; adjust per sensor/scene. The distance filter crop
   box around the calibration board is the most scene-dependent part.
3. **Run calibration** — launches the `fast_calib` node headless. RMSE and the
   extrinsic matrix `T_cam_lidar` appear in the **Results** panel, with a
   download button for `single_calib_result.txt`.
4. **Layers** — toggle the input cloud, distance-filtered cloud, RANSAC plane,
   edge points, fitted circle centers (red = LiDAR, green = camera/ArUco), the
   final colorized cloud, and the camera image. A jet-colorized **reprojected
   LiDAR depth map** is shown in a panel at the top-left corner of the page
   (toggle with "Depth image panel") for visual alignment checks. Everything
   is drawn in the LiDAR frame with the camera frame axes at the solved
   extrinsic pose.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CALIB_DATA_DIR` | `/data` | Directory scanned for `*.bag` scenes |
| `CALIB_OUTPUT_DIR` | `/output` | Where calibration outputs are written |
| `FAST_CALIB_DEFAULT_PARAMS` | `/opt/fast_calib/config/qr_params.yaml` | Default GUI parameter values |
| `FAST_CALIB_RUN_TIMEOUT` | `180` | Seconds to wait for the node |
| `VISER_PORT` | `8080` | Web server port |

## Notes

- Intermediate pipeline clouds are exported by the C++ node when the
  `save_intermediate` ROS param is `true` (the web runner sets it
  automatically). Plain `roslaunch fast_calib calib.launch` behavior is
  unchanged.
- Files written to the `output/` mount are owned by root inside the container.
