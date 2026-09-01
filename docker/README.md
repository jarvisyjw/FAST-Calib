# FAST-Calib Docker + Web UI

A minimal Docker image for FAST-Calib with a browser-based visualization and
interaction platform built on [viser](https://viser.studio).

## Contents

- `Dockerfile` — multi-stage build on the official `ros:noetic-ros-base`
  (Ubuntu 20.04, amd64 + arm64). Stage 1 compiles the catkin package; stage 2
  keeps only the ROS runtime, the built workspace, and the viser web server.
- `entrypoint.sh` — starts `roscore`, waits for the master, then launches the
  web server.
- `../web/server.py` — the viser app (online capture trigger, parameter
  editing, calibration runner, 3D visualization, result download).

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

## Using the web UI (online mode)

> **Full user manual: see the main [`README.md`](../README.md)**
> (quick start, procedure, UI reference, pipeline details, troubleshooting).

This branch runs **online calibration**: instead of reading a rosbag file, the
`fast_calib_online` node subscribes to live LiDAR/camera topics and captures
on demand. Each capture **accumulates several consecutive LiDAR frames into
one dense cloud** (a single 100 ms Livox frame is far too sparse for the
circle detection — the offline mode works because it accumulates the whole
bag). Imitate a live stream from a recorded bag by playing it back in a loop:

```bash
docker exec -it fastcalib bash -c \
  "source /opt/ros/noetic/setup.bash && source /opt/ws/devel/setup.bash && \
   rosbag play -l /data/<scene>.bag"
```

(with plain `docker run`, open a second shell into the running container;
the bag must be under the mounted `/data` folder.)

1. **Live capture** — set the LiDAR/camera topic names to match the
   publishing driver (or the `rosbag play` topics).
2. **Camera intrinsics / Target geometry / Distance filter** — prefilled from
   `config/qr_params.yaml`; adjust per sensor/scene. The distance filter crop
   box around the calibration board is the most scene-dependent part.
3. **Run calibration** — the node starts lazily on the first click, then each
   click calls the `/fast_calib_online/capture` service: it accumulates
   **"LiDAR frames per capture"** consecutive LiDAR frames (default 20 ≈ 2 s
   at 10 Hz — **keep the board still during this window**) and pairs them with
   the camera frame nearest to the middle of the window, runs the detection
   pipeline, and appends the 4 circle-center
   pairs to `output/circle_center_record.txt`. Failed detections are reported
   and **not** recorded. RMSE and `T_cam_lidar` appear in the **Results**
   panel, with a download button for `single_calib_result.txt`.
4. **Layers** — toggle the input cloud, distance-filtered cloud, RANSAC plane,
   edge points, fitted circle centers (red = LiDAR, green = camera/ArUco), the
   final colorized cloud, and the camera image. A jet-colorized **reprojected
   LiDAR depth map** is shown in a panel at the top-left corner of the page
   (toggle with "Depth image panel") for visual alignment checks. Everything
   is drawn in the LiDAR frame with the camera frame axes at the solved
   extrinsic pose.

5. **Multi-scene calibration** — after the **3rd successful capture** the
   joint solve runs **automatically**: it pools the last 3 recorded scenes
   (12 pairs) into one least-squares estimate and reports the joint RMSE plus
   the per-scene RMSE under the joint extrinsic. The joint result gets its own
   layer ("Colored cloud (multi-scene)") so it can be toggled against the
   single-scene result; the top-left depth panel switches to a **side-by-side
   comparison** (single-scene | multi-scene) rendered from the same dense
   cloud, and the **"Show multi-scene cloud only"** checkbox in Layers hides
   everything else for a clean view of the joint reconstruction. Use "Clear
   recorded scenes" before starting a new sensor/scene session.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CALIB_DATA_DIR` | `/data` | Mount point for rosbags played back with `rosbag play` |
| `CALIB_OUTPUT_DIR` | `/output` | Where calibration outputs are written |
| `FAST_CALIB_DEFAULT_PARAMS` | `/opt/fast_calib/config/qr_params.yaml` | Default GUI parameter values |
| `FAST_CALIB_RUN_TIMEOUT` | `180` | Seconds to wait for the capture service call |
| `VISER_PORT` | `8080` | Web server port |

## Notes

- Intermediate pipeline clouds are exported by the C++ node when the
  `save_intermediate` ROS param is `true` (the web runner sets it
  automatically). Plain `roslaunch fast_calib calib.launch` behavior is
  unchanged.
- Files written to the `output/` mount are owned by root inside the container.
