# FAST-Calib + viser web UI — minimal image based on the official ROS library.
# ros:noetic-ros-base (Ubuntu 20.04) supports both amd64 and arm64.

# ---------------------------------------------------------------------------
# Stage 1: build the catkin package
# ---------------------------------------------------------------------------
FROM ros:noetic-ros-base AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3-catkin-tools \
        ros-noetic-pcl-ros \
        ros-noetic-pcl-conversions \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-image-geometry \
        ros-noetic-rosbag \
        ros-noetic-tf \
        libopencv-contrib-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ws
COPY . src/fast_calib
RUN . /opt/ros/noetic/setup.sh && catkin_make -DCMAKE_BUILD_TYPE=Release

# ---------------------------------------------------------------------------
# Stage 2: runtime (ROS runtime + built package + viser web server)
# ---------------------------------------------------------------------------
FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-pcl-ros \
        ros-noetic-pcl-conversions \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-image-geometry \
        ros-noetic-rosbag \
        ros-noetic-tf \
        libopencv-contrib-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap a modern pip first: focal ships pip 20.0.2, which chokes on
# non-PEP440 versions in today's PyPI index.
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/pip/3.8/get-pip.py', '/tmp/get-pip.py')" \
    && python3 /tmp/get-pip.py --no-cache-dir \
    && rm /tmp/get-pip.py \
    && python3 -m pip install --no-cache-dir \
        "viser==1.1.0" \
        "numpy<1.25" \
        pillow \
        pyyaml

# Built catkin workspace + web app + default config
COPY --from=builder /opt/ws/devel /opt/ws/devel
COPY config/ /opt/fast_calib/config/
COPY web/ /opt/web/
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENV CALIB_DATA_DIR=/data \
    CALIB_OUTPUT_DIR=/output \
    FAST_CALIB_DEFAULT_PARAMS=/opt/fast_calib/config/qr_params.yaml \
    VISER_PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

WORKDIR /opt/web
ENTRYPOINT ["/opt/entrypoint.sh"]
