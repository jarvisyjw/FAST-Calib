#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
source /opt/ws/devel/setup.bash

# Start a ROS master (the fast_calib node reads params from the param server).
roscore > /tmp/roscore.log 2>&1 &
until rosnode list > /dev/null 2>&1; do
  sleep 0.5
done

mkdir -p "${CALIB_OUTPUT_DIR:-/output}"

exec python3 /opt/web/server.py
