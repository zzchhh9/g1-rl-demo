#!/bin/bash
# One-command fake-YOLO dodge run.
#
# Same as deploy.sh, but uses the fake YOLO obstacle publisher instead of the
# camera. Because AUTO_START_SENSORS_BEFORE_DEPLOY defaults to 1, the stack
# brings up the RGBD/MID-360 driver + FAST-LIO odom + fake YOLO, waits until
# odom and YOLO are publishing, then runs the controller in this terminal.
#
# Extra args are forwarded to deploy_dodge_sdk_loco.py, e.g.:
#   ./deploy_fake.sh --return_yaw_gain=-1.5
#
# Override env as needed, e.g. YOLO_SOURCE=real ./deploy_fake.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
  YOLO_SOURCE="${YOLO_SOURCE:-fake}" \
  AUTO_START_SENSORS_BEFORE_DEPLOY="${AUTO_START_SENSORS_BEFORE_DEPLOY:-1}" \
  "$ROOT/scripts/g1_dodge_stack.sh" deploy "$@"
