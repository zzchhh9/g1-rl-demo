#!/bin/bash
# ============================================================================
# 5x GATED-RETURN + FAKE-YOLO sequence, obstacle from the FRONT 180deg only.
#
# Same confirmed baseline (commit 6e77e7e) deploy/stack that recovered at 0.09m.
# The fake "person" crosses only from the robot's front hemisphere
#   (heading 90..270  =>  enter bearing  -90deg(right) .. 0(front) .. +90deg(left)),
# never from behind.
#
# Each of the 5 passes: stand -> person crosses the front -> dodge -> return to
# origin -> short gap -> next pass. Single trigger per pass.
#
# (deploy+stack are currently reverted to the 6e77e7e baseline; the later
#  real-YOLO WIP is in `git stash@{0}` -> `git stash pop` to restore it.)
# ============================================================================
set -e
cd "$(dirname "$0")"
exec env \
    YOLO_SOURCE=sequence \
    START_RGBD=0 \
    FAKE_YOLO_MAX_PASSES=6 \
    FAKE_YOLO_HEADING_MIN=90 \
    FAKE_YOLO_HEADING_MAX=270 \
    AUTO_START_SENSORS_BEFORE_DEPLOY=1 \
    ./deploy.sh \
        --return_mode gated \
        --return_gated_lin_vel 0.35 \
        --no_exit_on_return_abort \
        "$@"
