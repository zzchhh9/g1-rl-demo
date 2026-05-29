#!/bin/bash
# ============================================================================
# Confirmed GATED-RETURN + FAKE-YOLO baseline.
#
# This runs the deploy/stack as they were at commit 6e77e7e
#   ("deploy: gated return-head recovery + randomized/sequenced fake YOLO"),
# which validated the dodge+recover at RETURN DONE 0.09m in:
#   #1 runs/rhea_20260528_233107   (DODGE STOP 0.50m -> RETURN DONE 0.09m)
#   #2 runs/rhea_20260528_235319   (RETURN DONE 0.09m)
#
# deploy_dodge_sdk_loco.py + scripts/g1_dodge_stack.sh are currently REVERTED to
# that version (the later real-YOLO WIP — auto_balance / settle-skip / dodge cap /
# startup-calib / camera fixes — is saved in `git stash@{0}`; run `git stash pop`
# to bring it back).
#
# Behavior: robot stands -> fake "person" makes ONE crossing -> dodge -> return.
# Single trigger (YOLO_SOURCE=fake = one approach), gated return, lin_vel 0.35 (= #1).
# ============================================================================
set -e
cd "$(dirname "$0")"
exec ./deploy_fake.sh \
    --return_mode gated \
    --return_gated_lin_vel 0.35 \
    --no_exit_on_return_abort \
    "$@"
