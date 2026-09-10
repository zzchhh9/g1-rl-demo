# G1 SDK Locomotion Dodge Runbook

Operator front door (遥操 / SLAM / quickstart): [`README.md`](../README.md).
Docs index: [`docs/README.md`](README.md).

This is the current recommended real-robot path. It bypasses the
`motion.pt` low-level locomotion policy and sends velocity commands to the
Unitree built-in high-level locomotion controller through
`LocoClient.SetVelocity()`.

`scripts/g1_dodge_stack.sh` currently defaults to `SLAM_BACKEND=fast_lio`
(`ROS_ODOM_TOPIC=/Odometry`). LIO-SAM remains available with
`SLAM_BACKEND=lio_sam` (`/lio_sam_ros2/mapping/odometry`) and is what
`./start_lidar_mapping.sh` uses. Both backends still bridge to DDS
`rt/dodge/odom`.

## Current Known-Good Flow

As of the last hardware check, the working stack is:

```text
RealSense -> YOLO DDS rt/yolo/person
MID-360 -> robot livox_ros_driver2 xfer_format=1
        -> robot lio_sam_ros2 /lio_sam_ros2/mapping/odometry
        -> robot UDP sender
        -> laptop UDP-to-DDS rt/dodge/odom
deploy_dodge_sdk_loco.py -> Unitree LocoClient.SetVelocity
```

The shortest operational sequence is:

```bash
cd /home/zz4723/g1-rl-demo
./start_yolo_lidar.sh
uv run python scripts/dds_odom_echo.py eno1 --topic rt/dodge/odom --duration 10
```

Then put the robot in blue locomotion mode with the remote and run:

```bash
./deploy.sh
```

Expected odom check:

```text
[echo] fresh age=0.0xs hz=5.0 n=... src=ros2:/lio_sam_ros2/mapping/odometry
```

Expected YOLO state when nobody is in front of the camera:

```text
rt/yolo/person publishes at about 9-10 Hz with {"n": 0}
```

Expected YOLO state when a person is in front of the camera:

```bash
uv run python scripts/dds_yolo_echo.py eno1 --duration 5
```

```text
[echo] fresh age=0.0xs hz=9-10 n=1 dist=+1.xx x=+1.xx y=... track=...
```

If deploy prints `dist=N/A obs_b=None`, do not debug locomotion first. Check
YOLO:

```bash
tmux capture-pane -pt g1-dodge:yolo -S -80
uv run python scripts/dds_yolo_echo.py eno1 --duration 5
```

The bad stale-lock signature is:

```text
n=0 ... locked=True locked_id=<old id>
```

Restart the YOLO window with the default no-lock launch:

```bash
START_RGBD=0 START_LIDAR_DRIVER=0 START_LIO=0 \
START_ODOM_BRIDGE=0 START_ODOM_ECHO=0 \
YOLO_LOCK_FIRST_TRACK=0 ./start_yolo_lidar.sh
```

By default, the sensor wrapper does not lock YOLO at camera startup:

```text
YOLO_LOCK_FIRST_TRACK=0
```

This is intentional. The deploy controller locks the first `track_id` that
actually triggers DODGE, not a random far-away track seen while the robot is
idle. If you force camera-side locking, use a distance gate:

```bash
YOLO_LOCK_FIRST_TRACK=1 YOLO_LOCK_DIST=1.0 ./start_yolo_lidar.sh
```

## Safety First

Keep a separate terminal ready for software zero velocity:

```bash
cd /home/zz4723/g1-rl-demo
uv run python scripts/g1_emergency_stop.py eno1
```

This is only a software stop. The real fallback is the remote/controller
firmware shortcut:

```text
Hold L2 -> press/hold B -> keep holding until G1 enters damping.
```

On this G1, the hardware damping shortcut takes about 5 seconds. That threshold
is firmware-side and is not configurable from this repo. The SDK dodge script
also has a software shortcut: while the script is running, holding `L2+B` for
2 seconds sends zero velocity, calls `Damp()`, and exits.

## Bring Up Camera

Robot reboots clear `/tmp`, so restart the RGBD publisher before YOLO:

```bash
cd /home/zz4723/g1-rl-demo
uv run --with paramiko python scripts/restart_rgbd_robot.py
```

Expected output includes:

```text
[rs] listening on :5005 (accepts repeatedly) ...
0.0.0.0:5005
```

Quick laptop-side TCP check:

```bash
uv run python - <<'PY'
import socket, struct, json
s = socket.create_connection(("192.168.123.164", 5005), timeout=5)
n = struct.unpack("!I", s.recv(4))[0]
print(json.loads(s.recv(n).decode()))
s.close()
PY
```

## Start YOLO Bridge

Terminal 1:

```bash
cd /home/zz4723/g1-rl-demo
uv run --with "pillow==9.5.0" --with "ultralytics==8.4.51" --with "lap" \
  python scripts/yolo_to_dds_laptop.py \
    --robot-ip 192.168.123.164 \
    --depth-offset 0.20 \
    --print-every 20
```

If this fails with `ConnectionRefusedError`, restart the camera publisher.

## One-Command Sensor Bringup

The dodge controller still needs YOLO for person position. MID-360 odometry is
only for robot displacement/recover.

Instead of starting camera, YOLO, LiDAR driver, and odom bridge manually, run:

```bash
cd /home/zz4723/g1-rl-demo
./start_yolo_lidar.sh
```

This creates a `tmux` session named `g1-dodge` with:

```text
yolo        RealSense TCP -> YOLO -> DDS rt/yolo/person
odom_udp    robot ROS2 /lio_sam_ros2/mapping/odometry -> laptop UDP
odom_bridge laptop UDP -> DDS rt/dodge/odom
odom_echo   DDS odom monitor
```

It also restarts the robot-side RGBD publisher and starts the robot-side
MID-360 ROS2 driver and robot-side LIO. Attach logs with:

```bash
tmux attach -t g1-dodge
```

Default odometry topic:

```bash
ROS_ODOM_TOPIC=/lio_sam_ros2/mapping/odometry
```

If you want a different local LIO/SLAM stack, pass `LIO_CMD` and set
`ROS_ODOM_TOPIC` accordingly.

## Put G1 In Blue Locomotion Mode

Use the remote:

```text
L2 + UP until the controller light is blue.
```

The script requires the SDK locomotion state to become:

```text
fsm=200 mode=1 balance=1
```

It calls `SetFsmId(200)` and `SetBalanceMode(1)` at startup, then waits for
that stable walking state before sending dodge velocity.

## Run SDK Dodge

Terminal 2:

```bash
cd /home/zz4723/g1-rl-demo
uv run python /home/zz4723/g1-rl-demo/deploy_dodge_sdk_loco.py eno1 \
  --source yolo \
  --max_vel 0.30 \
  --safety_dist 1.0 \
  --clear_margin 0.00 \
  --debug_obs
```

Shortcut:

```bash
cd /home/zz4723/g1-rl-demo
./deploy.sh
```

`deploy.sh` defaults to `--return_odom_source external`, so it expects fresh
`rt/dodge/odom`. To use the old command-integrated return path for debugging:

```bash
RETURN_ODOM_SOURCE=cmd ./deploy.sh
```

Before running `./deploy.sh`, verify:

```bash
uv run python scripts/dds_odom_echo.py eno1 --topic rt/dodge/odom --duration 5
```

If that prints `no odom` or `STALE`, do not run dodge. Restart sensors:

```bash
scripts/g1_dodge_stack.sh stop
./start_yolo_lidar.sh
```

If `./start_yolo_lidar.sh` reports that `192.168.123.164:22` is unreachable,
the robot-side services cannot be restarted. Check the laptop Ethernet first:

```bash
ip -br addr show eno1
ip route get 192.168.123.164
ip neigh show 192.168.123.164
```

Good state:

```text
eno1 UP ... 192.168.123.222/24
192.168.123.164 dev eno1 ...
192.168.123.164 lladdr ... REACHABLE
```

Bad state:

```text
eno1 DOWN ...
192.168.123.164 FAILED
```

Fix the Ethernet cable/adapter or robot Jetson network first, then rerun the
sensor startup.

Expected startup:

```text
[SDK LOCO DODGE]
  return: on mode=geo ... online_map=3/8
  return geo: pure LIO displacement + LIO yaw closed-loop, no checkpoint head, no online learned frame
[loco] ready check fsm=200(code=0) mode=1(code=0) balance=1(code=0)
[ENABLE] SDK loco dodge commands enabled
```

Expected dodge:

```text
[DODGE START] dist=...
[DODGE ] cmd=[...]
[DODGE STOP] dist=... est_disp=...
[RETURN START] est_disp=... est_xy=[...]
[RETURN ] cmd=[...] ... rframe=online(...) ret=[...]
[RETURN DONE] est_disp=...
```

The dodge action should have `dir=away` and negative `par`, meaning the velocity
component points away from the person.

## Run SDK Dodge With MID-360 Odometry Return

For real dodge-and-recover, use MID-360 LiDAR odometry instead of command
integration. Full setup is in
[`docs/mid360_lio_recover.md`](mid360_lio_recover.md).

Controller command after `rt/dodge/odom` is fresh:

```bash
cd /home/zz4723/g1-rl-demo
uv run python /home/zz4723/g1-rl-demo/deploy_dodge_sdk_loco.py eno1 \
  --source yolo \
  --max_vel 0.30 \
  --safety_dist 1.0 \
  --clear_margin 0.00 \
  --return_odom_source external \
  --odom_topic rt/dodge/odom \
  --odom_staleness 0.35 \
  --debug_obs
```

If external odometry is missing, the script raises instead of doing blind
return.

## Return Behavior

The normal SDK return path is now:

```text
MID-360/LIO odom displacement + live LIO yaw -> geometric closed-loop command
to the dodge origin -> LocoClient.SetVelocity
```

The online SDK-command/odom fit and checkpoint head are diagnostic only in the
default geo path.

The default return mode is `--return_mode geo`. The checkpoint return head is
not used in the normal real-robot SDK path because its online frame/head output
was unstable on the robot logs. Odometry is the closed-loop state: the
controller keeps commanding the body velocity that should reduce the LIO
displacement from the `DODGE START` origin.

In `--return_mode geo`, `rframe=odom_yaw` means the return command is computed
from live LIO yaw. The online SDK-command/odom frame is still printed under
`[DBG-FRAME]`, but it is diagnostic only and is not used by the geo controller.
`rframe=online_frozen_yaw(n)` only applies to non-geo debug modes.

`--return_mode head` is still available for replay/debugging. In that mode,
`guard=on` means the raw checkpoint output was corrected to preserve progress
and cap lateral velocity. Do not use the head mode as the default real-robot
return path unless you are deliberately comparing it against `geo`.

When `--debug_obs` is enabled, return logs also include:

```text
[DBG-ODOM] raw=... origin=... disp_w=... lio_yaw=... low_yaw=...
[DBG-FRAME] samples=... rot=... map=... det=... rms=... last_cmd=... last_dodom=...
[DBG-RETURN] disp_w=... disp_b=... back_w=... cmd_odom=... prog_dot=... prog_cos=...
             dodom_w=... actual_dot=... actual_cos=... final_target=...
```

Use these to validate the coordinate system:

- `raw - origin == disp_w`: the LIO displacement since `DODGE START`.
- In geo mode, `rot` is the live LIO yaw rotation used for return. The
  `[DBG-FRAME]` online fit is diagnostic only.
- `disp_b = rot.T @ disp_w`: the same displacement in the return frame.
- `cmd_odom = rot @ cmd`: the current SDK command projected back into LIO.
- `prog_dot = dot(cmd_odom, -disp_w)` and `prog_cos` should be positive during
  return. If they are negative, the command is moving away from the origin and
  the frame/sign mapping is wrong.
- `actual_dot = dot(dodom_w, previous_back_w)` is the measured LIO progress
  from the last odom update. If it stays non-positive, the robot is not really
  returning even if the commanded direction looks correct.

Without `--return_odom_source external`, the SDK path does not expose reliable
G1 odometry. The fallback displacement estimate is still rough:

```text
estimated displacement += commanded SDK velocity * dt
```

That is good enough only for debugging. Tune only after checking the logs:

```bash
# Return too little:
--return_odom_scale 1.3

# Return too much:
--return_odom_scale 0.7

# Disable return:
--no_return
```

Useful return knobs for the default LIO geometric return:

```text
--return_mode geo
--return_max_vel 0.25
--return_max_lat_vel 0.08
--return_gain 0.8
--return_lat_gain 0.35
--return_done_dist 0.10
--return_timeout 15.0
--return_no_progress_timeout 3.0
--return_clear_delay 0.5
--return_map_window 8
--return_map_min_samples 3
# Enabled by default:
# --return_freeze_frame
```

The checkpoint head and old hand-written return controller are still available
only for debugging:

```text
--return_mode head
--return_head_ckpt checkpoints/return_head_v23b_v6.pt
--return_head_guard
```

```text
--return_mode p
--return_max_lat_vel 0.08
--return_gain 0.8
--return_lat_gain 0.35
--return_lateral_sign 1
```

`RETURN` is interrupted immediately if a person enters `safety_dist` again.
If a person is outside `safety_dist` but still inside
`safety_dist + clear_margin`, return holds zero velocity and pauses its timeout
instead of burning the timeout while blocked.

Expected blocked-return log:

```text
[RETURN WAIT] obstacle dist=...m inside clear margin; holding zero
```

The 2026-05-18 replay showed that feeding raw LIO yaw directly into return can
produce the same wrong main direction as the old P controller. The default
therefore uses the online SDK-command/odom frame first and LIO yaw only as a
fallback.

## Current Important Implementation Details

- `./start_yolo_lidar.sh`
  - Restarts robot RGBD publisher on TCP `:5005`.
  - Starts robot-side `livox_ros_driver2` with `xfer_format=1`.
  - Checks actual live `/livox/imu` and Livox CustomMsg counts; if the first
    handshake creates topics but no data, it restarts the driver once.
  - Starts robot-side LIO-SAM and checks `/lio_sam_ros2/mapping/odometry`.
    The LIO runtime params are patched to use `livox/imu` and the current Livox
    CustomMsg `lidar_id` as `lidarYsn`; otherwise this robot can show
    `Please check lidar ysn!!!` and publish no odometry.
  - Starts `tmux` windows for YOLO, robot ROS2 odom UDP sender, laptop UDP-to-DDS
    bridge, and odom echo.

- `deploy_dodge_sdk_loco.py`
  - Uses `LocoClient.SetVelocity`, not `rt/lowcmd`.
  - Requires `fsm=200 mode=1 balance=1`.
  - Enforces close-range away motion with `close_escape`.
  - Disables yaw by default (`--max_ang_vel 0.0`) so the camera keeps seeing the person.
  - Sends repeated zero velocity on exit before `StopMove()`.
  - Has software `L2+B` damping after `--l2b_damp_hold 2.0`.
  - Can use external DDS odometry for return with `--return_odom_source external`.
  - Uses LIO geometric closed-loop return by default (`--return_mode geo`).
    The return head checkpoint is now a debugging option, not the default
    real-robot SDK return controller.
  - Locks to the YOLO `track_id` that triggers the first DODGE by default
    (`--lock_first_yolo_track`). Later different IDs are ignored during that
    dodge/return episode, so a second person should not interrupt return. The
    lock is cleared after return completes, after return timeout, or if no
    return is needed, so the next episode can lock a new `track_id`.
  - For external return, the return origin is reset at `DODGE START` so idle LIO
    drift before the person approaches is not counted. Geo return transforms
    odom displacement with live LIO yaw (`rframe=odom_yaw`). The learned online
    command/odom frame is diagnostic only unless a non-geo debug mode is used.
  - Return timeout pauses while external odometry is stale or while another
    person is still in the clear-margin band. It also extends when progress is
    being made and no longer zeroes the residual estimate on timeout.

- `deploy/dodge_policy.py`
  - Can take KF obstacle velocity directly, avoiding fake velocity spikes from sparse YOLO frames.

- `deploy_dodge_real.py`
  - Keeps the low-level `motion.pt` path available, but it is not the recommended path now.

## Known Limits

- Hardware `L2+B` damping hold time is firmware-side; this repo cannot change it.
- Software stops depend on laptop + DDS + SDK RPC still working.
- Accurate return requires `--return_odom_source external` with fresh
  `rt/dodge/odom`; command-integrated return is only a debugging fallback.
- If YOLO drops close to the person, `DODGE STOP dist=infm` can appear. The KF hold
  reduces this, but D435i FOV and close-range depth still limit robustness.

## Useful Commands

Show running sensor/LIO windows and robot-side processes:

```bash
scripts/g1_dodge_stack.sh status
```

Attach logs:

```bash
tmux attach -t g1-dodge
```

Stop perception/LIO/odom processes:

```bash
scripts/g1_dodge_stack.sh stop
```

Manually restart only the MID-360 ROS2 driver and verify live data:

```bash
uv run --with paramiko python scripts/start_lidar_ros2_driver_robot.py \
  --host 192.168.123.164 \
  --ros-domain 42
```

Manually restart only robot-side LIO and verify odometry:

```bash
uv run --with paramiko python scripts/start_lio_mid360_robot.py \
  --host 192.168.123.164 \
  --ros-domain 42 \
  --check
```
