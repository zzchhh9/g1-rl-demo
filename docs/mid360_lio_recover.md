# MID-360 Odometry For SDK Dodge Recover

Goal: keep Unitree's built-in SDK locomotion for walking, and use MID-360
LiDAR-inertial odometry for return-to-start after dodge.

The controller-side interface is now fixed:

```text
ROS2 nav_msgs/Odometry from robot-side LIO
    -> scripts/ros2_odom_udp_sender.py
    -> scripts/udp_odom_to_dds.py
    -> DDS std_msgs/String rt/dodge/odom
    -> deploy_dodge_sdk_loco.py --return_odom_source external
```

## Current Verified Bringup

Use the wrapper first. It starts the robot-side MID-360 ROS2 driver, robot-side
LIO-SAM, the ROS2-to-UDP odom sender, the laptop UDP-to-DDS bridge, YOLO, and an
odom echo window:

```bash
cd /home/zz4723/g1-rl-demo
./start_yolo_lidar.sh
```

The wrapper does not lock YOLO at camera startup by default:

```text
YOLO_LOCK_FIRST_TRACK=0
```

The deploy controller locks the first `track_id` that actually starts DODGE.
That avoids the failure mode where the camera locks a far or stale person while
the robot is still idle. If camera-side locking is needed for a special test,
use a distance gate:

```bash
YOLO_LOCK_FIRST_TRACK=1 YOLO_LOCK_DIST=1.0 ./start_yolo_lidar.sh
```

The Livox driver must run with `xfer_format=1`, because this robot-side
`lio_sam_ros2` consumes Livox `CustomMsg`. PointCloud2 mode (`xfer_format=0`)
causes `lio_sam_ros2_imageProjection` to fail with:

```text
Unknown sensor type: 3
```

The wrapper now checks actual live Livox data. A successful driver check looks
like:

```text
[lidar-ros2] live_check imu=... custom=...
```

A successful LIO check looks like:

```text
[lio] using livox imuTopic=livox/imu lidarYsn=...
[lio] odom_count=... last=(...)
```

On the current G1, the stock `params_mid360.yaml` is not directly usable for
this standalone bringup: it points `imuTopic` at `dog_imu_raw` and contains a
fixed old `lidarYsn`. `scripts/start_lio_mid360_robot.py` patches the runtime
copy under `/tmp/lio-run/params_mid360_runtime.yaml` to:

```text
pointCloudTopic: "livox/lidar"
imuTopic: "livox/imu"
lidarYsn: "<current Livox CustomMsg lidar_id>"
```

The current verified MID-360 `lidar_id` was `192`. If LIO prints
`Please check lidar ysn!!!` or the check ends with `odom_count=0`, this runtime
parameter patch failed or the Livox data stream is stale.

For this G1, the Livox ROS2 driver publishes MID-360 IMU acceleration in units
of `g`, and the sensor z axis is opposite the convention expected by this
LIO-SAM build. `scripts/start_lio_mid360_robot.py` therefore republishes:

```text
/livox/imu -> /livox/imu_scaled
linear_acceleration *= -9.80511
```

and points LIO at `imuTopic: "livox/imu_scaled"`. Without this, the LIO log
repeatedly prints `Large velocity, reset IMU-preintegration!`, and the mapping
odometry can show yaw/position motion while the robot is visually stationary.

For dodge/recover we also force short-horizon odometry settings in the runtime
copy:

```text
useImuHeadingInitialization: false
imuRPYWeight: 0.0
loopClosureEnableFlag: false
```

These avoid using a raw MID-360 IMU as a heading source and prevent loop-closure
pose jumps from contaminating the local return-to-origin controller.

Final DDS check:

```bash
uv run python scripts/dds_odom_echo.py eno1 --topic rt/dodge/odom --duration 10
```

Expected:

```text
[echo] fresh age=0.0xs hz=5.0 n=... src=ros2:/lio_sam_ros2/mapping/odometry
```

## What Is 100% Fixed Now

- `deploy_dodge_sdk_loco.py` can consume external odometry.
- External odometry is fail-closed: if `--return_odom_source external` is used
  and `rt/dodge/odom` is not fresh at enable time, the script raises instead of
  silently using command integration.
- During RETURN, stale external odometry pauses return and sends zero velocity.
- ROS2 odometry is decoupled from the controller through a small DDS JSON bridge,
  so Point-LIO, FAST-LIO2, GLIM, or MOLA can be swapped without changing dodge.

The only remaining timing-dependent work is choosing when to run the physical
test and tuning the LIO launch/config on the real robot floor.

## Why Not Use The Existing Raw UDP Parser

`test_lidar.py` and `nearest_obstacle.py` parse point UDP packets directly. That
is enough for nearest-obstacle distance, but not enough for robust odometry:

- LIO needs both point cloud and MID-360 IMU.
- Livox non-repetitive scans need point timing for motion undistortion.
- Existing raw parser does not publish ROS2 `livox_ros_driver2/CustomMsg` or IMU.

For odometry, run `livox_ros_driver2` continuously and feed a LIO/SLAM stack.

## Recommended LIO Stack

Use a Livox-aware LIO stack first, not plain ICP:

1. Point-LIO ROS2
   - Best first candidate for a walking humanoid because it is designed for
     high-frequency LiDAR-inertial odometry.
   - Requires `livox_ros_driver2` point cloud + IMU.

2. FAST-LIO2 / FAST-LIO ROS2 port
   - Proven Livox LIO family.
   - Make sure it consumes Livox custom messages, not generic point cloud only.

3. GLIM or MOLA
   - Good full SLAM/localization candidates.
   - Heavier setup; useful after simple LIO works.

Plain KISS-ICP is useful only as a quick prototype. It does not use the MID-360
IMU and is less appropriate for G1 walking vibration.

## Start MID-360 ROS2 Driver

This keeps the robot-side Livox ROS2 driver alive on `ROS_DOMAIN_ID=42`.

```bash
cd /home/zz4723/g1-rl-demo
uv run --with paramiko python scripts/start_lidar_ros2_driver_robot.py \
  --host 192.168.123.164 \
  --ros-domain 42
```

Check from a ROS2 shell that can see domain 42:

```bash
source /opt/ros/humble/setup.bash   # or the ROS2 distro on that machine
export ROS_DOMAIN_ID=42
ros2 topic list | grep livox
ros2 topic echo /livox/imu
```

If the laptop cannot see the robot's ROS2 topics, run the LIO stack and odometry
bridge on the robot, or fix ROS2 discovery/networking for domain 42.

## Run LIO/SLAM

Run the selected LIO stack so that it publishes `nav_msgs/Odometry`.

For the current robot-side LIO-SAM setup:

```bash
cd /home/zz4723/g1-rl-demo
uv run --with paramiko python scripts/start_lio_mid360_robot.py \
  --host 192.168.123.164 \
  --ros-domain 42 \
  --check
```

Typical output topic names are one of:

```text
/Odometry
/odom
/aft_mapped_to_init
/state_estimator/odometry
```

Verify the type:

```bash
ros2 topic info /Odometry
```

It must say `nav_msgs/msg/Odometry` for the bridge below.

## Bridge ROS2 Odom To Unitree DDS

The current G1 setup runs ROS2 on the robot, while Unitree DDS runs on the
laptop. Use the UDP relay:

```bash
cd /home/zz4723/g1-rl-demo
PYTHONUNBUFFERED=1 uv run --with paramiko python scripts/ssh_ros2_odom_udp_sender.py \
  --robot-ip 192.168.123.164 \
  --local-ip 192.168.123.222 \
  --udp-port 5070 \
  --ros-topic /lio_sam_ros2/mapping/odometry \
  --print-every 10
```

In another terminal:

```bash
cd /home/zz4723/g1-rl-demo
PYTHONUNBUFFERED=1 uv run python scripts/udp_odom_to_dds.py \
  --net eno1 \
  --udp-port 5070 \
  --dds-topic rt/dodge/odom \
  --print-every 10
```

Use the real topic name if your LIO stack does not publish
`/lio_sam_ros2/mapping/odometry`.

Check DDS output before moving the robot:

```bash
cd /home/zz4723/g1-rl-demo
uv run python scripts/dds_odom_echo.py eno1 --duration 10
```

Expected:

```text
[echo] fresh age=0.02s hz=10.0 n=... x=... y=... yaw=...
```

Move the robot manually a small distance. `x/y` must change smoothly and return
near its previous value if you move the robot back.

## Run SDK Dodge With External Odom Return

Use this only after `dds_odom_echo.py` is fresh.

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

`deploy_dodge_sdk_loco.py` also locks to the YOLO `track_id` that starts the
first DODGE by default. Disable that guard only for debugging:

```bash
./deploy.sh --no_lock_first_yolo_track
```

The same controller command is wrapped by:

```bash
cd /home/zz4723/g1-rl-demo
./deploy.sh
```

Full sensor/perception bringup is wrapped by:

```bash
./start_yolo_lidar.sh
```

Expected startup:

```text
[ODOM-DDS] subscribing to rt/dodge/odom
[ODOM] external origin set at [...]
[ENABLE] SDK loco dodge commands enabled
```

Expected return:

```text
[ODOM] external dodge origin reset at [...]
[DODGE STOP] dist=... est_disp=...
[RETURN START] est_disp=...
[RETURN  ] ... odom=external/fresh(...)
[RETURN DONE] est_disp=...
```

Return commands are body-frame SDK velocities. The default controller is now
`--return_mode geo`: it takes the LIO displacement since `DODGE START`, rotates
it with live LIO yaw, and commands the back-to-origin velocity directly. The
learned SDK-command/LIO frame is diagnostic only in geo mode because it can
become ill-conditioned when dodge commands saturate or are nearly collinear.

When inspecting logs, this is the quick sanity check: `prog_dot`,
`prog_cos`, `actual_dot`, and `actual_cos` should stay positive during return.
If `prog_dot` is positive but `actual_dot` stays non-positive, the command looks
right in the model but the robot is not actually reducing the LIO displacement.

If odometry becomes stale during return:

```text
[RETURN WAIT] external odom stale age=...s; holding zero
```

That is intentional. It prevents blind return.

If a person is detected outside `safety_dist` but still inside
`safety_dist + clear_margin`, return also holds zero velocity. The return timeout
is paused in both stale-odom and blocked-by-person cases, so a second person does
not consume the entire return timeout while the robot is waiting.

Expected blocked-return log:

```text
[RETURN WAIT] obstacle dist=...m inside clear margin; holding zero
```

## Acceptance Test

Before testing with a person:

1. Put the robot in blue mode.
2. Run `./start_yolo_lidar.sh`.
3. Run `dds_odom_echo.py`.
4. Push/move the robot by roughly 0.3 m and verify `x/y` changes by the same
   order of magnitude.
5. Move it back and verify `x/y` returns close to the initial value.
6. Confirm `rt/yolo/person` is publishing at about 9-10 Hz.

Only then run dodge with `--return_odom_source external`.

## Failure Modes

- `./start_yolo_lidar.sh` says `cannot reach 192.168.123.164:22`: the laptop
  cannot SSH to the robot Jetson. Check `ip -br addr show eno1` and
  `ip neigh show 192.168.123.164`. If `eno1` is down or the neighbor is
  `FAILED`, fix the Ethernet/robot network before debugging ROS2 or DDS.
- `start_lidar_ros2_driver_robot.py` shows `live_check imu=0 custom=0`: the
  driver handshook but is not receiving/publishing data. The script retries once.
  If it still fails, stop and restart the sensor stack.
- `start_lio_mid360_robot.py --check` raises `no odometry`: do not run dodge.
  First confirm `/livox/imu` and `/livox/lidar` are publishing; then check that
  `/tmp/lio-run/params_mid360_runtime.yaml` uses `imuTopic: "livox/imu"` and the
  current Livox `lidarYsn`.
- `dds_odom_echo.py` shows no odom: ROS2 bridge is not seeing the LIO topic, or
  DDS is on the wrong network interface.
- `dds_odom_echo.py` is stale: bridge process is alive but LIO stopped
  publishing, or the robot-to-laptop UDP sender has lost the Ethernet route.
- `deploy_dodge_sdk_loco.py` refuses to enable: correct behavior; start odometry
  first.
- Return goes the wrong way: inspect `cmd_odom`, `prog_dot`, `actual_dot`, and
  `rframe` in the `RETURN` log. With `--return_mode geo`, `return_lateral_sign`
  is not applied; if `prog_dot` is positive but `actual_dot` is not, the LIO yaw
  frame or low-level locomotion response is wrong.
- Return magnitude is off but direction is correct: odometry scale or frame
  tracking is drifting; tune LIO, not the locomotion checkpoint.

Recovery commands:

```bash
scripts/g1_dodge_stack.sh stop
./start_yolo_lidar.sh
uv run python scripts/dds_odom_echo.py eno1 --topic rt/dodge/odom --duration 10
```
