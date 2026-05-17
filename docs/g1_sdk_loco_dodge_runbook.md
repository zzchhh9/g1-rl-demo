# G1 SDK Locomotion Dodge Runbook

This is the current recommended real-robot path. It bypasses the
`motion.pt` low-level locomotion policy and sends velocity commands to the
Unitree built-in high-level locomotion controller through
`LocoClient.SetVelocity()`.

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
  --safety_dist 1.5 \
  --clear_margin 0.35 \
  --debug_obs
```

Expected startup:

```text
[loco] ready check fsm=200(code=0) mode=1(code=0) balance=1(code=0)
[ENABLE] SDK loco dodge commands enabled
```

Expected dodge:

```text
[DODGE START] dist=...
[DODGE ] cmd=[...]
[DODGE STOP] dist=... est_disp=...
[RETURN START] est_disp=... est_xy=[...]
[RETURN ] cmd=[...]
[RETURN DONE] est_disp=...
```

The dodge action should have `dir=away` and negative `par`, meaning the velocity
component points away from the person.

## Return Behavior

The SDK path does not expose reliable G1 odometry. Return therefore uses a rough
estimate:

```text
estimated displacement += commanded SDK velocity * dt
```

That is good enough for "step back toward the start", but it is not precise
odometry. Tune only after checking the logs:

```bash
# Return too little:
--return_odom_scale 1.3

# Return too much:
--return_odom_scale 0.7

# Disable return:
--no_return
```

Useful return knobs:

```text
--return_max_vel 0.20
--return_min_vel 0.12
--return_gain 0.8
--return_done_dist 0.10
--return_timeout 5.0
--return_clear_delay 0.5
```

`RETURN` is interrupted immediately if a person enters `safety_dist` again.

## Current Important Implementation Details

- `deploy_dodge_sdk_loco.py`
  - Uses `LocoClient.SetVelocity`, not `rt/lowcmd`.
  - Requires `fsm=200 mode=1 balance=1`.
  - Enforces close-range away motion with `close_escape`.
  - Disables yaw by default (`--max_ang_vel 0.0`) so the camera keeps seeing the person.
  - Sends repeated zero velocity on exit before `StopMove()`.
  - Has software `L2+B` damping after `--l2b_damp_hold 2.0`.

- `deploy/dodge_policy.py`
  - Can take KF obstacle velocity directly, avoiding fake velocity spikes from sparse YOLO frames.

- `deploy_dodge_real.py`
  - Keeps the low-level `motion.pt` path available, but it is not the recommended path now.

## Known Limits

- Hardware `L2+B` damping hold time is firmware-side; this repo cannot change it.
- Software stops depend on laptop + DDS + SDK RPC still working.
- Return is command-integrated dead-reckoning, not true odometry.
- If YOLO drops close to the person, `DODGE STOP dist=infm` can appear. The KF hold
  reduces this, but D435i FOV and close-range depth still limit robustness.
