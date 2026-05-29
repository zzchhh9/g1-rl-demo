# G1 Reinforcement Learning Demo

This repository provides a clean, reproducible, and easy-to-use starter environment for running the **[Unitree G1 Robotic's Reinforcement Learning Demo](https://support.unitree.com/home/en/G1_developer/rl_control_routine)** and the **[Unitree RL Gym project](https://github.com/unitreerobotics/unitree_rl_gym#)**.

The primary goal is to simplify the setup process using the high-performance package manager **[uv](https://docs.astral.sh/uv/)**, allowing you to get the simulation running with minimal operations.

> [!NOTE]
> An NVIDIA graphics card with current drivers is required for Isaac Gym.
> Isaac Sim's pip installation typically requires GLIBC 2.35+, which may not work on Ubuntu 20.04

## Setup Instructions

### 1. Clone the Repository

Clone this repository along with its required submodules. Then, check out the specific version of `rsl_rl` needed for compatibility.

```bash
# Clone the repository and submodules
git clone --recurse-submodules https://github.com/Nagi-ovo/g1-rl-demo.git
cd g1-rl-demo

# Checkout the correct rsl_rl version
pushd third_party/rsl_rl
git checkout v1.0.2
popd
```

### 2. Install Dependencies

#### Install uv
We use `uv` for fast and deterministic dependency management.

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

#### Download Isaac Gym
Download **Isaac Gym Preview 4** from the [NVIDIA Developer website](https://developer.nvidia.com/isaac-gym/download).

Extract the archive and move the **entire `isaacgym` folder** into the `third_party/` directory of this project.

### 3. Install the Python Environment

This single command will create a virtual environment (`.venv`) and install all project dependencies specified in `pyproject.toml`.

```bash
uv sync
```

### 4. (Optional) Verify Isaac Gym Installation

You can confirm that Isaac Gym is installed correctly by running one of its examples.

```bash
pushd third_party/isaacgym/python/examples
uv run 1080_balls_of_solitude.py
popd
```

## Running the Pre-trained Demo

### 1. Place the Policy Model

This project includes a pre-trained policy model (`policy_lstm_1.pt`). Move it to the correct directory for the simulation to find it.

```bash
mv policy_lstm_1.pt third_party/unitree_rl_gym/logs/g1/exported/policies/
```

### 2. Configure the Policy Path

Edit the configuration file at `third_party/unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml` to point to the correct policy file.

```yaml
# Change this:
# policy_path: "{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/g1/motion.pt"

# To this:
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/g1/exported/policies/policy_lstm_1.pt"
```

### 3. Run the Simulation

Launch the simulation using the pre-trained policy.

```bash
uv run third_party/unitree_rl_gym/legged_gym/scripts/play.py --task=g1
```

## LiDAR (Livox Mid-360) — Real Robot

We bypass Unitree's (incomplete) perception stack and pull the Mid-360 point
cloud directly from the robot via UDP. See:

- [`docs/lidar_how_it_works.md`](docs/lidar_how_it_works.md) — how the Mid-360
  works, the wire protocol, and the data path on the G1.
- [`docs/lidar_nearest_obstacle.md`](docs/lidar_nearest_obstacle.md) — using the
  point cloud to measure the nearest obstacle in real time.
- [`docs/mid360_lio_recover.md`](docs/mid360_lio_recover.md) — using MID-360
  LiDAR odometry for SDK dodge return-to-start.

Quick start (after the robot is on the same Ethernet at `192.168.123.x`):

```bash
sudo apt install -y sshpass             # one-time
./scripts/start_lidar.sh                # once per robot boot

uv run python test_lidar.py             # verify ~2k pkt/s, ~200k pts/s
uv run python nearest_obstacle.py       # real-time distance readout
./scripts/stop_lidar.sh                 # when done
```

## G1 Dodge-and-Recover Deployment

Real-robot dodge controller for the Unitree G1. The robot stands in Unitree's
blue high-level locomotion mode, watches for an approaching person (camera+YOLO,
or a faked obstacle), **dodges** sideways out of the way, then **recovers** by
walking back to its start pose using MID-360 LiDAR odometry.

The controller is `deploy_dodge_sdk_loco.py`. It is driven through
`scripts/g1_dodge_stack.sh`, which brings up the perception stack (RGBD camera,
YOLO bridge, MID-360 ROS2 driver, FAST-LIO odometry, the DDS odom/YOLO bridges)
and then launches the controller in the foreground, recording odometry to a
timestamped `runs/<RUN_ID>/` directory.

> [!IMPORTANT]
> Put the G1 in **blue (high-level) locomotion mode** with the remote before
> deploying. The controller drives `LocoClient.SetVelocity`, not the low-level
> `motion.pt` policy. Have the E-stop ready.

---

### 1. Quick start

Wrapper scripts (all live at repo root) layer on top of `g1_dodge_stack.sh`:

| Script | What it runs |
| --- | --- |
| `./deploy.sh` | `g1_dodge_stack.sh deploy` — real camera + YOLO, geo return. The production path. |
| `./deploy_fake.sh` | Same stack, but `YOLO_SOURCE=fake` — one scripted "person" crossing instead of the camera. |
| `./run_gated_fake_baseline.sh` | Confirmed gated-return baseline (one fake crossing, `--return_mode gated --return_gated_lin_vel 0.35`). |
| `./run_gated_fake_seq5_front.sh` | 6-pass fake sequence, obstacle only from the **front 180°** (`FAKE_YOLO_HEADING_MIN/MAX=90/270`), gated return. |
| `scripts/g1_dodge_stack.sh {sensors,deploy,status,stop}` | The underlying stack manager. |

`AUTO_START_SENSORS_BEFORE_DEPLOY=1` (default) means a single command starts
sensors, waits for fresh odom + a YOLO heartbeat, then runs the controller:

```bash
# Real camera, analytic (geo) return — the normal path:
./deploy.sh

# Fake obstacle, learned (gated) return — no camera needed:
./run_gated_fake_baseline.sh

# Front-hemisphere-only fake sequence, 6 passes:
./run_gated_fake_seq5_front.sh
```

Any extra args are forwarded straight to `deploy_dodge_sdk_loco.py`, e.g.:

```bash
./deploy.sh --return_mode gated --return_done_dist 0.08 --max_dodge_dist 1.5
```

Manual three-window flow if you prefer to drive it yourself:

```bash
scripts/g1_dodge_stack.sh sensors     # camera + YOLO + MID-360 + LIO + odom bridge
scripts/g1_dodge_stack.sh status      # tmux windows + robot-side lidar processes
scripts/g1_dodge_stack.sh deploy --return_mode geo
scripts/g1_dodge_stack.sh stop        # tear everything down
```

---

### 2. The dodge controls

These shape **how the robot escapes** before any recover starts. Defaults are
the in-code `deploy_dodge_sdk_loco.py` defaults.

| Flag | Default | What it does |
| --- | --- | --- |
| `--dodge_dir_latch` / `--no_dodge_dir_latch` | latch **on** | Latch the lateral escape side at dodge start. A crossing obstacle can otherwise make the dodge wag left↔right; the net displacement cancels and the online return-frame fit starves, which is what makes the recover walk the wrong way. `--no_dodge_dir_latch` allows the side to switch mid-dodge. |
| `--dodge_latch_deadband` | `0.10` | Lateral command magnitude (m/s) needed to commit the latched dodge side. |
| `--min_dodge_dist` | `0.5` | Keep escaping in the latched direction until the dodge has moved at least this far from origin (m), so every dodge is big enough for a clean return-frame fit. `<=0` disables. |
| `--min_dodge_time` | `2.5` | Max seconds spent reaching `--min_dodge_dist` before the return is allowed regardless. |
| `--max_dodge_dist` | `2.0` | Cap on dodge displacement from origin (m). At/above this the dodge holds position instead of running further away. `<=0` disables the cap. |
| `--max_vel` | `0.30` | Per-axis dodge velocity cap (m/s). |
| `--safety_dist` | `1.0` | Obstacle distance (m) that triggers the dodge; dodge stops once the obstacle is back outside it. |
| `--clear_margin` | `0.00` | Extra clearance (m) added on top of `--safety_dist` before the obstacle counts as cleared. |

The latch + `min_dodge_dist`/`min_dodge_time` combination is the key to a clean
recover: it guarantees a single-sided, sufficiently large escape so the online
return-frame fit has real displacement to work with.

---

### 3. Return / recover modes

After the obstacle clears (held clear for `--return_clear_delay`, default `1.0`s)
and the robot settles, the controller walks back toward the dodge origin.

Pick the mode with `--return_mode`:

| Mode | What it is |
| --- | --- |
| `geo` (default) | Analytic. Uses SLAM displacement plus live LIO yaw to command directly toward the dodge origin. Hardware-validated. |
| `gated` | Learned `return_head` (6-dim MLP) from `--gated_ckpt` (`checkpoints/v23b_gated_return.pt`). |
| `head` | Trained return head from `checkpoints/return_head_v23b_v6.pt`. Replay/debug only — the stack forces it back to `geo` for real deploys unless `ALLOW_RETURN_HEAD=1`. |
| `p` | Legacy hand-written P controller. |

Geo-mode return tuning (defaults):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--return_gain` | `0.8` | P gain, estimated displacement → forward return velocity. |
| `--return_lat_gain` | `0.35` | Lateral P gain. Keep below `--return_gain`; SDK lateral tracking is less stable. |
| `--return_max_vel` | `0.25` | Per-axis return velocity cap. |
| `--return_min_vel` | `0.12` | Minimum return command norm while displacement is still above `--return_done_dist`. |
| `--return_max_lat_vel` | `0.08` | Lateral return cap. |
| `--return_yaw_source` | `fused_lowstate` | Yaw source for world→SDK conversion. `fused_lowstate` holds SLAM yaw at the dodge origin and applies lowstate's relative yaw change (most robust against LIO yaw drift); also `slam`, `lowstate`. |
| `--return_probe` / `--no_return_probe` | probe **on** | Measure the SDK +x/+y directions with short SLAM-observed probe motions before returning. Disable to use LIO yaw directly. |

Gated-mode return tuning:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--gated_ckpt` | `checkpoints/v23b_gated_return.pt` | Learned return head weights. |
| `--return_gated_lin_vel` | `0.5` | Linear velocity scale (m/s) for the gated XY action. `0.5` matches training; the validated runs use `0.35` to match geo return speed. |
| `--return_gated_yaw_sign` | `-1.0` | Sign on the heading error fed to the gated head's yaw input. `-1` matches geo; flip to `+1` if return yaw rotates the wrong way. |

The validated gated baseline (see `run_gated_fake_baseline.sh`) is
`--return_mode gated --return_gated_lin_vel 0.35 --no_exit_on_return_abort`,
which recovered to **0.09 m** in runs `rhea_20260528_233107` and
`rhea_20260528_235319`.

---

### 4. Fake-YOLO testing & the front-180 sequence

Set `YOLO_SOURCE` to swap the obstacle source (`g1_dodge_stack.sh`):

| `YOLO_SOURCE` | Behavior |
| --- | --- |
| `real` (default) | Live camera + YOLO person tracker. |
| `fake` | One scripted crossing — a single dodge → return. |
| `sequence` | `FAKE_YOLO_MAX_PASSES` scripted crossings back to back; each is stand → cross → dodge → recover → gap → next. |

Fake / sequence env knobs (set before the runner script):

| Env var | Default | Purpose |
| --- | --- | --- |
| `FAKE_YOLO_MAX_PASSES` | `5` | Number of crossings in `sequence` mode. |
| `FAKE_YOLO_START_DELAY` | `15.0` | Seconds before the first crossing. |
| `FAKE_YOLO_POST_GAP` | `5.0` | Seconds after each recover before the next pass. |
| `FAKE_YOLO_HEADING_MIN` | `0` | Lower bound of the crossing heading range (deg). |
| `FAKE_YOLO_HEADING_MAX` | `360` | Upper bound of the crossing heading range (deg). |
| `FAKE_YOLO_START_DIST` / `FAKE_YOLO_END_DIST` | `0.50` / `1.60` | Obstacle near/far range (m). |
| `FAKE_YOLO_BEARING_DEG` | `0.0` | Fixed bearing for single `fake` mode. |

**Front-180 constraint.** Heading `90..270` makes the "person" approach only
from the robot's **front hemisphere** (enter bearing −90° right … 0 front …
+90° left), never from behind. That is exactly what
`run_gated_fake_seq5_front.sh` sets:

```bash
YOLO_SOURCE=sequence \
FAKE_YOLO_MAX_PASSES=6 \
FAKE_YOLO_HEADING_MIN=90 \
FAKE_YOLO_HEADING_MAX=270 \
./deploy.sh --return_mode gated --return_gated_lin_vel 0.35 --no_exit_on_return_abort
```

`START_RGBD=0` is set there too: the fake/sequence sources don't need the camera.

---

### 5. Tuning recover precision

How close the robot lands to its origin, and how it handles a return that starts
heading the wrong way.

**Success radius — `--return_done_dist` (default `0.10`).**
Estimated displacement below this counts as "returned." Lower it for a tighter
landing (the validated runs reached ~0.09 m); raise it if SLAM jitter makes the
robot oscillate near the goal without latching done.

**Time budget — `--return_timeout` (default `15.0`).**
Max seconds in one RETURN phase. Related guards:

- `--return_no_progress_timeout` (`3.0`) — abort if real LIO displacement makes
  no positive progress toward origin for this long.
- `--return_bad_progress_abort_count` (`3`) — abort after this many fresh odom
  updates move clearly away from origin (`<=0` disables).
- `--return_settle_max_wait` (`4.0`) — max extra seconds holding zero while
  waiting for the settle gate before giving up on the attempt.

**Frame-flip recovery — `--return_max_frame_flips` (default `1`).**
When the return is detected walking **away** from origin, the online return
frame was almost always fit ~180° backwards (det=+1 but reversed). Instead of
aborting, the controller flips the frame 180° (negates `disp_b`) and retries up
to this many times. Set `0` for the old behavior (abort immediately on a
wrong-way return).

**Settle gate before return** (both must pass over `--return_stationary_time`,
default `1.00`s): `--return_stationary_disp` (`0.03` m net displacement) and
`--return_stationary_speed` (`0.03` m/s average SLAM speed).

**Abort behavior — `--exit_on_return_abort` / `--no_exit_on_return_abort`.**
Default exits the controller on an unrecoverable abort so the final `StopMove`
runs immediately. The fake/sequence runners use `--no_exit_on_return_abort` to
stay in READY and continue to the next pass instead of quitting.

Practical recipe for a tight, robust recover:

```bash
./deploy.sh \
  --return_mode geo \
  --return_done_dist 0.08 \
  --return_timeout 20 \
  --return_max_frame_flips 1 \
  --max_dodge_dist 1.5 --min_dodge_dist 0.5
```
