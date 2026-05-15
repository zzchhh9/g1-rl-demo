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

Quick start (after the robot is on the same Ethernet at `192.168.123.x`):

```bash
sudo apt install -y sshpass             # one-time
./scripts/start_lidar.sh                # once per robot boot

uv run python test_lidar.py             # verify ~2k pkt/s, ~200k pts/s
uv run python nearest_obstacle.py       # real-time distance readout
./scripts/stop_lidar.sh                 # when done
```

## Real-Robot Dodge Deployment

See [`DEPLOY_REAL_G1.md`](DEPLOY_REAL_G1.md) for the full deployment guide
(locomotion + dodge policy + LiDAR + safety procedure).