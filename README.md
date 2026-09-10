# G1 Dodge / LiDAR / LIO 真机栈

Unitree G1 **真机躲避 + 回位** 仓库。当前主路径不是 Isaac Gym 训练 starter，而是：

- 机器人停在 Unitree **蓝色高层 locomotion**（遥控器 `L2 + UP`）
- 笔记本通过 `LocoClient.SetVelocity` 发速度（`deploy_dodge_sdk_loco.py`）
- RealSense + YOLO 看人（DDS `rt/yolo/person`）
- 头顶 **Livox Mid-360** + 机器人侧 **FAST-LIO / LIO-SAM** 给出里程计
- 里程计经 UDP 桥到 DDS `rt/dodge/odom`，供 dodge 后 **走回起点**

详细操作手册：[docs/g1_sdk_loco_dodge_runbook.md](docs/g1_sdk_loco_dodge_runbook.md)。文档目录：[docs/README.md](docs/README.md)。

> **不要**把 `deploy_dodge_real.py` / `motion.pt` 低层策略当成现在的默认。那条路径还在仓库里，但真机推荐栈已经换成 SDK 高层 loco。

---

## 1. 这是什么 / What this repo is now

| 角色 | 现在用什么 |
| --- | --- |
| 走路 | Unitree 内置高层 locomotion（蓝灯），`LocoClient.SetVelocity` |
| 看人 | 机器人 RealSense TCP → 笔记本 YOLO → `rt/yolo/person` |
| 回位 | Mid-360 LIO 位移 + yaw → `--return_mode geo`（默认） |
| 编排 | `scripts/g1_dodge_stack.sh`（`./deploy.sh` / `./start_yolo_lidar.sh` 都是它的包装） |
| 记录 | 每次 deploy 写到 `runs/<RUN_ID>/`（odom jsonl、log、可选 mp4） |

数据流（dodge 生产路径）：

```text
RealSense  → YOLO  → DDS rt/yolo/person
Mid-360    → 机器人 livox_ros_driver2 (xfer_format=1)
           → FAST-LIO /Odometry   或   LIO-SAM /lio_sam_ros2/mapping/odometry
           → 机器人 UDP sender :5070
           → 笔记本 udp_odom_to_dds.py
           → DDS rt/dodge/odom
deploy_dodge_sdk_loco.py → LocoClient.SetVelocity
```

默认 SLAM 后端是 **FAST-LIO**（`SLAM_BACKEND=fast_lio`，ROS topic `/Odometry`）。建图脚本走 **LIO-SAM**。两条后端最终都桥到同一个 DDS topic `rt/dodge/odom`。

---

## 2. 网络与机器 / Network

脚本默认值（可用环境变量覆盖）：

| 项 | 默认 |
| --- | --- |
| 笔记本网卡 | `eno1`（`NET`） |
| 笔记本 IP | `192.168.123.222` |
| 机器人 Jetson | `192.168.123.164`（SSH `unitree` / `123`） |
| Mid-360 | `192.168.123.120` |
| ROS domain（机器人侧） | `42` |
| DDS odom | `rt/dodge/odom` |
| DDS YOLO | `rt/yolo/person` |

连通性自检：

```bash
ip -br addr show eno1
ip route get 192.168.123.164
ip neigh show 192.168.123.164
ping -c 2 192.168.123.164
```

好的状态：`eno1 UP ... 192.168.123.222/24`，邻居 `REACHABLE`。`eno1 DOWN` 或邻居 `FAILED` 时先修网线 / Jetson 网络，不要先调 ROS/DDS。

`scripts/g1_dodge_stack.sh` 和 `./start_lidar_mapping.sh` 默认会尝试把 `eno1` 配到 `192.168.123.222/24`（`AUTO_FIX_NET=1`）。

---

## 3. 环境 / Laptop setup

```bash
cd /path/to/g1-rl-demo
# 若刚 clone：git submodule update --init --recursive

# uv：https://docs.astral.sh/uv/
wget -qO- https://astral.sh/uv/install.sh | sh

# 真机 deploy 不需要完整 Isaac Gym。若 uv sync 因 isaacgym 失败：
mkdir -p third_party/isaacgym/python && \
  echo -e '[project]\nname="isaacgym"\nversion="0.0.0"' > third_party/isaacgym/python/pyproject.toml

uv sync
sudo apt install -y tmux sshpass
```

Checkpoint 已在仓库：`checkpoints/dodge_v23b_54400.pt`、`checkpoints/v23b_gated_return.pt`、`checkpoints/return_head_v23b_v6.pt`。

Isaac Gym / `unitree_rl_gym` 仿真见文末 [附录 A](#附录-a-isaac-gym--unitree_rl_gym-仿真可选)。

---

## 4. 快速上手 / Quickstart（已验证主流程）

每次真机前：急停遥控器在手、工作区清空、电量够、网通。

```bash
# 1) 遥操：把 G1 打进蓝色高层 locomotion（见下一节）
#    遥控器：按住 L2 + UP，直到手柄指示灯变蓝
#    目标状态：fsm=200  mode=1  balance=1

# 2) 另开一个终端，软件急停随时可用
uv run python scripts/g1_emergency_stop.py eno1   # 先别跑；需要时再跑

# 3) 生产路径：相机 + YOLO + Mid-360 + LIO + dodge（geo 回位）
./deploy.sh
```

`./deploy.sh` = `scripts/g1_dodge_stack.sh deploy`。默认 `AUTO_START_SENSORS_BEFORE_DEPLOY=1`：先起感知，等到 `rt/dodge/odom` 和 `rt/yolo/person` 新鲜，再在当前终端跑控制器。

分步（自己盯 log）：

```bash
./start_yolo_lidar.sh                 # = g1_dodge_stack.sh sensors
uv run python scripts/dds_odom_echo.py eno1 --topic rt/dodge/odom --duration 10
uv run python scripts/dds_yolo_echo.py eno1 --duration 5
./deploy.sh                           # 或 scripts/g1_dodge_stack.sh deploy
scripts/g1_dodge_stack.sh status
scripts/g1_dodge_stack.sh stop
```

Odom 正常时应类似：

```text
[echo] fresh age=0.0xs hz=... n=... src=ros2:/Odometry
```

（LIO-SAM 时 `src=ros2:/lio_sam_ros2/mapping/odometry`。）`no odom` / `STALE` **不要**上 dodge。

无人在镜头前：`rt/yolo/person` 大约 9–10 Hz 且 `{"n": 0}`。有人：

```text
[echo] fresh ... n=1 dist=+1.xx x=+1.xx ...
```

无相机、假人穿越（不需要 RealSense）：

```bash
./deploy_fake.sh
# 已验证 gated 回位基线（约 0.09 m）：
./run_gated_fake_baseline.sh
```

日志：`tmux attach -t g1-dodge`。Web 轨迹：`http://127.0.0.1:8765`。

---

## 5. 遥操 / Remote teleop

Dodge **不**用低层 `rt/lowcmd` 走路。操作员要做的是：用遥控器把机器人送进 **蓝色高层 locomotion**，之后速度由笔记本 SDK 发。仓库里写到的按键如下（**不要发明其它组合**）。

### 5.1 进入蓝色高层 locomotion（dodge 前置）

来源：[`deploy_dodge_sdk_loco.py`](deploy_dodge_sdk_loco.py) 文件头、[docs/g1_sdk_loco_dodge_runbook.md](docs/g1_sdk_loco_dodge_runbook.md)。

1. 机器人站稳、吊装/防摔按现场规程处理。
2. 遥控器：**按住 `L2 + UP`，直到手柄指示灯变蓝**。
3. 脚本期望 SDK 状态变成：

   ```text
   fsm=200  mode=1  balance=1
   ```

4. 控制器启动时会调 `SetFsmId(200)` 和 `SetBalanceMode(1)`，并等到这个稳态再发 dodge 速度。
5. 默认 **立刻使能**（不等 `A`）。内置高层模式会吃掉 `A`，所以默认 `--wait_for_a` 是关的。只有显式传入 `--wait_for_a` 时才需要再按 **A** 才开始发速度。

启动成功日志应包含：

```text
[loco] ready check fsm=200(code=0) mode=1(code=0) balance=1(code=0)
[ENABLE] SDK loco dodge commands enabled
```

还没进蓝灯就跑 deploy 时，脚本会提示：

```text
Hold L2 + UP on the remote until the controller light is blue.
```

蓝灯模式下 Unitree 固件自己的摇杆走路仍可能存在；**本仓库没有单独记录摇杆映射**，dodge 使能后以 `SetVelocity` 为准。不要同时用手杆和脚本抢速度。

### 5.2 急停 / E-stop（分层，都要会）

| 层级 | 动作 | 说明 |
| --- | --- | --- |
| 固件阻尼（最终兜底） | **先按住 `L2`，再按住 `B`，一直按到 G1 进入 damping** | 这台 G1 上大约 **5 秒**，固件侧阈值，仓库改不了 |
| 脚本内软件阻尼 | dodge 脚本在跑时，**按住 `L2+B` 约 2 秒** | `--l2b_damp_hold 2.0`：发零速 → `Damp()` → 退出 |
| 脚本内 SELECT | 按 **SELECT** | 立刻 `Damp()` 并退出 |
| 另一终端软件停速 | `uv run python scripts/g1_emergency_stop.py eno1` | 只对高层 `SetVelocity` 连发零速；**代替不了**物理急停 |
| 可选看门狗 | `uv run python scripts/g1_l2b_damp_watchdog.py eno1` | 监听 `rt/lowstate` 上的 `L2+B`，到时 `Damp()` |

软件停依赖笔记本 + 网线 + DDS 还活着。网断了只剩固件 `L2`→`B` 阻尼和断电。

独立看门狗（dodge 脚本没在跑、但仍要遥控器阻尼时）：

```bash
uv run python scripts/g1_l2b_damp_watchdog.py eno1
```

### 5.3 建议的操作顺序

1. 网通、`ping 192.168.123.164`。
2. 急停遥控器电池、人就位；需要时先开 `g1_emergency_stop.py` 那个终端。
3. **`L2 + UP` 直到蓝灯**。
4. `./start_yolo_lidar.sh`（或直接 `./deploy.sh` 自动起传感器）。
5. `dds_odom_echo.py` 确认 fresh；YOLO 有心跳。
6. 再跑 / 等控制器 `[ENABLE]`。
7. 异常：SELECT，或 `L2` 再按住 `B` 直到 damping。然后 `scripts/g1_dodge_stack.sh stop`。

### 5.4 不是这条路径的遥控器（低层 `motion.pt`）

[`deploy_dodge_real.py`](deploy_dodge_real.py) / `unitree_rl_gym` `deploy_real.py` 是 **低层 `rt/lowcmd` + `motion.pt`**，按键不同：

| 键 | 低层路径含义 |
| --- | --- |
| `L2+R2` | 零力矩下进入调试/阻尼（unitree_rl_gym 文档） |
| **START** | 退出零力矩，过渡到默认站姿 |
| **A** | 开始行走 / dodge |
| 左摇杆 | 前/后、左/右侧移 |
| 右摇杆 | 转向 |
| **SELECT** | 紧急停止 |

这不是现在的推荐 dodge 路径。步骤见 [DEPLOY_REAL_G1.md](DEPLOY_REAL_G1.md) 和 [附录 B](#附录-b-低层-motionpt-路径不推荐)。

---

## 6. SLAM / LIO / Mapping

Dodge 回位吃的是 **LiDAR-inertial odometry 的位移**，不是 `start_lidar.sh` 那条裸 UDP 点云。两条 LiDAR 通路不要混用。

### 6.1 两条 LiDAR 通路

| 通路 | 脚本 | 用途 |
| --- | --- | --- |
| **ROS2 + LIO**（dodge / 建图） | `scripts/start_lidar_ros2_driver_robot.py` 保持 `livox_ros_driver2` 活着，`xfer_format=1`（Livox `CustomMsg`） | FAST-LIO 或 LIO-SAM |
| **裸 UDP 转发**（测距工具） | `scripts/start_lidar.sh` / `scripts/stop_lidar.sh` | 握手后 **杀掉** driver，只把 `:56301/:56401/:56201` 转到笔记本；给 `test_lidar.py`、`nearest_obstacle.py` |

建图 / dodge **必须**走第一行。`xfer_format=0`（PointCloud2）会让这台机器人上的 `lio_sam_ros2_imageProjection` 报 `Unknown sensor type: 3`。

原理与协议：[docs/lidar_how_it_works.md](docs/lidar_how_it_works.md)。回位如何用 odom：[docs/mid360_lio_recover.md](docs/mid360_lio_recover.md)。最近障碍：[docs/lidar_nearest_obstacle.md](docs/lidar_nearest_obstacle.md)。

### 6.2 Dodge 栈里的 LIO（默认 FAST-LIO）

`./start_yolo_lidar.sh` / `./deploy.sh` 会：

1. （可选）重启机器人 RGBD `:5005`
2. 起机器人侧 Mid-360 ROS2 driver（`ROS_DOMAIN_ID=42`）
3. 按 `SLAM_BACKEND` 起 LIO
4. `scripts/ssh_ros2_odom_udp_sender.py`：机器人 ROS2 odom → 笔记本 UDP `:5070`
5. `scripts/udp_odom_to_dds.py`：UDP → DDS `rt/dodge/odom`

| `SLAM_BACKEND` | 机器人脚本 | ROS odom topic | 备注 |
| --- | --- | --- | --- |
| `fast_lio`（**默认**） | `scripts/start_fast_lio_mid360_robot.py` | `/Odometry` | 期望机器人已编译 Ericsii/FAST_LIO_ROS2 于 `/home/unitree/fastlio2_ws` |
| `lio_sam` | `scripts/start_lio_mid360_robot.py` | `/lio_sam_ros2/mapping/odometry` | 运行时会改 `params_mid360_runtime.yaml`（IMU topic、`lidarYsn`） |

切 LIO-SAM：

```bash
SLAM_BACKEND=lio_sam ./deploy.sh
```

手动只重启某一段：

```bash
uv run --with paramiko python scripts/start_lidar_ros2_driver_robot.py \
  --host 192.168.123.164 --ros-domain 42

uv run --with paramiko python scripts/start_fast_lio_mid360_robot.py \
  --host 192.168.123.164 --ros-domain 42 --check

uv run --with paramiko python scripts/start_lio_mid360_robot.py \
  --host 192.168.123.164 --ros-domain 42 --check
```

换本地 LIO 时设 `LIO_CMD='ros2 launch ...'` 并改 `ROS_ODOM_TOPIC`。桥接不改，控制器始终订 `rt/dodge/odom`。

### 6.3 只建图（LIO-SAM + 存 PCD）

不跑 YOLO / dodge，只绕场走、存图：

```bash
./start_lidar_mapping.sh
# 慢走 / 推机器人绕测试区
uv run python scripts/dds_odom_echo.py eno1 --topic rt/dodge/odom --duration 10
tmux attach -t g1-lidar-map

# 停的时候 LIO 先收 SIGINT，给时间把 PCD flush 到机器人 /tmp/lio-run/pcd/<MAP_NAME>/
MAP_DIR='...' ./stop_lidar_mapping.sh    # 脚本结束时会打印实际 MAP_DIR
```

`start_lidar_mapping.sh` 会起 driver + LIO-SAM（`--save-pcd --reset-map-dir`）以及 odom UDP/DDS/echo 的 tmux（session `g1-lidar-map`）。默认地图目录：`/tmp/lio-run/pcd/map_YYYYMMDD_HHMMSS`（在 **机器人** 上，重启会丢）。

Dodge 栈里 LIO-SAM 也可存图：`SAVE_LIO_MAP=1`（默认开），目录 `MAP_DIR`（默认 `/tmp/lio-run/pcd/$MAP_NAME`）。FAST-LIO 默认不做这套 PCD 建图。

### 6.4 Odom 桥与 dodge 怎么用 SLAM

```text
机器人 nav_msgs/Odometry
  → scripts/ssh_ros2_odom_udp_sender.py   (ROS_ODOM_TOPIC)
  → UDP :5070
  → scripts/udp_odom_to_dds.py
  → DDS std_msgs/String  rt/dodge/odom
  → deploy_dodge_sdk_loco.py --return_odom_source external
```

控制器默认 `--return_odom_source external`：使能时 `rt/dodge/odom` 不新鲜会直接报错，不会偷偷改用指令积分。RETURN 期间 odom 过期会停速并打：

```text
[RETURN WAIT] external odom stale age=...s; holding zero
```

Geo 回位：从 `DODGE START` 起的 LIO 位移 + 当前 yaw → 指向原点的 `SetVelocity`。原点在 dodge 开始时重置，避免 idle 漂移算进回位。

上人之前的验收（[docs/mid360_lio_recover.md](docs/mid360_lio_recover.md)）：

1. 蓝灯模式
2. `./start_yolo_lidar.sh`
3. `dds_odom_echo.py` fresh
4. 把机器人挪约 0.3 m，`x/y` 同量级变化；挪回去应接近原值
5. `rt/yolo/person` 约 9–10 Hz

LIO-SAM 运行时注意（机器人 `/tmp/lio-run/params_mid360_runtime.yaml`）：

- `imuTopic: livox/imu_scaled`（Livox 加速度按 *g* 且 z 轴相反，脚本会 `* -9.80511` 转发）
- `lidarYsn` 必须等于当前 CustomMsg `lidar_id`（否则 `Please check lidar ysn!!!`、`odom_count=0`）
- dodge 短时域：`useImuHeadingInitialization: false`、`imuRPYWeight: 0.0`、`loopClosureEnableFlag: false`

### 6.5 裸 UDP 点云（不是 SLAM）

机器人重启后若只要点云测距：

```bash
./scripts/start_lidar.sh
uv run python test_lidar.py          # ~2k pkt/s, ~200k pts/s
uv run python nearest_obstacle.py
./scripts/stop_lidar.sh
```

链路：Mid-360 `.120` → Jetson `.164` → UDP forward → 笔记本 `.222`。机器人文件在 `/tmp/livox-run/`，重启即清。

---

## 7. Dodge 控制与回位（摘要）

完整 flag 以 `uv run python deploy_dodge_sdk_loco.py -h` 为准。`./deploy.sh` 后面的参数会原样转给控制器。

### 7.1 躲避

| Flag | 默认 | 作用 |
| --- | --- | --- |
| `--dodge_dir_latch` / `--no_dodge_dir_latch` | latch **on** | 锁定侧移方向，避免左右对消、回位拟合饿死 |
| `--min_dodge_dist` | `0.5` | 至少逃这么远（m）才允许回位；`<=0` 关闭 |
| `--min_dodge_time` | `2.5` | 达不到 min dist 时最多等这么久（s） |
| `--max_dodge_dist` | `2.0` | 位移上限（m） |
| `--max_vel` | `0.30` | 躲避每轴速度上限（m/s） |
| `--safety_dist` | `1.0` | 触发躲避的障碍距离（m） |
| `--clear_margin` | `0.00` | 出安全距离后再加这点才算清除 |

### 7.2 回位 `--return_mode`

| Mode | 含义 |
| --- | --- |
| `geo`（默认） | 解析：SLAM 位移 + yaw，硬件验证过 |
| `gated` | `checkpoints/v23b_gated_return.pt`；假人基线用 `--return_gated_lin_vel 0.35` |
| `head` | `return_head_v23b_v6.pt`，回放/对比；真机栈会强制改回 `geo`，除非 `ALLOW_RETURN_HEAD=1` |
| `p` | 手写 P 控制 |

Geo 常用：`--return_gain 0.8`、`--return_lat_gain 0.35`、`--return_max_vel 0.25`、`--return_done_dist 0.10`、`--return_timeout 15`、`--return_yaw_source fused_lowstate`。

紧回位示例：

```bash
./deploy.sh \
  --return_mode geo \
  --return_done_dist 0.08 \
  --return_timeout 20 \
  --return_max_frame_flips 1 \
  --max_dodge_dist 1.5 --min_dodge_dist 0.5
```

更细的 abort / settle / frame-flip 见旧 README 逻辑，仍以脚本帮助和 [docs/g1_sdk_loco_dodge_runbook.md](docs/g1_sdk_loco_dodge_runbook.md) 为准。

### 7.3 假人 YOLO

| `YOLO_SOURCE` | 行为 |
| --- | --- |
| `real`（默认） | 相机 + YOLO |
| `fake` | 一次脚本穿越 → 一次 dodge→return |
| `sequence` | `FAKE_YOLO_MAX_PASSES` 次；每次 stand → cross → dodge → recover |

前半球 6 次（heading 90°..270°）：

```bash
./run_gated_fake_seq5_front.sh
```

---

## 8. 常用命令

```bash
scripts/g1_dodge_stack.sh sensors|deploy|status|stop
./start_yolo_lidar.sh          # sensors
./deploy.sh                    # 真相机
./deploy_fake.sh               # 假人
./start_lidar_mapping.sh       # 只建图（LIO-SAM）
./stop_lidar_mapping.sh

uv run python scripts/dds_odom_echo.py eno1 --topic rt/dodge/odom --duration 10
uv run python scripts/dds_yolo_echo.py eno1 --duration 5
uv run python scripts/g1_emergency_stop.py eno1
```

Odom / YOLO 坏了：

```bash
scripts/g1_dodge_stack.sh stop
./start_yolo_lidar.sh
```

YOLO 窗口卡在旧 track（`n=0 ... locked=True`）时，只重启 YOLO、不要动 LIO：

```bash
START_RGBD=0 START_LIDAR_DRIVER=0 START_LIO=0 \
START_ODOM_BRIDGE=0 START_ODOM_ECHO=0 \
YOLO_LOCK_FIRST_TRACK=0 ./start_yolo_lidar.sh
```

默认 `YOLO_LOCK_FIRST_TRACK=0`：由控制器锁定真正触发 DODGE 的 `track_id`。

---

## 9. 文档地图

| 文档 | 内容 |
| --- | --- |
| [docs/README.md](docs/README.md) | 索引 |
| [docs/g1_sdk_loco_dodge_runbook.md](docs/g1_sdk_loco_dodge_runbook.md) | **当前推荐**真机 runbook |
| [docs/mid360_lio_recover.md](docs/mid360_lio_recover.md) | LIO 回位、桥接、验收 |
| [docs/lidar_how_it_works.md](docs/lidar_how_it_works.md) | Mid-360 协议与 UDP 通路 |
| [docs/lidar_nearest_obstacle.md](docs/lidar_nearest_obstacle.md) | 裸点云最近障碍 |
| [DEPLOY_REAL_G1.md](DEPLOY_REAL_G1.md) | 低层 `motion.pt` / sim2sim / 早期 LiDAR 工具（非默认） |
| [DEPLOY_PLAN.md](DEPLOY_PLAN.md) | 早期执行计划（历史） |

---

## 附录 A. Isaac Gym / unitree_rl_gym 仿真（可选）

需要 NVIDIA 驱动。Isaac Sim pip 通常要 GLIBC 2.35+，Ubuntu 20.04 可能不行。

```bash
# third_party/rsl_rl 需要 v1.0.2
pushd third_party/rsl_rl && git checkout v1.0.2 && popd

# 从 NVIDIA 下载 Isaac Gym Preview 4，整个 isaacgym 目录放到 third_party/
uv sync

mv policy_lstm_1.pt third_party/unitree_rl_gym/logs/g1/exported/policies/
# 按需改 third_party/unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml 的 policy_path

uv run third_party/unitree_rl_gym/legged_gym/scripts/play.py --task=g1
```

本仓库的 dodge sim2sim：

```bash
MUJOCO_GL=egl uv run python deploy_dodge_mujoco.py --duration 30 --obstacle_speed 0.30
```

应看到 `IDLE → DODGE → RETURN → STOP`。

---

## 附录 B. 低层 motion.pt 路径（不推荐）

```bash
uv run python third_party/unitree_rl_gym/deploy/deploy_real/deploy_real.py eno1 configs/g1.yaml
# 或
uv run python deploy_dodge_real.py eno1 configs/g1.yaml
```

遥控器：**START** 站起 → **A** 走路 → **SELECT** 急停。详见 [DEPLOY_REAL_G1.md](DEPLOY_REAL_G1.md)。真机躲避请走 SDK 蓝灯 + `./deploy.sh`。
