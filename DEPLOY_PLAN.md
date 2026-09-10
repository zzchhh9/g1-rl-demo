# G1 真机部署计划

> **状态**：这是 2026-05 的执行计划，主路径后来换成了 SDK 高层 loco dodge。
> 现在怎么跑：仓库 [README.md](README.md)、[docs/g1_sdk_loco_dodge_runbook.md](docs/g1_sdk_loco_dodge_runbook.md)。
>
> 这是 **执行级别的计划文档**（不是教程）。基于：
> - 已实现：LiDAR UDP forwarder 链路、RealSense+YOLO 链路、sim2sim 验证、`deploy_dodge_real.py` 4 层安全网。
> - 现状：机器人被防摔架吊起，dodge policy 已用真实 YOLO 数据在 sim 中通过两个完整 DODGE→RETURN 周期。
> - 未做：真机的 zero-torque → stand → walk → dodge 全链路点亮。
>
> 详细教程见 [DEPLOY_REAL_G1.md](DEPLOY_REAL_G1.md)。

---

## 1. 依赖盘点（Unitree G1 + 第三方开源）

| 项目 | 用途 | 我们用了多少 | 状态 |
|---|---|---|---|
| [`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python) | DDS lowcmd/lowstate、VideoClient、MotionSwitcher | 全部用，pinned 在 `third_party/` | ✅ 验证 |
| [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) | 仓库基线：`deploy_real.py` + g1.yaml + motion.pt locomotion 策略 | `deploy_dodge_real.py` extends 它 | ✅ sim2sim 验证 |
| [`unitree_sdk2`](https://github.com/unitreerobotics/unitree_sdk2) (C++) | 同上的 C++ 版（更全） | 用其 IDL/API 头文件做参考；C++ binary `livox_ros_driver2_node` 在机器人侧用 | 间接用 |
| [`livox_ros_driver2`](https://github.com/Livox-SDK/livox_ros_driver2) | Mid-360 ROS2 驱动 | 机器人侧已编译 binary，跑 1 次握手让 LiDAR 进入推流态 | ✅ 验证 |
| [`Livox-SDK2`](https://github.com/Livox-SDK/Livox-SDK2) | C 库（`liblivox_lidar_sdk_shared.so`） | 机器人侧 `/usr/local/lib/`，间接通过 ROS driver | ✅ 验证 |
| [`librealsense2`](https://github.com/IntelRealSense/librealsense) + `pyrealsense2` | D435i 深度采集 | 机器人侧装好（in `/opt/ros/noetic/`），通过 `LD_LIBRARY_PATH` 用 | ✅ 验证 |
| [`ultralytics`](https://github.com/ultralytics/ultralytics) (YOLOv8m + ByteTrack) | 人检测 + 多帧 ID 跟踪 | 笔记本侧 | ✅ 验证 |
| `cyclonedds` (随 unitree_sdk2py 一起) | DDS transport | 用于 `rt/yolo/person` 等自定义 topic | ✅ 验证 |
| Unitree 官方 `videohub_pc4` 服务 | 机器人自带 RGB JPEG RPC 服务 | **冲突**：占用 `/dev/video4`，必须 `mscli stopservice` 让 pyrealsense2 接管 | ⚠️ 注意 |
| Unitree 官方 `master_service`（运控板） | 实时电机控制 + IMU 发布 `rt/lowstate` | 我们的命令通过 `rt/lowcmd` 给它 | ✅ 验证 |

**不用的**（可能容易混淆）：
- `unitree_legged_sdk`（Go1 老 SDK，G1 不用）
- `rt/utlidar/voxel_map` 系列 topic（Go2 的协议，G1 上不存在 —— [details](docs/lidar_how_it_works.md)）

---

## 2. 当前能力 vs 缺口

### ✅ 已经验证的（不需要重做）
1. **传感器链路**：LiDAR 点云 / RealSense 深度+RGB 都能稳定流到笔记本，深度校准过（-0.2m 偏置）
2. **感知链路**：YOLO + ByteTrack + bbox 内中位深度 + 单目几何 → world-frame (x_fwd, y_left, dist, bearing) 发到 `rt/yolo/person`
3. **决策链路**：dodge policy + return head 在 sim 上对真实 YOLO 数据 (live + replay) 都跑通了两个完整 DODGE→RETURN 周期，机器人 z 全程稳在 0.78m
4. **安全网**：`deploy_dodge_real.py` 已经包含
   - YOLO staleness 看门狗 (>0.5s 旧 → 视为无检测)
   - IMU 稳定性闸门 (|roll/pitch| > 17° → cmd 强制清零)
   - 速度斜率限制 (Δlin ≤ 0.04 m/s/tick, Δang ≤ 0.10 rad/s/tick → 加速度 ≤ 2 m/s²)
   - 速度上限 (默认 0.08 m/s)
   - 手臂硬锁 (arm_waist_kp=300 N·m/rad)
   - SELECT 急停
5. **滤波器链**：KF (CV 模型 + 自适应 R + Mahalanobis 4σ 门) 已经在 sim 中验证能压住 close-range YOLO+depth 噪声爆发

### ❌ 还没做的（必须按顺序解决）
1. **未在真机上跑过任何完整流程** — 哪怕是单纯 stand
2. **`deploy_dodge_real.py` 的 YOLO 路径没在真机上验证过** — 只在 sim 里跑过
3. **KF 自适应 R 还没移植到 real_robot.py** — sim 用了，real 里还是 staleness-only 的简单 detector
4. **没有 fallback 切换** —— 如果 RealSense 挂了，应自动切到 LiDAR；现在没接好
5. **`record_replay_stitch.py` 一些边缘 case** — 譬如全程无检测时 wrapper 退出且 err 被吞，可观测性不好
6. **机器人侧 publisher 是 throw-away**（放在 `/tmp/`，每次重启都丢失）— 没有 systemd unit 持久化

---

## 3. 部署计划（按 GO/NO-GO 门控）

### Phase 0: 部署前准备（不动机器人，一次性 30 分钟）

- [ ] 备份当前 git 状态：`git status` 干净、可 revert
- [ ] 笔记本电量 > 80%，紧急 kill 渠道（pkill/Ctrl-C）熟悉
- [ ] 急停遥控器**电池满**，SELECT 键反应灵敏（按下不松）
- [ ] 防摔架挂稳，**机器人脚距地 ≥ 5cm**（摔下来不会脚先着地）
- [ ] 工作区 3m 范围内**没有人也没有杂物**
- [ ] 把 `deploy_dodge_real.py` 默认值再降一档（保守版起步）：
  - `--max_vel 0.05`（默认 0.08 → 0.05）
  - `--safety_dist 2.0`（默认 2.5 → 2.0 触发更晚）
  - `--max_tilt_deg 12`（默认 17° → 12° 更敏感）

**Exit gate**: 所有上面打 ✓

---

### Phase 1: 传感器单元验证（不动机器人，10 分钟）

```bash
# 1.1 LiDAR
./scripts/start_lidar.sh
uv run python test_lidar.py --duration 10
# 期望: 2 kHz 包率，~200k 点率

# 1.2 RealSense + YOLO bridge
ssh unitree@192.168.123.164 "/unitree/sbin/mscli stopservice video_hub_pc4"
ssh unitree@... "nohup env LD_LIBRARY_PATH=/opt/ros/noetic/lib/aarch64-linux-gnu \
    python3 /tmp/rgbd_publisher.py > /tmp/rgbd_pub.log 2>&1 &"
# 笔记本上短跑 10s 看 DDS rt/yolo/person
uv run python scripts/yolo_to_dds_laptop.py --duration 10
```

**Exit gate**:
- [ ] LiDAR `nearest_obstacle.py` 看你站在 1-3m 处距离合理
- [ ] YOLO bridge `det=`% 在你画面里时 > 80%；不在画面时 0%
- [ ] DDS topic `rt/yolo/person` 用一个 subscriber 能收到 ~3.2 Hz 消息

---

### Phase 2: Locomotion 基线（机器人 zero-torque → stand → walk）

**只用 unitree_rl_gym 原版 `deploy_real.py`，不接 dodge。**

```bash
uv run python third_party/unitree_rl_gym/deploy/deploy_real/deploy_real.py \
    eno1 configs/g1.yaml
# START → 站姿（2 秒）
# A → 走路（遥控器控制）
# SELECT → 急停
```

**逐项过门**：
- [ ] zero-torque 状态机器人挂在架子里**不抖动**（关节都松着）
- [ ] START 后机器人**平滑**升到默认站姿（不甩腿）
- [ ] **站 30s 不晃**（z 在 IMU 里基本恒定）
- [ ] A 后左摇杆**轻推前进**有反应，速度 < 0.1 m/s（你手动控制摇杆量）
- [ ] **侧移 0.1 m/s 稳定**（这一项关键 — dodge 主要靠侧移）
- [ ] SELECT 任何时候能立刻进 zero-torque（**测 3 次**）

**Exit gate**: 所有 ✓。任何一项失败就 STOP，根因排查。

---

### Phase 3: sim2sim 用真实当前 YOLO 数据验证

**目的**：确认现场环境（光照、背景、相机角度）下，sim 里 dodge 行为符合预期。这一步**不动真机马达**。

```bash
# 同时跑：bridge 录 + sim replay + 拼视频
uv run --with "pillow==9.5.0" --with "ultralytics==8.4.51" --with "lap" \
    --with "imageio==2.35.1" --with "imageio-ffmpeg==0.5.1" \
    python scripts/record_replay_stitch.py \
        --duration 50 \
        --out /tmp/preflight.mp4 \
        --kf-gate 4.0 --kf-meas-std-close 0.30
```

你按剧本走：**站 2m → 走近 0.5m → 退 3m → 再走近 → 退**。

**Exit gate**:
- [ ] 视频里 sim G1 触发**至少 2 次** DODGE
- [ ] 每次 DODGE 后**完整 RETURN 收敛**（控制台显示 RESET）
- [ ] Sim 期间 G1 没倒（z > 0.7m 全程）
- [ ] 红球（YOLO 检测）**没明显瞬移**

---

### Phase 4: 真机 Dodge — Stand Mode（max_vel=0，最保守）

**目的**：在真机上跑 `deploy_dodge_real.py`，但 max_vel=0 让 dodge "想动但不能动"。验证：obs/cmd 通路、不会乱发命令、IMU 稳定性闸门工作。

```bash
# 终端 1：保持 RGBD publisher 在跑
# 终端 2：YOLO bridge
uv run --with ... python scripts/yolo_to_dds_laptop.py
# 终端 3：deploy_dodge_real with max_vel=0
uv run python deploy_dodge_real.py eno1 g1.yaml \
    --source yolo \
    --max_vel 0.0 \
    --safety_dist 2.0 \
    --max_tilt_deg 12
```

操作：START → 站 → A → 进 dodge mode → 你走近到 1.5m → 看终端日志

**Exit gate**:
- [ ] 日志显示 `[DODGE START]` 在你走进 2m 内时
- [ ] 但机器人**完全不动**（max_vel=0 把 cmd 卡死成 0）
- [ ] 持续 30s 你走来走去，机器人 z 保持稳定
- [ ] SELECT 仍然能急停
- [ ] 没有 `[UNSTABLE]` 警告（除非你故意撞架子）

---

### Phase 5: 真机 Dodge — Minimal Motion（max_vel=0.03 m/s）

**目的**：真正让机器人动**一点点**。3cm/s 即使方向错也撞不出问题（防摔架吊着、加速度 2 m/s² 上限 → 实际最大行程 < 5cm）。

```bash
uv run python deploy_dodge_real.py eno1 g1.yaml \
    --source yolo \
    --max_vel 0.03 \
    --safety_dist 2.5 \
    --max_tilt_deg 12 \
    --max_dcmd_lin 0.02
```

**Exit gate**:
- [ ] 一次 DODGE 周期内机器人**没倒**
- [ ] 侧移方向**正确**（远离你而不是冲向你）
- [ ] RETURN 后机器人**接近原位**（架子吊着位移可能不到 5cm 也算正常）
- [ ] 整个过程**没有抖手抖脚**（手臂托住）

**如果不过**：根因分析。最常见：
- 方向反 → IMU quat 顺序问题 ([w,x,y,z] vs [x,y,z,w])
- 抖动 → max_dcmd 还太大，降到 0.01

---

### Phase 6: 真机 Dodge — 慢速正常动作（max_vel=0.08 m/s）

```bash
uv run python deploy_dodge_real.py eno1 g1.yaml \
    --source yolo \
    --max_vel 0.08 \
    --safety_dist 2.0 \
    --max_tilt_deg 15
```

**Exit gate**:
- [ ] 多次 dodge 都成功
- [ ] 你的快速冲撞（1 m/s 走近）也能稳住（如果倒，加大 safety_dist）
- [ ] 至少 5 个 DODGE→RETURN 周期连续成功

---

### Phase 7: 接近训练分布（max_vel=0.20，safety=1.5m）

只在 Phase 6 完美通过后做。这一步开始接近 dodge policy 训练时的速度分布。

```bash
uv run python deploy_dodge_real.py eno1 g1.yaml \
    --source yolo \
    --max_vel 0.20 \
    --safety_dist 1.5 \
    --max_tilt_deg 17
```

**Exit gate**：训练全速 (0.5 m/s) 之前需要严格风险评估。**架子能不能挡住一次 0.5 m/s 速度下的剪切力？不能就别上。**

---

## 4. 风险登记表

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 机器人摔倒（架子挂着，腿弯折受损） | 中 | 高 | Phase 4-5 max_vel 极小；架子调到刚好抗住自由落体 |
| RealSense 突然断连 → YOLO 沉默 | 中 | 中 | staleness 看门狗 0.5s → cmd 自动归零（已实现） |
| YOLO 把背景物当人，dodge 乱触发 | 中 | 中 | Phase 1 测试时校验 det% 行为；prod 用 `--conf 0.40+` |
| IMU 抖动触发假阳性 unstable | 低 | 低 | max_tilt 调高，或加滞后 |
| LiDAR forwarder 死了没察觉 | 中 | 低 | 不用 LiDAR 不影响（已切到 YOLO） |
| 笔记本崩溃 / 网线掉 → 控制断 | 低 | 极高 | **物理急停按钮在手**；DDS 心跳监控；任何情况下机器人无 cmd 输入应自动 zero-cmd |
| dodge policy 对侧身/蹲下/小孩误判距离 | 中 | 中 | 训练数据偏成人正面，先在受控环境只用成人测试 |
| 防摔架本身震动影响 IMU | 低 | 低 | Phase 4 静态时检查 IMU 噪声 |

---

## 5. Go/No-Go 决策树

```
Phase 0 (准备) → [失败] STOP，补齐
   ↓ ✓
Phase 1 (传感器) → [失败] 检查 publisher/bridge，机器人是否上电
   ↓ ✓
Phase 2 (locomotion) → [失败] 跑 unitree_rl_gym 自己的 deploy_real，policy 本身没问题再说
   ↓ ✓
Phase 3 (sim2sim 真数据) → [失败] dodge policy 对你这个环境的数据失败，需要重训
   ↓ ✓
Phase 4 (real, max_vel=0) → [失败] obs/cmd 通路问题，根因分析
   ↓ ✓
Phase 5 (real, max_vel=0.03) → [失败] 方向反 / 抖动，调 IMU 或斜率
   ↓ ✓
Phase 6 (real, max_vel=0.08) → 这就算 "demo 可用"
   ↓ (谨慎)
Phase 7 (real, max_vel=0.20)
```

---

## 6. 缺口补齐 backlog（在真机部署之前可以做）

按优先级：

| # | 缺口 | 工作量 | 何时做 |
|---|---|---|---|
| 1 | ~~把 KF (自适应 R + Mahalanobis 门) 从 sim 移植到 `deploy_dodge_real.py` 的 YoloDdsObstacleDetector~~ **✅ 完成 (2026-05-17)** | 1h | ~~Phase 4 之前 **必做**~~ Done |
| 2 | `rgbd_publisher` 写成 systemd unit，重启不丢 | 30min | Phase 6 之前 |
| 3 | LiDAR 路径作为 YOLO 失效时的 fallback（两条 source 并行，谁先有 fresh msg 就用谁）| 2h | Phase 7 之前 |
| 4 | 命令行实时绘制 dist/bearing 曲线（便于排查触发不及时）| 1h | 可选 |
| 5 | record_replay_stitch.py 加入异常处理：YOLO 全程无检测时不退出，发空 npy 并跳过 sim | 30min | Phase 3 之前可改 |
| 6 | `--source both` 在 `deploy_dodge_real.py`：同时订阅两个 source，取更近的 | 2h | Phase 7 时考虑 |

---

## 7. 关键文件对照

| 文件 | 角色 | Phase |
|---|---|---|
| `scripts/start_lidar.sh` | LiDAR 链路 bringup | 1, 6 |
| `scripts/rgbd_publisher_robot.py` | 机器人侧 RGB+D TCP 发布 | 1, 4-7 |
| `scripts/yolo_to_dds_laptop.py` | 笔记本 YOLO+depth → DDS bridge | 1, 4-7 |
| `scripts/record_replay_stitch.py` | sim2sim 用真实 YOLO 数据 | 3 |
| `deploy_dodge_mujoco.py` | sim 验证（含 KF replay） | 3 |
| `deploy_dodge_real.py` | 真机 dodge 部署 | 4-7 |
| `third_party/unitree_rl_gym/deploy/deploy_real/deploy_real.py` | locomotion 基线（不含 dodge） | 2 |
| `test_lidar.py`, `nearest_obstacle.py` | LiDAR 工具 | 1 |
| `DEPLOY_REAL_G1.md` | 详细教程 | 全程参考 |
| `docs/lidar_*.md` | LiDAR 深入 | 故障时参考 |
| `g1_camera_dump/README.md` | 相机/RealSense 细节 | 故障时参考 |

---

## 8. 第一次真机部署 day-of checklist（建议打印）

```
[ ] 防摔架检查 - 螺丝紧、钢丝绳没磨损、机器人脚距地 5cm
[ ] 急停按钮在手 - 测 3 次 SELECT 进 zero-torque
[ ] 网线接好 - ping 192.168.123.164 通
[ ] 笔记本电量 > 80%
[ ] 工作区 3m 无人无障碍
[ ] 录像设备 ON（手机/桌面录屏，留证据复盘）
[ ] git 当前 commit 记下: ____________
[ ] Phase 1 通过：传感器 OK
[ ] Phase 2 通过：locomotion stand OK
[ ] Phase 3 通过：sim2sim 现场数据 OK
[ ] Phase 4 通过：real max_vel=0 OK
[ ] Phase 5 通过：real max_vel=0.03 OK
[ ] (今天到此为止？停下来 review 录像、写笔记)
[ ] Phase 6 (max_vel=0.08) → 下一天再做
```

---

最后更新：2026-05-17
