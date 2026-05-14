# G1 真机部署指南：Dodge+Return 策略

## Checkpoint 清单

clone 之后所有文件已经在仓库里，不需要单独下载：

```
g1-rl-demo/
├── checkpoints/
│   ├── dodge_v23b_54400.pt        ← dodge 策略 (1.4MB, transformer 18→3)
│   └── return_head_v23b_v6.pt     ← 回位策略 (390KB, MLP 2→3)
├── deploy/
│   ├── dodge_policy.py            ← 推理代码 (只依赖 torch + numpy)
│   ├── lidar_sim.py               ← LiDAR 仿真 (sim2sim 用，真机不用)
│   └── safety.py                  ← 安全模块
├── deploy_dodge_mujoco.py         ← sim2sim 验证脚本
├── third_party/unitree_rl_gym/
│   ├── deploy/pre_train/g1/motion.pt    ← locomotion (12-DOF LSTM，仓库自带)
│   └── deploy/deploy_real/deploy_real.py ← 真机遥控行走脚本
└── policy_lstm_1.pt               ← 备用 locomotion checkpoint
```

---

## 第一步：克隆仓库 + 环境准备

```bash
git clone https://github.com/zzchhh9/g1-rl-demo.git
cd g1-rl-demo
git submodule update --init --recursive

# 用 uv 安装依赖
# 注意：pyproject.toml 里有 isaacgym 依赖（训练用），deploy 不需要。
# 如果 uv sync 报错找不到 isaacgym，创建空目录绕过：
mkdir -p third_party/isaacgym/python && \
  echo -e '[project]\nname="isaacgym"\nversion="0.0.0"' > third_party/isaacgym/python/pyproject.toml

uv sync

# 验证 checkpoint 能加载
uv run python -c "
import torch
d = torch.load('checkpoints/dodge_v23b_54400.pt', map_location='cpu', weights_only=False)
print(f'Dodge policy: {len(d[\"model_state_dict\"])} keys ✅')
r = torch.load('checkpoints/return_head_v23b_v6.pt', map_location='cpu', weights_only=False)
print(f'Return head: {len(r[\"model_state_dict\"])} keys ✅')
l = torch.jit.load('third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt')
print(f'Locomotion: TorchScript loaded ✅')
"
```

预期输出：
```
Dodge policy: 48 keys ✅
Return head: 4 keys ✅
Locomotion: TorchScript loaded ✅
```

---

## 第二步：sim2sim 验证（必须通过再接真机）

```bash
# 无头模式运行
MUJOCO_GL=egl uv run python deploy_dodge_mujoco.py --duration 25

# 或者录制视频
MUJOCO_GL=egl uv run python deploy_dodge_mujoco.py --record sim2sim_test.mp4 --duration 25
```

**必须看到以下输出才算通过：**
```
✓ IDLE → DODGE (LiDAR) → RETURN → STOP
```

**如果看到 `✗`**：检查 checkpoint 是否正确加载、PYTHONPATH 是否设对。

**看视频确认：**
- [ ] 机器人全程站稳不倒（z ≈ 0.78）
- [ ] 红球靠近时机器人侧移躲避
- [ ] 红球远离后机器人回到原位
- [ ] 全程机器人和红球不接触

---

## 第三步：单独测试 locomotion（遥控器走路）

**这一步是安全底线 — locomotion 走不稳的话，绝对不能加 dodge。**

```bash
# 用仓库自带的 deploy_real.py（通过 uv 运行）
uv run python third_party/unitree_rl_gym/deploy/deploy_real/deploy_real.py eth0 configs/g1.yaml
# eth0 替换为你的网卡名（用 ip link show 查看）
# configs/g1.yaml 在 third_party/unitree_rl_gym/deploy/deploy_real/configs/ 下
```

**操作流程：**
1. 遥控器按 **START** → 退出零力矩状态
2. 机器人自动移到默认站姿（2 秒过渡）
3. 按 **A** → 开始行走控制
4. **左摇杆**：上=前进，下=后退，左右=侧移
5. **右摇杆**：左右=转向
6. 按 **SELECT** → **紧急停止**（任何时候都能按）

**逐项验证：**
- [ ] 站稳 30 秒不晃
- [ ] 前进 0.3 m/s 稳定（左摇杆轻推）
- [ ] **侧移 0.3 m/s 稳定**（最关键 — dodge 需要快速侧移）
- [ ] 转向 0.5 rad/s 稳定
- [ ] 松开摇杆后机器人停下不倒
- [ ] SELECT 键能立即停止

---

## 第四步：LiDAR 验证

> ⚠️ **注意：真机 LiDAR 集成代码需要你自己写。**
> 仓库里的 `deploy/lidar_sim.py` 是 MuJoCo 仿真用的，不能直接用于真机。
> 真机需要：Livox SDK → 点云 → 障碍物检测 → body frame 坐标。

**Livox Mid-360 驱动启动：**

```bash
# 方案 A: ROS2（推荐）
ros2 launch livox_ros_driver2 msg_MID360_launch.py
ros2 topic echo /livox/lidar --once  # 确认有 PointCloud2 数据

# 方案 B: Livox SDK 直连
# 参考 https://github.com/Livox-SDK/Livox-SDK2
```

**验证清单：**
- [ ] LiDAR 有数据输出（topic 有消息）
- [ ] 让人站在 2m 外，能在点云中看到
- [ ] 更新频率 ~10 Hz

**你需要写的代码（约 50 行）：**

```python
# 伪代码 — 真机 LiDAR 障碍物检测
def detect_obstacle_from_lidar(point_cloud, robot_pos, robot_yaw):
    # 1. 过滤地面点 (z < 0.3m)
    points = point_cloud[point_cloud[:, 2] > 0.3]
    # 2. 过滤远距离点 (> 5m)
    dists = np.linalg.norm(points[:, :2] - robot_pos[:2], axis=1)
    points = points[dists < 5.0]
    # 3. 取最近点的质心作为障碍物位置
    if len(points) == 0:
        return None
    centroid = points.mean(axis=0)
    return centroid  # [x, y, z] world frame
```

---

## 第五步：集成 Dodge 到真机（低速）

在 `deploy_real.py` 的 `Controller.run()` 中，把遥控器命令替换为 dodge 策略输出：

```python
# 在 Controller.__init__ 中加载 dodge policy:
from deploy.dodge_policy import DodgePolicy
self.dodge = DodgePolicy('checkpoints/dodge_v23b_54400.pt')
self.dodge_active = False
self.dodge_start_pos = None
self.dodge_start_yaw = 0.0
self.safety_distance = 2.0  # 第一次测试用 2m，留足反应时间
self.MAX_LIN_VEL = 0.15     # 第一次只给 0.15 m/s !!!

# 在 Controller.run() 中，替换 self.cmd 的来源：
obstacle_pos = detect_obstacle_from_lidar(...)  # 你写的 LiDAR 检测
if obstacle_pos is not None:
    dist = np.linalg.norm(robot_pos[:2] - obstacle_pos[:2])
else:
    dist = float('inf')

if dist < self.safety_distance and not self.dodge_active:
    self.dodge_active = True
    self.dodge_start_pos = robot_pos[:2].copy()
    self.dodge_start_yaw = robot_yaw
    self.dodge.reset(robot_pos[:2], robot_yaw)

if self.dodge_active:
    if dist < self.safety_distance + 0.1:
        obs18 = self.dodge.build_obs(robot_pos, robot_yaw, obstacle_pos)
        vel = self.dodge.get_velocity_command(obs18)
        self.cmd[0] = np.clip(vel[0], -self.MAX_LIN_VEL, self.MAX_LIN_VEL)
        self.cmd[1] = np.clip(vel[1], -self.MAX_LIN_VEL, self.MAX_LIN_VEL)
        self.cmd[2] = np.clip(vel[2], -0.3, 0.3)
    else:
        # 障碍物远离 → 回位
        disp_w = robot_pos[:2] - self.dodge_start_pos
        yaw_err = robot_yaw - self.dodge_start_yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        # 简单 P 控制器回位
        self.cmd[0] = np.clip(-2.0 * disp_w[0], -self.MAX_LIN_VEL, self.MAX_LIN_VEL)
        self.cmd[1] = np.clip(-2.0 * disp_w[1], -self.MAX_LIN_VEL, self.MAX_LIN_VEL)
        self.cmd[2] = np.clip(-2.0 * yaw_err, -0.3, 0.3)
        if np.linalg.norm(disp_w) < 0.2 and abs(yaw_err) < 0.15:
            self.dodge_active = False
else:
    # 正常遥控器控制
    self.cmd[0] = self.remote_controller.ly
    self.cmd[1] = self.remote_controller.lx * -1
    self.cmd[2] = self.remote_controller.rx * -1
```

---

## 第六步：逐步提速

**每一步都要确认安全后再进下一步：**

| 轮次 | MAX_LIN_VEL | safety_distance | 障碍物（人）速度 | 预期 |
|---|---|---|---|---|
| 1 | **0.15 m/s** | 2.0m | 人慢走 0.1 m/s | 机器人缓慢侧移 |
| 2 | 0.25 m/s | 1.5m | 人正常走 0.2 m/s | 机器人明显侧移 |
| 3 | 0.35 m/s | 1.0m | 人正常走 0.3 m/s | 接近训练配置 |
| 4 | 0.50 m/s | 0.6m | 人快走 0.3 m/s | 全速（训练配置） |

**每轮之间必须确认：**
- [ ] 机器人没有倒
- [ ] 机器人朝正确方向移动（远离障碍物）
- [ ] 障碍物远离后机器人停下或回位
- [ ] 没有异常抖动

---

## 安全检查清单（每次测试前）

- [ ] **急停按钮在手边**（遥控器 SELECT 键）
- [ ] 工作区域清空（无杂物、无其他人）
- [ ] 电量 > 50%
- [ ] 先跑 30 秒纯站立确认稳定
- [ ] LiDAR 数据正常
- [ ] MAX_LIN_VEL 从 0.15 开始

**立即按 SELECT 的情况：**
- 机器人明显倾斜 > 15°
- 机器人向障碍物方向移动（应该远离）
- 任何异常抖动或震荡
- LiDAR 丢失超过 1 秒

---

## 故障排除

| 问题 | 原因 | 解决 |
|---|---|---|
| 机器人不动 | `cmd` 没传到 locomotion obs | 打印 `self.cmd`，确认非零 |
| 机器人动了方向反 | body frame 坐标系搞反 | X=前 Y=左，检查 IMU quat 格式 `[w,x,y,z]` |
| 机器人摔倒 | MAX_LIN_VEL 太大 | 降到 0.1 m/s；确认第三步通过 |
| LiDAR 检测不到人 | 驱动没启动 / 反射率低 | 检查 topic；让人穿亮色衣服 |
| 回位不准 | 里程计漂移 | 短时间(< 20s)内漂移应 < 10cm |
| yaw 回不来 | P 增益太小 | 把 yaw P 增益从 2.0 调到 3.0 |

---

## obs 维度速查

### Dodge Policy (18 dim)
```
[0:3]   机器人速度 (body frame)         ← IMU 状态估计
[3:6]   障碍物位置 (body frame)         ← LiDAR 检测
        [5] = -0.48 (硬编码，不用管)
[6:9]   障碍物速度 (body frame)         ← LiDAR 帧间差分
        [8] = 0.0 (硬编码)
[9:11]  位移 (从 dodge 触发点算)        ← 里程计
[11:14] 静态障碍物 (可选，默认全 0)     ← 硬编码
[14]    yaw 偏移                        ← IMU
[15:18] 上一步动作                      ← 内部状态
```

### Locomotion Policy (47 dim)
```
[0:3]   角速度 × 0.25                  ← IMU gyro
[3:6]   重力方向                        ← IMU quat 计算
[6:9]   速度命令 × [2.0, 2.0, 0.25]    ← dodge policy 输出
[9:21]  关节位置 (相对默认) × 1.0       ← 电机编码器
[21:33] 关节速度 × 0.05                ← 电机编码器
[33:45] 上一步动作                      ← 内部状态
[45:47] sin/cos 步态相位                ← 内部计时器 (0.8s 周期)
```
