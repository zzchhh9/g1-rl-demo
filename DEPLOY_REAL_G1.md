# G1 真机部署指南：Dodge+Return 策略

## Checkpoint 清单

clone 之后所有文件已经在仓库里，不需要单独下载：

```
g1-rl-demo/
├── checkpoints/
│   ├── dodge_v23b_54400.pt           ← dodge 策略 (1.4MB, transformer 18→3)
│   └── return_head_v23b_v6.pt        ← 回位策略 (390KB, MLP 2→3)
├── deploy/
│   ├── dodge_policy.py               ← 推理代码 (只依赖 torch + numpy)
│   ├── lidar_sim.py                  ← LiDAR 仿真 (sim2sim 用)
│   └── safety.py                     ← 安全模块
├── deploy_dodge_mujoco.py            ← sim2sim 验证脚本
├── deploy_dodge_real.py              ← 真机 dodge 部署脚本 ⭐
├── test_lidar.py                     ← LiDAR 测试工具 (Livox Mid-360 UDP 直连) ⭐
├── nearest_obstacle.py               ← 独立的实时最近障碍物距离工具 ⭐
├── scripts/
│   ├── start_lidar.sh                ← 机器人侧 LiDAR bringup（重启后必跑）
│   └── stop_lidar.sh                 ← 停止 LiDAR 数据流
├── docs/
│   ├── lidar_how_it_works.md         ← Mid-360 原理 + 协议详解
│   └── lidar_nearest_obstacle.md     ← 实时距离测量实现细节
├── third_party/unitree_rl_gym/
│   ├── deploy/pre_train/g1/motion.pt ← locomotion (12-DOF LSTM，仓库自带)
│   └── deploy/deploy_real/           ← 真机遥控行走脚本
└── policy_lstm_1.pt                  ← 备用 locomotion checkpoint
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
# 无头模式运行（30 秒，障碍物 0.30 m/s）
MUJOCO_GL=egl uv run python deploy_dodge_mujoco.py --duration 30 --obstacle_speed 0.30

# 录制视频
MUJOCO_GL=egl uv run python deploy_dodge_mujoco.py \
  --record sim2sim_test.mp4 --duration 30 --obstacle_speed 0.30
```

**必须看到以下输出才算通过：**
```
✓ IDLE → DODGE (LiDAR) → RETURN → STOP
```

**如果看到 `✗`**：
- 检查 checkpoint 是否正确加载（第一步的验证要通过）
- 尝试加长时间 `--duration 40`
- 检查 `uv sync` 是否成功

**看视频确认：**
- [ ] 机器人全程站稳不倒（z ≈ 0.78）
- [ ] 红球靠近时机器人侧移躲避
- [ ] 红球远离后机器人回到原位（disp < 0.2m）
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

**重要**：G1 上 **没有** `rt/utlidar/voxel_map` 这种 DDS topic（那是 Go2 的协议）。我们走的是 **Livox Mid-360 UDP 直连**，不经任何 Unitree 服务。详情见 [docs/lidar_how_it_works.md](docs/lidar_how_it_works.md)。

**数据链路：**

```
Mid-360 (.120) ─UDP─► Robot Jetson (.164) ─UDP forward─► Laptop (.222)
```

机器人侧只跑两样东西，都在 `/tmp/livox-run/`（重启清零）：
1. 一次性 driver 启动（告诉 LiDAR "把点云发到 .164"）
2. 一个 30 行的 Python UDP 转发器（持续把 .164:56301/56401/56201 → .222 同端口）

**一次性 bringup（机器人重启后必须重做）：**

```bash
# 需要先装 sshpass
sudo apt install -y sshpass

# 启动整套 LiDAR 流水线
./scripts/start_lidar.sh
# 输出:
#   [1/4] preparing /tmp/livox-run on robot ...
#   [2/4] uploading MID360 config + forwarder + start scripts ...
#   [3/4] running driver briefly to tell LiDAR to start streaming ...
#       ✓ LiDAR streaming started
#   [4/4] starting UDP forwarder ...
#   ✓ LiDAR pipeline up.
```

**笔记本端测试：**

```bash
# 测试 1: 基础连接 — 检查是否收到 UDP 包并能解析
uv run python test_lidar.py
# 预期:
#   ✅ 通过: 收到 ~13000 包，累计 ~1.2M 点
#   包速率: ~2090 pkt/s   (Mid-360 典型 ~2000)
#   点速率: ~200000 pt/s  (典型 ~200k)

# 测试 2: 障碍物检测 — 让人站在机器人前方 1-3m
uv run python test_lidar.py --detect

# 测试 3: 实时最近障碍物距离（独立工具，不依赖 deploy_dodge_real.py）
uv run python nearest_obstacle.py
# 输出:
#   t=12.3s  pkts=24580  最近: 1.83m  方位=+15.4°  (x=+1.76, y=+0.48, z=+0.95)  n=312
```

**关闭：**
```bash
./scripts/stop_lidar.sh
```

**日常重连 — 第二次以后用这个流程（非首次部署）：**

第一次部署后，机器人/笔记本之间的状态可能是：
- 转发器还活着（短时间离开） → 直接跑工具即可
- 机器人重启了 / 转发器被 kill / 长时间断开 → 需要重新 bringup

判断哪种情况，**一行 UDP 探测**：

```bash
timeout 2 python3 -c "import socket; s=socket.socket(2,2); s.bind(('',56301)); s.settimeout(1.5); n=0
try:
  while True: s.recvfrom(2048); n+=1
except: pass
print('packets/1.5s =', n)"
```

- 输出 `packets/1.5s = 几千` → 数据流还在，直接 `uv run python nearest_obstacle.py`
- 输出 `packets/1.5s = 0` → 流断了，按下面 4 步恢复：

```bash
# 1. sshpass 已装则跳过
sudo apt install -y sshpass

# 2. 网络确认
ping -c 2 192.168.123.164

# 3. 重新启动数据流
./scripts/start_lidar.sh

# 4. 跑工具
uv run python nearest_obstacle.py
```

**验证清单：**
- [ ] `test_lidar.py` 基础测试通过（包率 ~2 kHz，点率 ~200k/s）
- [ ] `nearest_obstacle.py` 输出稳定的距离/方位读数
- [ ] `--detect` 测试能检测到 1-3m 内的人
- [ ] 障碍物坐标合理（x=前方为正, y=左侧为正, z>0.3 在地面以上）

**如果完全收不到数据**（上面 UDP 探测 = 0 且 `start_lidar.sh` 已重跑过仍无效）：

1. 笔记本到机器人连通性：`ping 192.168.123.164`
2. 转发器是否在机器人上活着：
   ```bash
   sshpass -p 123 ssh unitree@192.168.123.164 'pgrep -af forward.py'
   ```
3. LiDAR 本体是否还在线：`ping 192.168.123.120`
4. driver 启动日志：
   ```bash
   sshpass -p 123 ssh unitree@192.168.123.164 'tail -30 /tmp/livox-run/driver.log'
   ```
   预期看到 `successfully change work mode`；如果是 `bad_alloc caught` 则机器人侧 DDS 撑爆，重启机器人

**深入排查见**：[docs/lidar_how_it_works.md](docs/lidar_how_it_works.md)、[docs/lidar_nearest_obstacle.md](docs/lidar_nearest_obstacle.md)

---

## 第五步：集成 Dodge 到真机（低速）

在 `deploy_real.py` 的 `Controller.run()` 中，把遥控器命令替换为 dodge 策略输出：

**已写好的脚本 `deploy_dodge_real.py`（直接用，不用改 deploy_real.py）：**

```bash
# 第一次测试（超保守：0.10 m/s，2.5m 触发距离）
uv run python deploy_dodge_real.py eth0 configs/g1.yaml

# 自定义速度和距离
uv run python deploy_dodge_real.py eth0 configs/g1.yaml --max_vel 0.15 --safety_dist 2.0
```

操作流程和 deploy_real.py 一样：START → 站起 → A → 进入 dodge 模式。
遥控器正常控制走路，障碍物靠近时自动切换到 dodge 策略。

---

## 第六步：逐步提速

**每一步都要确认安全后再进下一步：**

| 轮次 | MAX_LIN_VEL | safety_distance | 障碍物（人）速度 | 预期 |
|---|---|---|---|---|
| 0 | **0.10 m/s** | **2.5m** | 人站着不动，缓慢靠近 | 确认方向正确，手臂不晃 |
| 1 | 0.15 m/s | 2.0m | 人慢走 0.1 m/s | 机器人缓慢侧移 |
| 2 | 0.25 m/s | 1.5m | 人正常走 0.2 m/s | 机器人明显侧移 |
| 3 | 0.35 m/s | 1.0m | 人正常走 0.3 m/s | 接近训练配置 |
| 4 | 0.50 m/s | 0.6m | 人快走 0.3 m/s | 全速（训练配置） |

> ⚠️ **手臂安全**：locomotion 只控制 12-DOF 腿部，手臂由 PD 控制器锁定在默认位置
> （`arm_waist_kps` 最高 300 N·m/rad）。如果发现手臂抖动或晃动，立即停止并降低
> locomotion 速度。手臂不参与 dodge 动作。

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
- [ ] LiDAR 数据正常（`./scripts/start_lidar.sh` 已跑、`test_lidar.py` 通过）
- [ ] MAX_LIN_VEL 从 0.10 开始

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
| LiDAR 没有数据 | 转发器没起 / driver 没握手 | `./scripts/start_lidar.sh` 重做 bringup |
| LiDAR 检测不到人 | 反射率低 / 站位太远 | 站在 1-3m 内，穿亮色衣服；调 `nearest_obstacle.py --min-h 0.2` |
| LiDAR 数据延迟大 | 累积窗口太大 | `nearest_obstacle.py --window 0.05`（更快但更稀疏） |
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
